import logging
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from adet.utils.comm import reduce_sum
from adet.layers import ml_nms, IOULoss, Ranking_Loss
from adet.history import History
import numpy as np

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores

"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    # NOTE: make negtives to zero
    left_right[left_right < 0] = 0

    top_bottom = reg_targets[:, [1, 3]]
    top_bottom[top_bottom < 0] = 0

    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
              (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])

    ctrness = torch.sqrt(ctrness)
    #NOTE: added new lines for nan case
    ctrness[ctrness != ctrness] = 0
    return ctrness


def _divide_list(l, num_loc_list):
    start = 0
    end = 0
    result = ()
    for i in num_loc_list:
        end += i
        result += (l[start:end],)
        start += i
    return result


def compute_confidence_targets(data, rank_target):
    if rank_target == 'entropy':
        return torch.sigmoid(data)


def get_target_margin(instances):
    target1 = instances.gt_ctrs
    target2 = torch.roll(target1, -1)
    # make target pair
    greater = target1 > target2
    greater = greater.to(torch.float)
    less = target1 < target2
    less = less.to(torch.float) * (-1)
    target = greater + less

    # cale margin this is the ctrness margin
    margin = abs(target1 - target2)

    # set pairs not within same img, same level, same instance to 0
    for i in range(len(target) - 1):
        if (instances.im_inds[i] != instances.im_inds[i + 1]) or \
                (instances.fpn_levels[i] != instances.fpn_levels[i + 1]) or \
                (instances.gt_inds[i] != instances.gt_inds[i + 1]):
            target[i] = 0
            margin[i] = 0
    target[-1] = 0

    return target, margin


# box x0,y0,x1,y1
def is_in_box(loc, box):
    if box[0] < loc[0] < box[2] and box[1] < loc[1] < box[3]:
        return 1
    else:
        return 0


def num_loc_per_level(level, box):
    count = 0
    for loc in level:
        count += is_in_box(loc, box)
    return count


def count_loc_num(locations, box):
    count = 0
    for level in locations:
        count += num_loc_per_level(level, box)
    return count


def multi_level_location_per_instance(locations, gt_instances):
    # locations: list of n levels
    # gt_instances: list of n imgs

    for img in gt_instances:
        count = 0
        targets_per_box = []
        for box in img.gt_boxes:
            count += count_loc_num(locations, box)
            targets_per_box.append(count)
            count = 0
        img.total_targets_per_box = targets_per_box


class FCOSOutputs(nn.Module):
    def __init__(self, cfg):
        super(FCOSOutputs, self).__init__()

        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        # NOTE: add ranking loss to ctr
        self.ctr_ranking_loss = Ranking_Loss()

        self.pre_nms_thresh_test = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.FCOS.NMS_TH
        self.thresh_with_ctr = cfg.MODEL.FCOS.THRESH_WITH_CTR

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.history = dict()

    def _divide_list(self, l, num_loc_list):
        start = 0
        end = 0
        for i in num_loc_list:
            end += i
            yield tuple(l[start:end])
            start += i

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            if torch.is_tensor(training_targets[im_i]):
                training_targets[im_i] = torch.split(
                    training_targets[im_i], num_loc_list, dim=0
                )
            else:
                training_targets[im_i] = _divide_list(training_targets[im_i], num_loc_list)

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            if torch.is_tensor(targets_per_level[0]):
                targets_level_first.append(
                    torch.cat(targets_per_level, dim=0)
                )
            else:
                flat_list = []
                for list in targets_per_level:
                    for item in list:
                        flat_list.append(item)
                targets_level_first.append(flat_list)

        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        # NOTE: targets is the gt_instances which as class id
        labels = []
        image_ids = []
        reg_targets = []
        target_inds = []
        target_inds_in_img = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):

            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes
            image_id = targets_per_im.image_id

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            # NOTE: add a new instance, record locations_to_gt_inds
            num_targets_im_stack = locations_to_gt_inds

            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            image_id = [image_id[0]] * len(locations_to_gt_inds)

            labels.append(labels_per_im)
            image_ids.append(image_id)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)
            target_inds_in_img.append(num_targets_im_stack)
        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds,
            "image_ids": image_ids,
            "target_inds_in_img": target_inds_in_img
        }

    def losses(self, logits_pred, reg_pred, ctrness_pred, locations, gt_instances, top_feats=None):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """
        # NOTE: record in history
        for instance in gt_instances:
            if instance.image_id[0] in self.history:
                # TODO: check if this really works
                for anno in self.history[instance.image_id[0]]:
                    anno.max_correctness_update()
            else:
                anno_list = []
                for i in range(len(instance.image_id)):
                    anno_list.append(History(i))
                self.history[instance.image_id[0]] = anno_list

        training_targets = self._get_ground_truth(locations, gt_instances)

        # NOTE: count how many locations per instance
        # multi_level_location_per_instance(locations, gt_instances)

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        instances = Instances((0, 0))
        instances.labels = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["labels"]
        ], dim=0)

        instances.gt_inds = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["target_inds"]
        ], dim=0)

        instances.gt_inds_per_im = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["target_inds_in_img"]
        ], dim=0)



        instances.im_inds = cat([
            x.reshape(-1) for x in training_targets["im_inds"]
        ], dim=0)
        instances.reg_targets = cat([
            # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            x.reshape(-1, 4) for x in training_targets["reg_targets"]
        ], dim=0, )
        instances.locations = cat([
            x.reshape(-1, 2) for x in training_targets["locations"]
        ], dim=0)
        instances.fpn_levels = cat([
            x.reshape(-1) for x in training_targets["fpn_levels"]
        ], dim=0)

        instances.logits_pred = cat([
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred
        ], dim=0, )


        # TODO: just for testing, delete later
        # loss = self.ctr_ranking_loss(logits_pred[0],logits_pred[0])

        instances.reg_pred = cat([
            # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
            x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred
        ], dim=0, )
        instances.ctrness_pred = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.permute(0, 2, 3, 1).reshape(-1) for x in ctrness_pred
        ], dim=0, )

        if len(top_feats) > 0:
            instances.top_feats = cat([
                # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in top_feats
            ], dim=0, )

        flat_list = []
        for list in training_targets["image_ids"]:
            for item in list:
                flat_list.append(item)
        # instances.image_id = flat_list

        save = instances
        result = self.fcos_losses(instances)
        result[0]["instances"].image_ids = [flat_list[i] for i in result[0]["instances"].pos_inds]

        # NOTE: temp added code

        return result

    def fcos_losses(self, instances):
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()

        '''
            The negtive sample part:
            pred_inds: the inds for where cls preds a value
        '''
        thresh = self.pre_nms_thresh_train
        max_logits, _ = instances.logits_pred.max(dim=1)
        bool_inds = max_logits.sigmoid() > thresh
        pred_inds = torch.nonzero(bool_inds).squeeze(1)

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # neg_inds = []
        # for value in pred_inds:
        #     if len(torch.nonzero(pos_inds == value)) == 0:
        #         neg_inds.append(value)
        #
        # neg_inds = torch.tensor(neg_inds, device='cuda')

        combined = torch.cat((pos_inds, pred_inds))
        uniques, counts = combined.unique(return_counts=True)
        comb_inds = uniques


        num_comb_local = comb_inds.numel()
        total_num_comb = reduce_sum(comb_inds.new_tensor([num_comb_local])).item()
        num_comb_avg = max(total_num_comb / num_gpus, 1.0)

        comb_ctrness_targets = compute_ctrness_targets(instances.reg_targets[comb_inds])
        if comb_inds.numel() > 0:
            comb_ctrness_loss = F.binary_cross_entropy_with_logits(
                instances.ctrness_pred[comb_inds],
                comb_ctrness_targets,
                reduction="sum"
            ) / num_comb_avg
        else:
            comb_ctrness_loss = instances.ctrness_pred[comb_inds].sum() * 0




        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        confidence = instances.logits_pred[pos_inds, labels[pos_inds]]

        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        instances.unique_id = instances.fpn_levels*1000000+instances.gt_inds




        if pos_inds.numel() > 0:
            reg_loss = self.loc_loss_func(
                instances.reg_pred,
                instances.reg_targets,
                ctrness_targets
            ) / loss_denorm

            ctrness_loss = F.binary_cross_entropy_with_logits(
                instances.ctrness_pred,
                ctrness_targets,
                reduction="sum"
            ) / num_pos_avg
        else:
            reg_loss = instances.reg_pred.sum() * 0
            ctrness_loss = instances.ctrness_pred.sum() * 0

        #TODO: remove this later
        confidence_aware = False
        if confidence_aware:
            # NOTE: apply loss_within_instance:
            arg_rank_target = 'entropy'
            confidence = compute_confidence_targets(confidence, arg_rank_target)

            # make input pair
            rank_input1 = confidence
            rank_input2 = torch.roll(confidence, -1)

            rank_target, rank_margin = get_target_margin(instances)

            # if want to use ctrness_margin
            ctrness_margin_factor = 1
            rank_margin *= ctrness_margin_factor

            rank_input2 = rank_input2 + rank_margin * rank_target

            # ranking loss
            ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()
            ranking_loss = ranking_criterion(rank_input1,
                                             rank_input2,
                                             rank_target)
            # print("ranking_loss: ",ranking_loss)

        ctr_loss_rank = self.ctr_ranking_loss(instances)

        losses = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            "loss_fcos_combined_ctr": comb_ctrness_loss

        }
        # "loss_fcos_ctr": ctrness_loss
        # "loss_fcos_neg_ctr": neg_ctrness_loss
        # "ctr_loss_rank": ctr_loss_rank,

        # "loss_fcos_ctr": ctrness_loss

        extras = {
            "instances": instances,
            "loss_denorm": loss_denorm
        }
        return extras, losses

    def predict_proposals(
            self, logits_pred, reg_pred, ctrness_pred,
            locations, image_sizes, top_feats=None, gt_instances=None, ctr_eval = False
    ):
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        if ctr_eval and gt_instances is not None:
            training_targets = self._get_ground_truth(locations, gt_instances)
            reg_targets = training_targets['reg_targets']
            gt_ctr_preds = []
            for i in range(len(reg_targets)):
                gt_ctr_pred = compute_ctrness_targets(reg_targets[i])
                gt_ctr_pred = torch.reshape(gt_ctr_pred,ctrness_pred[i].shape)
                # NOTE: calculated ctrness is [0,1], make it logit and regulate the 0 (0 logit is -inf)
                gt_ctr_pred = torch.log(gt_ctr_pred) - torch.log1p(-gt_ctr_pred)
                gt_ctr_pred[gt_ctr_pred < -20] = -20
                gt_ctr_preds.append(gt_ctr_pred)

        # ctrness_pred = gt_ctr_preds

        # #NOTE: use gt_preds as reg_preds (> -10) is the logits for 0
        # for i in range(len(gt_ctr_preds)):
        #     # ctrness_pred[i][gt_ctr_preds[i] > -20] = gt_ctr_preds[i][gt_ctr_preds[i] > -20]
        #     ctrness_pred[i][gt_ctr_preds[i] < -19] = -20

        # ctrness_pred = gt_ctr_preds

        # #NOTE: use gt_reg_preds
        # reg_targets = training_targets['reg_targets']
        # gt_reg_preds = []
        # for i in range(len(reg_targets)):
        #     gt_reg_pred = reg_targets[i]
        #     gt_reg_pred[gt_reg_pred.min(dim=1)[0]<0,:] = 0
        #     gt_reg_pred = torch.reshape(gt_reg_pred.permute(1, 0), reg_pred[i].shape)
        #     gt_reg_preds.append(gt_reg_pred)
        # reg_pred = gt_reg_preds



        sampled_boxes = []



        bundle = {
            "l": locations, "o": logits_pred,
            "r": reg_pred, "c": ctrness_pred,
            "s": self.strides,
        }

        if len(top_feats) > 0:
            bundle["t"] = top_feats

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"]
            c = per_bundle["c"]
            t = per_bundle["t"] if "t" in bundle else None

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, c, image_sizes, t
                )
            )

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = l.new_ones(
                    len(per_im_sampled_boxes), dtype=torch.long
                ) * i

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        # NOTE: add image_sizes to boxlists

        return boxlists

    def forward_for_single_feature_map(
            self, locations, logits_pred, reg_pred,
            ctrness_pred, image_sizes, top_feat=None
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
