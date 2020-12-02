import os

from detectron2.data.datasets.register_coco import register_coco_instances, register_coco_panoptic_separated
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances
from detectron2.data.datasets.builtin import _PREDEFINED_SPLITS_COCO, _RAW_CITYSCAPES_SPLITS
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances, load_cityscapes_semantic


# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_train_xxxxxxxxxxxxxxxxxxxxxxxxx": ("coco/train2017", "coco/annotations/instances_train2017_xxx.json"),

}



def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

    # register new train json
    register_coco_instances(
        "coco_2017_train_pano_instance",
        _get_builtin_metadata("coco"),
        os.path.join(root, "coco/annotations/instances_from_pano_train2017.json"),
        os.path.join(root, "coco/train2017"),
    )

    prefix = "coco_2017_train_pano_instance_panoptic"
    panoptic_root = 'coco/panoptic_train2017'
    panoptic_json = 'coco/annotations/panoptic_train2017.json'
    semantic_root = 'coco/panoptic_stuff_train2017'
    prefix_instances = prefix[: -len("_panoptic")]
    instances_meta = MetadataCatalog.get(prefix_instances)
    image_root, instances_json = instances_meta.image_root, instances_meta.json_file

    register_coco_panoptic_separated(
        prefix,
        _get_builtin_metadata("coco_panoptic_separated"),
        image_root,
        os.path.join(root, panoptic_root),
        os.path.join(root, panoptic_json),
        os.path.join(root, semantic_root),
        instances_json,
    )

    # register test-dev panoptic
    register_coco_panoptic_separated(
        "coco_2017_test-dev_panoptic",
        _get_builtin_metadata("coco_panoptic_separated"),
        "datasets/coco/test2017",
        "datasets/coco/annotations/panoptic_val2017_100",
        'datasets/coco/annotations/panoptic_val2017_100.json',
        'datasets/coco/panoptic_stuff_val2017_100',
        "datasets/coco/annotations/image_info_test-dev2017.json"
    )


def register_all_cityscapes(root="datasets"):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        # implement the 'panoptic_seg' key
        pano_key = key.format(task="panoptic_seg")
        DatasetCatalog.register(
            pano_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_panoptic(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(pano_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

def load_cityscapes_panoptic(image_dir, gt_dir, from_json=True, to_polygons=True):
    ret_instances = load_cityscapes_instances(image_dir, gt_dir, from_json, to_polygons)
    ret_semantic = load_cityscapes_semantic(image_dir,gt_dir)

    for ret_i, ret_s in zip(ret_instances, ret_semantic):
        assert ret_i['file_name'] == ret_s['file_name'], f"image for instance and sem doesn't match!"
        ret_i["sem_seg_file_name"] = ret_s["sem_seg_file_name"]

    return ret_instances

register_all_coco()
register_all_cityscapes()