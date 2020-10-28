import os

from detectron2.data.datasets.register_coco import register_coco_instances, register_coco_panoptic_separated
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances
from detectron2.data.datasets.builtin import _PREDEFINED_SPLITS_COCO
from detectron2.data import DatasetCatalog, MetadataCatalog

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

register_all_coco()