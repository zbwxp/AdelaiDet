from detectron2.config import CfgNode


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()

def add_point_sup_config(cfg):
    """
    Add config for point supervision.
    """
    # Use point annotation
    cfg.INPUT.POINT_SUP = False
    # Sample only part of points in each iteration.
    # Default: 0, use all available points.
    cfg.INPUT.SAMPLE_POINTS = 0
    cfg.MODEL.BOXINST.POINT_ANNO = 0
    cfg.MODEL.BOXINST.PAIRWISE.ENABLED = True
    cfg.MODEL.BOXINST.LOSS_WEIGHT = 1.0