import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou


def merge_mirror_boxes(original, mirrored, width, nms_iou=.4, max_det=300):
    """Fuse same-class boxes from two views in original-image coordinates."""
    mirrored = mirrored.clone()
    mirrored[:, [0, 2]] = width - mirrored[:, [2, 0]]
    overlap = box_iou(original[:, :4], mirrored[:, :4]).numpy()
    same_class = (original[:, 5, None] == mirrored[:, 5]).numpy()
    left, right = np.nonzero(same_class & (overlap >= .5))
    order = np.argsort(-overlap[left, right], kind="stable")
    used_left = torch.zeros(len(original), dtype=torch.bool)
    used_right = torch.zeros(len(mirrored), dtype=torch.bool)
    fused = []
    for index in order:
        a, b = left[index], right[index]
        if used_left[a] or used_right[b]:
            continue
        used_left[a] = used_right[b] = True
        views = torch.stack((original[a], mirrored[b]))
        scores = views[:, 4]
        box = (views[:, :4] * scores[:, None]).sum(dim=0) / scores.sum()
        fused.append(torch.cat((box, scores.mean().reshape(1), views[0, 5:6])))
    unmatched = torch.cat((original[~used_left], mirrored[~used_right]))
    unmatched[:, 4] *= .75
    candidates = torch.cat((torch.stack(fused), unmatched)) if fused else unmatched
    keep = batched_nms(candidates[:, :4], candidates[:, 4], candidates[:, 5], nms_iou)[:max_det]
    return candidates[keep]


def recover_large_missing_holes(primary, low_resolution, class_id, confidence, hole_cutoff, image_area):
    """Keep primary detections and add large missing holes supported by both resolutions."""
    output = primary[primary[:, 4] >= confidence]
    primary_holes = primary[primary[:, 5] == class_id]
    candidates = low_resolution[(low_resolution[:, 5] == class_id) & (low_resolution[:, 4] >= max(.50, confidence))]
    area = (candidates[:, 2:4] - candidates[:, :2]).prod(dim=1)
    support = box_iou(candidates[:, :4], primary_holes[primary_holes[:, 4] >= .005, :4])
    accepted = primary_holes[primary_holes[:, 4] >= max(confidence, hole_cutoff)]
    duplicates = box_iou(candidates[:, :4], accepted[:, :4])
    keep = (area >= .003 * image_area) & (support >= .30).any(dim=1) & ~(duplicates >= .40).any(dim=1)
    return torch.cat((output, candidates[keep]))


def confirm_other_defects(primary, low_resolution, missing_hole_id, hole_cutoff):
    """Check other defect boxes across scales when both views see several missing holes."""
    holes = primary[(primary[:, 5] == missing_hole_id) & (primary[:, 4] >= hole_cutoff)]
    low_holes = low_resolution[(low_resolution[:, 5] == missing_hole_id) & (low_resolution[:, 4] >= .50)]
    if len(holes) < 3 or len(low_holes) < 3:
        return primary
    hole_overlap = box_iou(holes[:, :4], low_holes[:, :4])
    if int((hole_overlap >= .30).any(dim=1).sum()) < 3:
        return primary

    evidence = low_resolution[low_resolution[:, 4] >= .01]
    overlap = box_iou(primary[:, :4], evidence[:, :4])
    same_class = primary[:, 5, None] == evidence[:, 5]
    confirmed = ((overlap >= .40) & same_class).any(dim=1)
    return primary[(primary[:, 5] == missing_hole_id) | confirmed]
