VISDRONE_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "pre-twisted suspension clamp"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bag-type suspension clamp"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "compression-type strain clamp"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "wedge-type strain clamp"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "hanging board"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "u-type hanging ring"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "yoke plate"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "parallel groove clamp"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "shockproof hammer"},
    {"color": [60, 0, 100], "isthing": 1, "id": 10, "name": "spacer"},
    {"color": [80, 170, 130], "isthing": 1, "id": 11, "name": "grading ring"},
    {"color": [50, 70, 0], "isthing": 1, "id": 12, "name": "shielded ring"},
    {"color": [220, 10, 80], "isthing": 1, "id": 13, "name": "weight"},
    {"color": [20, 150, 130], "isthing": 1, "id": 14, "name": "adjusting board"},
    {"color": [250, 170, 30], "isthing": 1, "id": 15, "name": "insulator"}
]

UAVDT_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "car"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "truck"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bus"}
]


def _get_visdrone_instances_meta():
    thing_ids = [k["id"] for k in VISDRONE_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VISDRONE_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 15, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 9]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VISDRONE_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_uavdt_instances_meta():
    thing_ids = [k["id"] for k in UAVDT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in UAVDT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 3, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 2]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in UAVDT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret
