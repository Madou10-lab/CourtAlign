import numpy as np

from courtalign_common.evaluation.metric_homography import (
    derive_badminton_homography_from_zone_mask,
    homography_from_corners,
    rasterize_template,
    template_space_iou_from_homographies,
)
from courtalign_common.evaluation.metric_templates import badminton_doubles_zone_template
from courtalign_2s.registration.corner_extraction import extract_corners_from_binary_mask


def test_courtalign_2s_corner_extractor_recovers_quadrilateral_corners():
    mask = np.zeros((120, 160), dtype=np.uint8)
    polygon = np.asarray([[35, 25], [125, 30], [20, 95], [140, 100]], dtype=np.int32)
    fill_order = polygon[[0, 1, 3, 2]].reshape(-1, 1, 2)
    import cv2

    cv2.fillPoly(mask, [fill_order], 1)

    result = extract_corners_from_binary_mask(
        mask,
        frame_shape_hw=mask.shape,
        closing_iterations=1,
        erosion_iterations=0,
        min_area_px=50,
    )

    assert result.status == "ok"
    assert result.corners_tl_tr_bl_br.shape == (4, 2)
    assert np.allclose(result.corners_tl_tr_bl_br, polygon, atol=6.0)


def test_badminton_metric_homography_round_trip_from_synthetic_mask():
    template = badminton_doubles_zone_template(
        width_m=6.1,
        length_m=13.4,
        singles_width_m=5.18,
        doubles_long_service_from_back_m=0.72,
        short_service_from_net_m=1.98,
        source_note="test",
    )
    image_corners = np.asarray(
        [
            [240.0, 90.0],
            [720.0, 100.0],
            [120.0, 880.0],
            [840.0, 870.0],
        ],
        dtype=np.float64,
    )
    metric_to_mask = homography_from_corners(template.fullcourt_corners_tl_tr_bl_br, image_corners)
    mask = rasterize_template(template, metric_to_mask, shape=(1000, 1000))

    derived = derive_badminton_homography_from_zone_mask(
        mask,
        template,
        image_width=1000,
        image_height=1000,
        min_zone_miou=0.85,
        min_fullcourt_iou=0.90,
    )

    assert derived.status == "valid"
    assert derived.validation["zone_miou"] > 0.85
    assert derived.validation["fullcourt_iou"] > 0.90
    assert derived.validation["n_correspondence_points"] == 52
    assert len(derived.correspondences["image_points_by_class"]) == 13
    assert np.asarray(derived.correspondences["image_points_by_class"]["1"]).shape == (4, 2)

    whole_template_iou = template_space_iou_from_homographies(
        metric_to_mask,
        metric_to_mask,
        template,
        class_ids=range(1, 14),
    )
    assert np.isclose(whole_template_iou["fullcourt_occupancy_iou"], 1.0)
    assert np.isclose(whole_template_iou["zone_miou"], 1.0)
