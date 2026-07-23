import numpy as np

from courtalign_common.evaluation.line_iou import line_iou_by_tolerance
from courtalign_common.evaluation.metric_homography import homography_from_corners, scale_homography_to_mask
from courtalign_common.evaluation.official_protocol import (
    OfficialCourtSpec,
    geometric_errors,
    handling_outcome,
    iou_part,
    iou_whole,
    pck_h,
)


def synthetic_spec():
    return OfficialCourtSpec(
        dataset_id="synthetic",
        sport="synthetic",
        width_m=10.0,
        length_m=20.0,
        zones={1: np.asarray([(0, 0), (10, 0), (0, 20), (10, 20)], dtype=np.float64)},
        landmarks=np.asarray([(0, 0), (10, 0), (0, 20), (10, 20), (5, 10)], dtype=np.float64),
        line_segments=[((0, 0), (10, 0)), ((0, 20), (10, 20))],
    )


def test_official_metrics_are_exact_for_identical_homographies():
    spec = synthetic_spec()
    image_corners = np.asarray([(100, 60), (500, 80), (40, 360), (560, 380)], dtype=np.float64)
    homography = homography_from_corners(spec.outer_corners, image_corners)
    mask_h = scale_homography_to_mask(homography, 640, 480, (240, 320))
    visible = np.zeros((240, 320), dtype=np.uint8)
    import cv2

    projected = cv2.perspectiveTransform(spec.outer_corners.astype(np.float32).reshape(-1, 1, 2), mask_h)
    polygon = projected.reshape(-1, 2)[[0, 1, 3, 2]].round().astype(np.int32)
    cv2.fillPoly(visible, [polygon], 1)

    assert np.isclose(iou_part(mask_h, mask_h, visible, spec, 40), 1.0)
    assert np.isclose(iou_whole(homography, homography, spec, 40), 1.0)

    errors = geometric_errors(mask_h, mask_h, homography, homography, visible, spec, 2500, (50, 50), 480)
    assert np.isclose(errors["projection_mean_m"], 0.0, atol=1e-10)
    assert np.isclose(errors["reprojection_mean_px"], 0.0, atol=1e-10)
    assert errors["n_reprojection_points"] == 2500

    landmarks = pck_h(homography, homography, spec.landmarks, (640, 480), [5, 10], [0.05, 0.1])
    assert landmarks["n_landmarks"] == 5
    assert landmarks["pck_5px"] == 1.0
    assert landmarks["pck_5pct_diag"] == 1.0


def test_iou_whole_decreases_for_shifted_prediction():
    spec = synthetic_spec()
    image_corners = np.asarray([(100, 60), (500, 80), (40, 360), (560, 380)], dtype=np.float64)
    gt = homography_from_corners(spec.outer_corners, image_corners)
    shifted = gt.copy()
    shifted[0, 2] += 40.0
    value = iou_whole(shifted, gt, spec, 40)
    assert value is not None
    assert 0.0 < value < 1.0


def test_reprojection_error_is_reported_in_native_image_pixels():
    spec = synthetic_spec()
    image_corners = np.asarray([(100, 60), (500, 80), (40, 360), (560, 380)], dtype=np.float64)
    gt_image_h = homography_from_corners(spec.outer_corners, image_corners)
    image_translation = np.asarray([[1, 0, 12], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pred_image_h = image_translation @ gt_image_h
    gt_mask_h = scale_homography_to_mask(gt_image_h, 640, 480, (240, 320))
    pred_mask_h = scale_homography_to_mask(pred_image_h, 640, 480, (240, 320))

    visible = np.zeros((240, 320), dtype=np.uint8)
    visible[20:220, 20:300] = 1
    errors = geometric_errors(
        pred_mask_h,
        gt_mask_h,
        pred_image_h,
        gt_image_h,
        visible,
        spec,
        2500,
        (50, 50),
        480,
    )

    assert np.isclose(errors["reprojection_mean_px"], 12.0)
    assert np.isclose(errors["reprojection_mean_height_normalized"], 12.0 / 480.0)


def test_handling_contract_counts_visible_failures_and_non_court_false_positives():
    assert handling_outcome(True, True) == {
        "visible": 1,
        "nonvisible": 0,
        "visible_success": 1,
        "false_registration": 0,
        "correct_handling": 1,
    }
    assert handling_outcome(True, False)["correct_handling"] == 0
    assert handling_outcome(True, False)["visible_success"] == 0
    assert handling_outcome(False, True)["false_registration"] == 1
    assert handling_outcome(False, True)["correct_handling"] == 0
    assert handling_outcome(False, False)["correct_handling"] == 1


def test_out_of_frame_prediction_is_penalized():
    spec = synthetic_spec()
    image_corners = np.asarray([(100, 60), (500, 80), (40, 360), (560, 380)], dtype=np.float64)
    gt = homography_from_corners(spec.outer_corners, image_corners)
    outside = np.asarray([[1, 0, 2000], [0, 1, 1200], [0, 0, 1]], dtype=np.float64) @ gt
    assert iou_whole(outside, gt, spec, 40) == 0.0
    pck = pck_h(outside, gt, spec.landmarks, (640, 480), [5, 10], [0.05, 0.1])
    assert pck["pck_10px"] == 0.0


def test_identity_aware_pck_exposes_mirrored_homography():
    spec = synthetic_spec()
    image_corners = np.asarray([(100, 60), (500, 80), (40, 360), (560, 380)], dtype=np.float64)
    gt = homography_from_corners(spec.outer_corners, image_corners)
    metric_reflection = np.asarray([[-1, 0, spec.width_m], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    mirrored = gt @ metric_reflection
    pck = pck_h(mirrored, gt, spec.landmarks, (640, 480), [5, 10], [0.05, 0.1])
    assert pck["pck_10px"] < 1.0
    # A rectangular support can be unchanged by a left/right reflection;
    # whole-court IoU therefore cannot be the only reported metric.
    assert np.isclose(iou_whole(mirrored, gt, spec, 40), 1.0)


def test_line_iou_tolerance_dilates_both_binary_line_masks():
    predicted = np.zeros((40, 80), dtype=bool)
    reference = np.zeros_like(predicted)
    predicted[18, 10:70] = True
    reference[20, 10:70] = True
    values = line_iou_by_tolerance(predicted, reference, [0, 3, 5])
    assert values["0"] == 0.0
    assert 0.0 < values["3"] < values["5"] < 1.0
