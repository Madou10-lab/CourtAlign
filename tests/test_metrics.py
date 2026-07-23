import numpy as np

from courtalign_common.evaluation.segmentation import confusion_matrix_from_masks, metrics_from_confusion


def test_segmentation_metrics_perfect_prediction():
    y_true = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    y_pred = y_true.copy()
    cm = confusion_matrix_from_masks(y_true, y_pred, 2)
    metrics = metrics_from_confusion(cm, ["background", "court"])
    assert metrics["mean_iou"] == 1.0
    assert metrics["mean_dice"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0
