import numpy as np

from perception.torchvision_keypoint_detector import _decode_keypoint_visibility


def test_partial_corner_visibility_is_not_forced_true():
    corners = np.array([[10.0, 90.0], [90.0, 90.0], [90.0, 10.0], [10.0, 10.0]])
    logits = np.array([5.0, 5.0, 5.0, -5.0])
    visible, confidence = _decode_keypoint_visibility(
        corners,
        image_width=100,
        image_height=100,
        keypoint_logits=logits,
        instance_score=0.99,
        confidence_threshold=0.5,
        min_quad_area_px2=16.0,
    )
    assert visible.tolist() == [True, True, True, False]
    assert confidence[3] < 0.5


def test_out_of_frame_corner_is_rejected():
    corners = np.array([[10.0, 90.0], [90.0, 90.0], [90.0, 10.0], [-2.0, 10.0]])
    visible, _ = _decode_keypoint_visibility(
        corners,
        image_width=100,
        image_height=100,
        keypoint_logits=np.full(4, 8.0),
        instance_score=0.99,
        confidence_threshold=0.5,
        min_quad_area_px2=16.0,
    )
    assert visible.tolist() == [True, True, True, False]


def test_impossible_quad_rejects_entire_observation():
    corners = np.array([[10.0, 90.0], [90.0, 10.0], [90.0, 90.0], [10.0, 10.0]])
    visible, _ = _decode_keypoint_visibility(
        corners,
        image_width=100,
        image_height=100,
        keypoint_logits=np.full(4, 8.0),
        instance_score=0.99,
        confidence_threshold=0.5,
        min_quad_area_px2=16.0,
    )
    assert not np.any(visible)
