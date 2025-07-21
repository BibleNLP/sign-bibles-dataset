import pandas as pd
import pytest

from sign_bibles_dataset.tasks.text2sign.find_segment_in_video.evaluation.evaluate_utils import (
    get_relevant_segments,
    count_relevant_segments,
    calculate_recall,
    calculate_precision,
    calculate_iou,
    calculate_miou,
    get_retrieved_segments,
)


@pytest.fixture
def dummy_ground_truth():
    """Creates a small ground truth dataframe for testing."""
    return pd.DataFrame(
        [
            {"query_text": "A", "video_id": "vid1", "start_frame": 0, "end_frame": 100},
            {"query_text": "A", "video_id": "vid1", "start_frame": 200, "end_frame": 300},
            {"query_text": "B", "video_id": "vid1", "start_frame": 400, "end_frame": 500},
            {"query_text": "A", "video_id": "vid2", "start_frame": 100, "end_frame": 200},
        ]
    )


@pytest.fixture
def named_prediction_sets():
    """Provides named sets of predictions with known expected precision/recall and retrieved segments."""
    return {
        "all_relevant_hit": {
            "predictions": [(0, 100), (200, 300), (500, 600)],
            "expected_recall": 1.0,
            "expected_precision": 2 / 3,
            "expected_retrieved": {0, 1},
        },
        "none_relevant_hit": {
            "predictions": [(400, 500), (500, 600), (600, 700)],
            "expected_recall": 0.0,
            "expected_precision": 0.0,
            "expected_retrieved": [],
        },
        "half_relevant_hit": {
            "predictions": [(0, 100), (500, 600)],
            "expected_recall": 0.5,
            "expected_precision": 0.5,
            "expected_retrieved": [0],
        },
        "redundant_full_precision": {
            "predictions": [(0, 90), (10, 100)],
            "expected_recall": 0.5,
            "expected_precision": 1.0,
            "expected_retrieved": [0],
        },
        "single_covering_all_relevant": {
            "predictions": [(0, 300)],
            "expected_recall": 1.0,
            "expected_precision": 1.0,
            "expected_retrieved": [0, 1],
        },
        "single_covering_whole_video": {
            "predictions": [(0, 500)],
            "expected_recall": 1.0,
            "expected_precision": 1.0,
            "expected_retrieved": [0, 1, 2, 3],
        },
    }


def test_get_relevant_segments(dummy_ground_truth):
    segments = get_relevant_segments(dummy_ground_truth, "A", "vid1")
    assert len(segments) == 2
    assert (0, 100) in segments
    assert (200, 300) in segments

    segments = get_relevant_segments(dummy_ground_truth, "B", "vid1")
    assert segments == [(400, 500)]

    segments = get_relevant_segments(dummy_ground_truth, "A", "vid2")
    assert segments == [(100, 200)]

    segments = get_relevant_segments(dummy_ground_truth, "C", "vid1")
    assert segments == []


def test_count_relevant_segments(dummy_ground_truth):
    assert count_relevant_segments(dummy_ground_truth, "A", "vid1") == 2
    assert count_relevant_segments(dummy_ground_truth, "B", "vid1") == 1
    assert count_relevant_segments(dummy_ground_truth, "A", "vid2") == 1
    assert count_relevant_segments(dummy_ground_truth, "C", "vid1") == 0


@pytest.mark.parametrize(
    "case_name",
    [
        "all_relevant_hit",
        "none_relevant_hit",
        "half_relevant_hit",
        "redundant_full_precision",
        "single_covering_all_relevant",
        "single_covering_whole_video",
    ],
)
def test_precision_recall(dummy_ground_truth, named_prediction_sets, case_name):
    case = named_prediction_sets[case_name]
    predictions = case["predictions"]
    expected_recall = case["expected_recall"]
    expected_precision = case["expected_precision"]

    relevant_segments = get_relevant_segments(dummy_ground_truth, "A", "vid1")

    recall = calculate_recall(predictions, relevant_segments)
    precision = calculate_precision(predictions, relevant_segments)

    assert recall == pytest.approx(expected_recall), f"Failed recall on case '{case_name}'"
    assert precision == pytest.approx(expected_precision), f"Failed precision on case '{case_name}'"


@pytest.mark.parametrize("case_name", ["all_relevant_hit", "half_relevant_hit"])
def test_get_retrieved_segments(dummy_ground_truth, named_prediction_sets, case_name):
    case = named_prediction_sets[case_name]
    relevant_segments = get_relevant_segments(dummy_ground_truth, "A", "vid1")
    retrieved_indices = get_retrieved_segments(case["predictions"], relevant_segments)
    expected_indices = {idx for idx, seg in enumerate(relevant_segments) if seg in case["expected_retrieved"]}
    assert retrieved_indices == expected_indices


def test_calculate_iou():
    assert calculate_iou((0, 100), (50, 150)) == 0.5
    assert calculate_iou((0, 100), (50, 100)) == 0.5
