import numpy as np

from fedsiggraph.anomaly.calibration import CompositeScoreCalibrator


def test_missing_context_does_not_receive_continuation_penalty():
    base = np.asarray([0.1, 0.2, 0.3, 0.4])
    continuation = np.asarray([np.nan, 1.0, np.nan, 1.4])
    mask = np.asarray([False, True, False, True])

    calibrator = CompositeScoreCalibrator.fit(
        base,
        continuation,
        mask,
        continuation_weight=0.5,
    )
    transformed = calibrator.transform(base, continuation, mask)

    base_only = calibrator.base.transform(base)
    assert transformed[0] == base_only[0]
    assert transformed[2] == base_only[2]
    assert transformed[1] != base_only[1]


def test_no_available_context_falls_back_to_base_only():
    base = np.asarray([0.1, 0.2, 0.3])
    continuation = np.asarray([np.nan, np.nan, np.nan])
    mask = np.asarray([False, False, False])

    calibrator = CompositeScoreCalibrator.fit(base, continuation, mask)
    assert calibrator.continuation is None
    np.testing.assert_allclose(
        calibrator.transform(base, continuation, mask),
        calibrator.base.transform(base),
    )
