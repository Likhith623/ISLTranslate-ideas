from types import SimpleNamespace

import numpy as np

from isl_slt.config import Features
from isl_slt.features.pose import motion, prepare_pose, selected_joints


def test_velocity_ablation_removes_values_and_validity():
    settings = Features()
    coordinates = np.zeros((3, 48, 3), dtype=np.float32)
    coordinates[:, 0, 0] = -1
    coordinates[:, 1, 0] = 1
    sequence = SimpleNamespace(coordinates=coordinates, valid=np.ones_like(coordinates, dtype=bool),
        names=selected_joints(settings), times=np.array([0., .04, .08]), segments=np.zeros(3, dtype=int),
        regions=np.zeros(3, dtype=int), metadata={})
    full = prepare_pose(sequence, settings)
    positions = prepare_pose(sequence, Features(velocity=False))
    assert full.values.shape[1] == 384
    assert full.values.shape[1] - positions.values.shape[1] == 144


def test_velocity_uses_actual_elapsed_seconds():
    for dt in (.04, .1):
        values = np.array([[[0., 0.]], [[2 * dt, 0.]]])
        velocity, valid = motion(values, np.ones((2, 1), bool), np.array([0., dt]), np.zeros(2, int))
        assert valid[1, 0]
        assert velocity[1, 0, 0] == 2.
