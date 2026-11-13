import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(5)


@pytest.fixture
def pose_batch():
    p = torch.randn(2, 7, 66, 3) * 0.1
    p[:, :, 0] = torch.tensor([-0.5, 0, 0])
    p[:, :, 1] = torch.tensor([0.5, 0, 0])
    p[:, :, 2] = torch.tensor([0, 0.7, 0])
    return p, torch.ones(2, 7, 66), torch.tensor([7, 5])
