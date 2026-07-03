"""CPU unit tests for rank-based adaptive-k C (exp_069). No training / no GPU."""
import unsloth  # noqa: F401  (import order: unsloth before trl-adjacent imports)
import torch
from src.rank_c import rank_C


def _sorted_lp(probs):
    """probs: list summing ~1 -> (1,N) sorted-desc logprobs tensor."""
    p = torch.tensor(probs, dtype=torch.float32)
    return torch.log(p).sort(descending=True).values.unsqueeze(0)


def test_rank1_gives_k1():
    # sampled token is the argmax -> rank 1 -> k=1 -> C = -log(top1)
    lp = _sorted_lp([0.7, 0.2, 0.1])
    C, k = rank_C(lp, sampled_logprob=lp[..., 0], ranks=torch.tensor([1]), cap=5, min_k=1)
    assert k.item() == 1
    assert torch.allclose(C, -lp[..., 0], atol=1e-5)


def test_rank3_gives_k3():
    lp = _sorted_lp([0.4, 0.3, 0.2, 0.1])
    C, k = rank_C(lp, sampled_logprob=lp[..., 2], ranks=torch.tensor([3]), cap=5, min_k=1)
    assert k.item() == 3
    expected = -(lp[..., :3].mean())
    assert torch.allclose(C, expected.unsqueeze(0), atol=1e-5)


def test_rank_clamped_to_cap():
    lp = _sorted_lp([0.3, 0.25, 0.2, 0.15, 0.06, 0.04])
    C, k = rank_C(lp, sampled_logprob=lp[..., 5], ranks=torch.tensor([7]), cap=5, min_k=1)
    assert k.item() == 5  # rank 7 clamped down to cap
    expected = -(lp[..., :5].mean())
    assert torch.allclose(C, expected.unsqueeze(0), atol=1e-5)


def test_min_k_floor():
    lp = _sorted_lp([0.9, 0.06, 0.04])
    C, k = rank_C(lp, sampled_logprob=lp[..., 0], ranks=torch.tensor([1]), cap=5, min_k=2)
    assert k.item() == 2  # rank 1 floored up to min_k=2


def test_batch_mixed_ranks():
    lp = torch.stack([
        _sorted_lp([0.6, 0.3, 0.1])[0],
        _sorted_lp([0.5, 0.3, 0.2])[0],
    ])  # (2, 3)
    ranks = torch.tensor([1, 3])
    C, k = rank_C(lp, sampled_logprob=lp[:, 0], ranks=ranks, cap=5, min_k=1)
    assert k.tolist() == [1, 3]
    assert torch.allclose(C[0], -lp[0, 0], atol=1e-5)
    assert torch.allclose(C[1], -lp[1, :3].mean(), atol=1e-5)


if __name__ == "__main__":
    test_rank1_gives_k1(); test_rank3_gives_k3(); test_rank_clamped_to_cap()
    test_min_k_floor(); test_batch_mixed_ranks()
    print("all rank_c tests passed")
