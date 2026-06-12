"""
diagnose_shaping.py — does the per-token EMA-flipped shaping actually change the
advantage relative to plain GRPO, and by how much / where?

We compare, on controlled batches:
  - GRPO token advantage  = seq_advantage broadcast to every token (what `grpo` does)
  - shaped token advantage = compute_gtpo_ema_flipped_advantages(...) (+ tag mask)

and report: correlation, relative diff-norm ||shaped - grpo|| / ||grpo||, the
within-sequence std (GRPO has 0; shaping injects per-token spread), the mean of
the shaped advantage per polarity group (z-norm centers it at ~0), and the effect
of alpha1/alpha2 (z-norm washes them out).

Pure tensor math — no model, no GPU. Run alongside training safely.
"""
import sys
import torch

sys.path.insert(0, ".")
from src.ema_flipped_utils import compute_gtpo_ema_flipped_advantages
from src.format_tag_mask import apply_tag_mask_to_token_advantages


def make_batch(G=8, T=200, conf_std=1.0, seed=0):
    """One GRPO group: G completions, group-normalized seq advantages, per-token
    confidence with controllable spread (conf_std). Returns (seq_adv, confidence, mask)."""
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(G, generator=g)                 # raw rewards
    seq_adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)   # GRPO group-norm
    base_conf = 6.0                                       # typical -mean top-k logp
    confidence = base_conf + conf_std * torch.randn(G, T, generator=g)
    confidence = confidence.clamp(min=0.1)
    mask = torch.ones(G, T)
    return seq_adv, confidence, mask


def grpo_token_adv(seq_adv, T):
    return seq_adv.view(-1, 1).expand(-1, T).contiguous()


def report(tag, shaped, grpo_adv, seq_adv, mask):
    s = shaped[mask.bool()]
    g = grpo_adv[mask.bool()]
    corr = torch.corrcoef(torch.stack([s, g]))[0, 1].item()
    rel = (shaped - grpo_adv).norm().item() / (grpo_adv.norm().item() + 1e-8)
    within = shaped.std(dim=1).mean().item()           # per-seq token spread
    within_grpo = grpo_adv.std(dim=1).mean().item()
    is_pos = seq_adv > 0
    mpos = shaped[is_pos].mean().item() if is_pos.any() else float("nan")
    mneg = shaped[~is_pos].mean().item() if (~is_pos).any() else float("nan")
    print(f"\n[{tag}]")
    print(f"  corr(shaped, grpo-broadcast)      = {corr:+.3f}   (1.0 => shaping is a no-op)")
    print(f"  ||shaped - grpo|| / ||grpo||      = {rel:.3f}")
    print(f"  within-seq token std  shaped={within:.3f}  grpo={within_grpo:.3f}  (grpo=0 => uniform)")
    print(f"  mean shaped adv  O+={mpos:+.3f}  O-={mneg:+.3f}   (z-norm centers each polarity ~0)")
    print(f"  std shaped={shaped[mask.bool()].std().item():.3f}  std grpo-broadcast={grpo_adv[mask.bool()].std().item():.3f}")


def main():
    G, T = 8, 200
    print("=" * 70)
    print("CONFIDENCE VARIES across tokens (conf_std=1.0) — shaping should bite")
    seq_adv, conf, mask = make_batch(G, T, conf_std=1.0, seed=1)
    shaped = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask,
                                                 alpha1=0.9, alpha2=0.1, lam=0.9)
    report("conf_std=1.0", shaped, grpo_token_adv(seq_adv, T), seq_adv, mask)

    print("\n" + "=" * 70)
    print("CONFIDENCE ~CONSTANT across tokens (conf_std=0.01) — bonus -> uniform")
    seq_adv, conf, mask = make_batch(G, T, conf_std=0.01, seed=1)
    shaped = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask,
                                                 alpha1=0.9, alpha2=0.1, lam=0.9)
    report("conf_std=0.01", shaped, grpo_token_adv(seq_adv, T), seq_adv, mask)

    print("\n" + "=" * 70)
    print("ALPHA WASH-OUT: does alpha1/alpha2 survive the z-norm?")
    seq_adv, conf, mask = make_batch(G, T, conf_std=1.0, seed=1)
    a = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, alpha1=0.9, alpha2=0.1, lam=0.9)
    b = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, alpha1=0.1, alpha2=0.9, lam=0.9)
    c = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, alpha1=0.5, alpha2=0.5, lam=0.9)
    print(f"  max|adv(0.9/0.1) - adv(0.1/0.9)| = {(a-b).abs().max().item():.2e}")
    print(f"  max|adv(0.9/0.1) - adv(0.5/0.5)| = {(a-c).abs().max().item():.2e}")
    print("  => if ~0, alpha1/alpha2 are completely washed out by the per-polarity z-norm")

    print("\n" + "=" * 70)
    print("TAG MASK effect: reverting tag tokens to seq-adv (here mark 3 of 200)")
    seq_adv, conf, mask = make_batch(G, T, conf_std=1.0, seed=1)
    shaped = compute_gtpo_ema_flipped_advantages(seq_adv, conf, mask, 0.9, 0.1, 0.9)
    tagm = torch.zeros(G, T, dtype=torch.bool); tagm[:, [0, 1, T-1]] = True
    masked = apply_tag_mask_to_token_advantages(shaped, seq_adv, tagm)
    changed = (masked != shaped).float().mean().item()
    print(f"  fraction of positions changed by a 3/200 tag mask = {100*changed:.2f}%")


if __name__ == "__main__":
    main()
