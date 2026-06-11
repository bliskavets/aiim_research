---
marp: true
title: RL Post-Training Methods for LLMs — GRPO and Beyond
author: ""
paginate: true
math: katex
theme: default
style: |
  section {
    font-size: 26px;
    padding: 48px 64px;
  }
  h1 { color: #1a3c6e; }
  h2 { color: #1a3c6e; }
  table { font-size: 22px; }
  code { font-size: 0.85em; }
  .small { font-size: 20px; color: #555; }
  .tag { color: #b5651d; font-weight: 600; }
---

<!-- _paginate: false -->

# RL Post-Training Methods for LLMs

## **GRPO**, **CISPO**, **GSPO**, **Tree-GRPO**

<br>

Policy-gradient methods for training reasoning models with reinforcement learning

<span class="small">Sources: DeepSeekMath (GRPO), MiniMax-M1 (CISPO), Qwen (GSPO), Tree Search for LLM Agent RL (Tree-GRPO)</span>

---

## Why these methods exist

- After SFT, models are fine-tuned with RL on **verifiable rewards** (math, code, reasoning).
- Vanilla PPO needs a separate **value model (critic)** — expensive and unstable on long reasoning chains.
- The shared idea of this family: drop the critic, evaluate actions **relative to a group** of samples.

<br>

**Evolution:** PPO → **GRPO** → DAPO → **CISPO** / **GSPO**, with **Tree-GRPO** for agents.

> Each step fights its own problem: critic cost, gradient clipping, dropped rare tokens, long-sequence noise.

---

## GRPO — the idea

**Group Relative Policy Optimization** (DeepSeekMath, 2024)

- For each prompt $q$, sample a **group** of $G$ responses $o_1, \dots, o_G$.
- Normalize rewards $r_i$ within the group to get the advantage **without a critic**:

$$ \hat{A}_{i,t} = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})} $$

- The group itself acts as the **baseline** → lower variance, no separate value network.
- Policy update follows the PPO scheme with **ratio clipping** plus a KL penalty to a reference model.

---

## GRPO — objective

$$ \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t} \hat{A}_{i,t},\ \mathrm{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right) \right] - \beta D_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}}) $$

where the importance ratio is

$$ r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} $$

- $\min(\cdot, \mathrm{clip}(\cdot))$ is the **trust region**: it bounds the size of the policy update.
- $\beta D_{\text{KL}}$ keeps the model close to the reference.

---

## GRPO — the weak spot

Clipping is applied to the **token's own ratio**. If a token's ratio leaves $[1-\varepsilon, 1+\varepsilon]$, its **gradient is zeroed out**.

<br>

🔴 Problem for reasoning:

- Rare but **critical "forking" tokens** — *"However", "Wait", "Aha", "Recheck"* — have low probability.
- They get large ratios → they are the first to be clipped and **drop out of training**.
- These are exactly the tokens that change the course of reasoning. Losing them → "logical confusion" on long chains.

> GRPO stabilizes training at the cost of discarding the most informative tokens.

---

## CISPO — the idea

**Clipped IS-weight Policy Optimization** (MiniMax-M1, 2025)

A key shift in *where* clipping happens:

| | What gets clipped |
|---|---|
| GRPO / PPO | the **token** update (via $\min$/clip on the ratio) |
| **CISPO** | the **importance-sampling weight** itself, not the token |

- The clip sits **inside a stop-gradient** → the weight is bounded in magnitude, but the **gradient through $\log \pi_\theta$ flows for every token**.
- It drops the trust-region clip: **no token is discarded**, including rare reasoning tokens.
- Entropy still stays within a reasonable range → stable exploration.

---

## CISPO — objective

$$ \mathcal{J}_{\text{CISPO}}(\theta) = \mathbb{E}\left[ \frac{1}{\sum_i |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \mathrm{sg}\left( \hat{r}_{i,t}(\theta) \right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right] $$

with the **clipped IS weight**

$$ \hat{r}_{i,t}(\theta) = \mathrm{clip}\left( r_{i,t}(\theta),\ 1 - \varepsilon^{\text{IS}}_{\text{low}},\ 1 + \varepsilon^{\text{IS}}_{\text{high}} \right) $$

- $\mathrm{sg}(\cdot)$ is the **stop-gradient** (`.detach()`): the weight becomes a gradient-free multiplier.
- $\varepsilon^{\text{IS}}_{\text{high}}$ is set **large** → more room to update rare tokens.
- A token's gradient is never zero → rare "fork" tokens are preserved.

---

## CISPO — pseudocode

```python
log_ratio          = per_token_logps - old_per_token_logps
importance_weights = torch.exp(log_ratio)                  # r = pi_theta / pi_old
clamped_ratios     = torch.clamp(importance_weights,
                                 max=epsilon_high).detach() # sg(clip(r))
per_token_loss     = -clamped_ratios * advantages.unsqueeze(1) * per_token_logps
```

<span class="small">The clip only bounds the <b>multiplier</b>; <code>log pi_theta</code> always stays trainable — so no token "drops out".</span>

---

## GRPO vs CISPO — summary

| | **GRPO** | **CISPO** |
|---|---|---|
| Critic / value net | none (group baseline) | none (group baseline) |
| Advantage | group normalization | group normalization |
| What gets clipped | the **token** ratio (trust region) | the **IS weight** under stop-gradient |
| Fate of rare tokens | gradient is **zeroed** | **all preserved** |
| KL penalty | yes, $\beta D_{\text{KL}}$ | optional / removed |
| Risk | losing reasoning tokens | tuning $\varepsilon^{\text{IS}}_{\text{high}}$ |

---

## CISPO — results (MiniMax-M1)

- 📈 **Outperforms GRPO and DAPO** on reasoning benchmarks.
- ⚡ Matches DAPO with **~50% of the training steps** (≈2× efficiency).
- 🎯 Smoother training trajectories, higher sample efficiency.
- 🧠 Especially strong on **long reasoning chains**, where rare tokens matter.

> Key insight: the problem was not the step size, but **which tokens take part in the gradient at all**.

---

## GSPO — the idea

**Group Sequence Policy Optimization** (Qwen, 2025)

The authors' diagnosis: GRPO's instability comes from the **misapplied importance weight at the token level**. The noise grows with response length and is amplified by clipping.

🔑 The fix: move everything to the **sequence level**, not the token level.

- Importance ratio, clipping, reward and optimization — **over the whole response**.
- The ratio is **length-normalized** (exponent $1/|o_i|$) → tames the exponential blow-up.
- Especially stabilizes **RL for MoE models**. Underpins the improvements in **Qwen3**.

---

## GSPO — objective

Sequence-level importance ratio with length normalization:

$$ s_i(\theta) = \left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \right)^{1/|o_i|} $$

PPO-style clipping, but over the **whole sequence**:

$$ \mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( s_i(\theta) \hat{A}_i,\ \mathrm{clip}(s_i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i \right) \right] $$

- The advantage $\hat{A}_i$ is **one per response** (same group normalization as GRPO).
- Paradox: GSPO clips **more** tokens, yet still **beats GRPO** → GRPO's token-level gradients are just noisier.

---

## Tree-GRPO — the idea

**Tree-based GRPO** (*Tree Search for LLM Agent RL*, 2025)

Built for **agents** and multi-step reasoning (multi-hop QA, tool use).

- Rollouts are sampled as a **tree**, not linearly: a node = a full agent interaction step, branches = divergent decision points.
- **Shared prefixes are reused** → more rollouts within the same token / tool-call budget.
- The tree yields **step-wise (process) signals** even when only a final (outcome) reward is available.

---

## Tree-GRPO — advantage

Relative advantages are computed at **two levels**:

| Level | What is compared |
|---|---|
| **Intra-tree** | branches within one tree (shared prefix) |
| **Inter-tree** | different trees in the group (like standard GRPO) |

- The intra-tree objective is **equivalent to step-level preference learning** (a DPO-like signal over steps).
- Hierarchical reward propagation through the tree → **cheaper** and more accurate on long agent trajectories.

---

<!-- Placeholder section for future methods. Duplicate the block below for each new method. -->

## Other methods — *(reserved)*

Room for the next methods to compare:

- **PPO** — the baseline actor-critic, the starting point.
- **DAPO** — Decoupled clip + Dynamic sampling (an evolution of GRPO).

<span class="small">Slides to be added in the next step.</span>

---

## The family — where clipping "lives"

| Method | Ratio level | What gets clipped | Focus |
|---|---|---|---|
| **GRPO** | token | token ratio (trust region) | baseline group-RL |
| **CISPO** | token | IS weight under stop-grad | keep rare tokens |
| **GSPO** | **sequence** | seq ratio (length-norm.) | stability, MoE |
| **Tree-GRPO** | token / step | like GRPO, but over a tree | agents, process reward |

---

## Sources

- DeepSeekMath: *GRPO* — arXiv:2402.03300
- MiniMax-M1: *Scaling Test-Time Compute Efficiently* (CISPO) — arXiv:2506.13585
- Qwen: *Group Sequence Policy Optimization* (GSPO) — arXiv:2507.18071
- *Tree Search for LLM Agent RL* (Tree-GRPO) — arXiv:2509.21240
- ms-swift docs — *Clipped Importance Sampling Policy Optimization (CISPO)*
- EmergentMind — survey articles on GRPO / CISPO / GSPO / Tree-GRPO
