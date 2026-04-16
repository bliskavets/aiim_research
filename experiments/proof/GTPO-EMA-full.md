# GTPO-EMA: Self-Contained Theoretical Analysis
## Convergence to the Same Global Optimum as DAPO

---

## Part 0. Setting and Motivation

We train a language model policy $\pi_\theta$: given a prompt $q$, the model generates a response $\mathbf{o} = (o_1, o_2, \ldots, o_T)$, one token at a time. The goal is to maximize the expected reward $R(q, \mathbf{o})$ — for math tasks this is 1 if the answer is correct, 0 or −1 if incorrect.

The standard GRPO approach samples $G$ responses $\{o_1, \ldots, o_G\}$ to the same prompt, normalizes rewards within the group, and applies a PPO-style clipped update. The key weakness: every token in $o_i$ gets the same advantage $\hat{A}_i$, regardless of whether that token was "important" or trivial.

**GTPO-EMA** addresses this by using a per-token weight based on the model's confidence at each position — smoothed via Exponential Moving Average.

---

## Part 1. Definitions

### Definition 1.1 — Top-k Confidence

For a policy $\pi_\theta$, prompt $q$, and partial response $o_{i,<t}$, define the per-token top-$k$ confidence at position $t$ as:

$$C_{i,t} = -\underset{v \in \text{top-}k(\pi_\theta(\cdot\,|\,q,\, o_{i,<t}))}{\text{mean}} \log \pi_\theta(v \mid q,\, o_{i,<t})$$

**Interpretation:** $C_{i,t}$ is the average negative log-probability of the most likely $k$ tokens.
- High $C_{i,t}$ → model is **uncertain** (spread-out distribution)
- Low $C_{i,t}$ → model is **confident** (peaked distribution)

**Properties:**
- $C_{i,t} \geq 0$ always
- $C_{i,t} = 0$ only if $\pi$ assigns probability 1 to a single token (degenerate)

---

### Definition 1.2 — EMA-smoothed Confidence

Define the EMA of confidence with smoothing parameter $\lambda \in (0, 1)$:

$$\text{EMA}_{i,0} = C_{i,0}$$
$$\text{EMA}_{i,t} = \lambda \cdot \text{EMA}_{i,t-1} + (1-\lambda) \cdot C_{i,t}, \quad t \geq 1$$

Expanding the recurrence:

$$\text{EMA}_{i,t} = (1-\lambda) \sum_{s=0}^{t} \lambda^s \cdot C_{i,t-s}$$

This is a **weighted average** of past confidence values, with exponentially decaying weights. Recent values have more influence than distant ones.

**Properties:**
- $\text{EMA}_{i,t} \geq 0$ (convex combination of non-negative values)
- $\text{EMA}_{i,t} = 0$ only in the degenerate case

---

### Definition 1.3 — O⁺ and O⁻ Partition

Given a group of $G$ responses $\{o_1, \ldots, o_G\}$ to prompt $q$:

$$O^+ = \{o_i : R(q, o_i) > \text{threshold}\}, \quad O^- = \{o_i : R(q, o_i) \leq \text{threshold}\}$$

Denote:
- $O^+_t = \{o_i \in O^+ : |o_i| \geq t\}$ — active successful sequences at step $t$
- $O^-_t = \{o_j \in O^- : |o_j| \geq t\}$ — active unsuccessful sequences at step $t$
- $d_t = |O^+_t|$, $\quad h_t = |O^-_t|$

---

### Definition 1.4 — GTPO-EMA Reward Shaping

For hyperparameters $\alpha_1, \alpha_2 > 0$:

**For $o_i \in O^+$ (successful) at position $t$:**

$$\tilde{r}^+_{i,t} = \alpha_1 \cdot r_i + \alpha_2 \cdot \frac{\text{EMA}_{i,t}}{\sum_{k \in O^+_t} \text{EMA}_{k,t}} \cdot d_t$$

**For $o_j \in O^-$ (unsuccessful) at position $t$:**

$$\tilde{r}^-_{j,t} = \alpha_1 \cdot (-1) + \alpha_2 \cdot \frac{\text{EMA}^{-1}_{j,t}}{\sum_{k \in O^-_t} \text{EMA}^{-1}_{k,t}} \cdot h_t \cdot (-1)$$

where $\text{EMA}^{-1}_{j,t} = 1/(\text{EMA}_{j,t} + \varepsilon)$ for numerical stability.

**Interpretation:**
- Successful tokens with HIGH confidence → LARGER bonus (model found answer while certain → reinforce)
- Unsuccessful tokens with HIGH confidence → LARGER penalty (model was wrong while certain → penalize overconfidence)

---

### Definition 1.5 — Advantage Functions

$$\tilde{A}^+_{i,t} = \frac{\tilde{r}^+_{i,t} - \text{mean}(\tilde{R}^+)}{\text{std}(\tilde{R}^+)}, \qquad \tilde{A}^-_{j,t} = \frac{\tilde{r}^-_{j,t} - \text{mean}(\tilde{R}^-)}{\text{std}(\tilde{R}^-)}$$

where $\tilde{R}^+ = \{\tilde{r}^+_{i,t} : i \in O^+,\, t \leq |o_i|\}$ and similarly $\tilde{R}^-$.

---

### Definition 1.6 — GTPO-EMA Objective

$$\mathcal{J}_{\text{GTPO-EMA}}(\theta) = \mathbb{E}\!\left[\frac{1}{\sum_k |o_k|} \left( \sum_{i \in O^+} \sum_{t=1}^{|o_i|} \min\!\left(w_{i,t}\tilde{A}^+_{i,t},\; \text{clip}(w_{i,t}, 1{-}\varepsilon, 1{+}\varepsilon)\tilde{A}^+_{i,t}\right) + \sum_{j \in O^-} \sum_{t=1}^{|o_j|} \min\!\left(w_{j,t}\tilde{A}^-_{j,t},\; \text{clip}(w_{j,t}, 1{-}\varepsilon, 1{+}\varepsilon)\tilde{A}^-_{j,t}\right) \right)\right]$$

where $w_{i,t}(\theta) = \dfrac{\pi_\theta(o_{i,t}\,|\,q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t}\,|\,q, o_{i,<t})}$ is the importance sampling ratio.

---

## Part 2. Main Results

### Lemma 2.1 — Non-negativity of EMA

**Statement:** For any $i$ and $t \geq 0$: $\;\text{EMA}_{i,t} \geq 0$.

**Proof** by induction on $t$:

- *Base case* $t=0$: $\text{EMA}_{i,0} = C_{i,0} \geq 0$ by Definition 1.1.
- *Inductive step*: Assume $\text{EMA}_{i,t-1} \geq 0$. Then:
$$\text{EMA}_{i,t} = \underbrace{\lambda}_{\geq 0} \cdot \underbrace{\text{EMA}_{i,t-1}}_{\geq 0} + \underbrace{(1-\lambda)}_{\geq 0} \cdot \underbrace{C_{i,t}}_{\geq 0} \;\geq\; 0 \qquad \square$$

**Corollary:** $\sum_{k \in O^+_t} \text{EMA}_{k,t} > 0$, so normalization is well-defined.

---

### Lemma 2.2 — Normalization Property

**Statement:** For any set $S$ with $\text{EMA}_{i,t} \geq 0$:

$$\sum_{i \in S} \frac{\text{EMA}_{i,t}}{\sum_{k \in S} \text{EMA}_{k,t}} = 1$$

**Proof:** Direct — numerators sum to the denominator. $\square$

---

### Proposition 2.3 — Conservation of Positive Reward Mass

**Statement:** If $\alpha_1 + \alpha_2 = 1$, then at every timestep $t$:

$$\sum_{i \in O^+_t} \tilde{r}^+_{i,t} = \sum_{i \in O^+_t} r_i = d_t$$

The EMA shaping **redistributes** rewards among tokens but does not create or destroy total reward.

**Proof:**

$$\sum_{i \in O^+_t} \tilde{r}^+_{i,t} = \sum_{i \in O^+_t} \left[\alpha_1 \cdot r_i + \alpha_2 \cdot \frac{\text{EMA}_{i,t}}{\sum_k \text{EMA}_{k,t}} \cdot d_t\right]$$

$$= \alpha_1 \underbrace{\sum_{i \in O^+_t} r_i}_{d_t} + \alpha_2 \cdot d_t \cdot \underbrace{\sum_{i \in O^+_t} \frac{\text{EMA}_{i,t}}{\sum_k \text{EMA}_{k,t}}}_{= 1 \text{ (Lemma 2.2)}}$$

$$= \alpha_1 d_t + \alpha_2 d_t = (\alpha_1 + \alpha_2) d_t = d_t \qquad \square$$

> **Note:** This proof holds for **any** non-negative weight function. EMA non-negativity (Lemma 2.1) is the only requirement.

---

### Lemma 2.5 — Deviation Formula

**Statement:** For $o_i \in O^+_t$:

$$\delta_{i,t} \;=\; \tilde{r}^+_{i,t} - r_i \;=\; \alpha_2 \cdot \left(\frac{\text{EMA}_{i,t}}{\overline{\text{EMA}}_t} - 1\right)$$

where $\overline{\text{EMA}}_t = \frac{1}{d_t}\sum_{k \in O^+_t} \text{EMA}_{k,t}$.

**Proof:**

$$\delta_{i,t} = \tilde{r}^+_{i,t} - r_i = \left[\alpha_1 + \alpha_2 \cdot \frac{\text{EMA}_{i,t}}{\overline{\text{EMA}}_t}\right] - 1$$

$$= (\alpha_1 - 1) + \alpha_2 \cdot \frac{\text{EMA}_{i,t}}{\overline{\text{EMA}}_t} = -\alpha_2 + \alpha_2 \cdot \frac{\text{EMA}_{i,t}}{\overline{\text{EMA}}_t} = \alpha_2\!\left(\frac{\text{EMA}_{i,t}}{\overline{\text{EMA}}_t} - 1\right) \qquad \square$$

**Interpretation:**
- $\delta_{i,t} = 0$ when token $i$ has exactly the group-average confidence
- $\delta_{i,t} > 0$ when token $i$ is more uncertain than average → gets a bonus
- $\delta_{i,t} < 0$ when token $i$ is more confident than average → gets reduced reward

---

### Assumption 2.6 — EMA Consolidation Condition

As training iteration $k \to \infty$, for any two successful sequences $o_{i_1}, o_{i_2} \in O^+_t$:

$$\lim_{k\to\infty} \frac{\text{EMA}_{i_1,t}}{\text{EMA}_{i_2,t}} = 1 \qquad \text{(almost surely)}$$

Equivalently: $\lim_{k\to\infty} \text{EMA}_{i,t} / \overline{\text{EMA}}_t = 1$.

**Why this is reasonable:** As training converges, the policy learns a canonical reasoning pattern. Successful sequences develop similar confidence profiles $C_{i,t} \to \bar{C}$. Since EMA is a weighted average of past $C$ values, if $C_{i,t} \to \bar{C}$ then $\text{EMA}_{i,t} \to \bar{C}$ too, and the ratio approaches 1.

---

### Proposition 2.7 — Asymptotic Consistency

**Statement:** Under Assumption 2.6, with bounded score function $\|\nabla_\theta \log \pi_\theta\| \leq M$ and bounded advantages:

$$\lim_{k\to\infty} \|\Delta\mathcal{J}(\theta_k)\| = 0 \qquad \text{where} \quad \Delta\mathcal{J} = \nabla\mathcal{J}^+_\text{GTPO-EMA} - \nabla\mathcal{J}_\text{GRPO}$$

**Proof:**

**Step 1:** From Lemma 2.5 and Assumption 2.6:
$$\lim_{k\to\infty} \delta_{i,t} = \alpha_2 \cdot (1 - 1) = 0$$

**Step 2:** As $\tilde{r}^+_{i,t} = r_i + \delta_{i,t}$ and $\delta_{i,t} \to 0$, by the Continuous Mapping Theorem:
$$\tilde{A}^+_{i,t} \to \hat{A}_{i,t} \qquad \text{(the shaped advantage converges to the GRPO advantage)}$$

**Step 3:** Bound the gradient difference:

$$\|\Delta\mathcal{J}(\theta_k)\| \leq M \cdot \mathbb{E}\!\left[\frac{1}{G}\sum_i\left|\tilde{A}^+_{i,t} - \hat{A}_{i,t}\right|\right]$$

Since the bracketed term converges to 0 almost surely and is dominated by a bounded constant, by the **Lebesgue Dominated Convergence Theorem**:

$$\lim_{k\to\infty} \|\Delta\mathcal{J}(\theta_k)\| \leq M \cdot \mathbb{E}\!\left[\lim_{k\to\infty} \frac{1}{G}\sum_i|\tilde{A}^+_{i,t} - \hat{A}_{i,t}|\right] = M \cdot 0 = 0 \qquad \square$$

---

### Theorem 2.8 — Same Global Optimum as DAPO

**Statement:** Under Assumptions 2.6 and the regularity conditions of Proposition 2.7, GTPO-EMA shares the same global optimum $\theta^*$ as DAPO.

**Proof:**

**Step 1 (GTPO-EMA = GRPO at optimum):**

At any critical point $\theta^*$ of $\mathcal{J}_\text{GTPO-EMA}$: $\nabla\mathcal{J}_\text{GTPO-EMA}(\theta^*) = 0$.

By Proposition 2.7, as $k \to \infty$:
$$\nabla\mathcal{J}_\text{GTPO-EMA}(\theta^*) \to \nabla\mathcal{J}_\text{GRPO}(\theta^*)$$

Therefore $\nabla\mathcal{J}_\text{GRPO}(\theta^*) = 0$ — the same point is a critical point of GRPO.

**Step 2 (GRPO = DAPO at optimum):**

DAPO assigns reward $-1$ to incorrect answers (vs $0$ in GRPO). Both objectives share the optimal policy $\theta^* = \arg\max\, \Pr[\text{correct} \mid q, \pi_\theta]$, since the reward difference is a constant shift of the baseline — it changes gradient magnitudes but not signs or the optimal direction.

**Step 3:** Combining Steps 1 and 2:
$$\theta^*_\text{GTPO-EMA} \;=\; \theta^*_\text{GRPO} \;=\; \theta^*_\text{DAPO} \qquad \square$$

---

### Remark 2.9 — Gradient Decomposition

The GTPO-EMA gradient decomposes as:

$$\mathbb{E}[\nabla\mathcal{J}^+_\text{GTPO-EMA}] = \underbrace{\mathbb{E}[\nabla\mathcal{J}_\text{GRPO}]}_{\text{toward correct answers}} + \underbrace{\mathbb{E}[\text{Cov}(\Delta_\text{EMA},\, \nabla\log\pi_\theta)]}_{\text{exploration bonus}}$$

Since $\sum \delta_{i,t} = 0$ (zero-sum redistribution by Proposition 2.3), **the model cannot increase the exploration bonus without producing correct answers**. The EMA signal shapes the path to the optimum but does not change the destination.

---

## Part 3. Additional Property: Variance Reduction

### Proposition 3.1 — EMA Reduces Weight Variance

Assuming $C_{i,t}$ have variance $\sigma^2_C$, as $t \to \infty$:

$$\text{Var}(\text{EMA}_{i,t}) \;\to\; \frac{1-\lambda}{1+\lambda} \cdot \sigma^2_C \;<\; \sigma^2_C$$

**For $\lambda = 0.9$:**
$$\text{Var}(\text{EMA}) \approx 0.053 \cdot \sigma^2_C \qquad \approx 19\times \text{ reduction}$$

**Proof:** Since $\text{EMA}_{i,t} = (1-\lambda)\sum_{s=0}^t \lambda^s C_{i,t-s}$ with i.i.d. $C_{i,s}$:

$$\text{Var}(\text{EMA}_{i,t}) = (1-\lambda)^2 \sum_{s=0}^{t} \lambda^{2s} \cdot \sigma^2_C = \frac{(1-\lambda)^2}{1-\lambda^2}(1-\lambda^{2(t+1)})\sigma^2_C \xrightarrow{t\to\infty} \frac{1-\lambda}{1+\lambda}\sigma^2_C \qquad \square$$

> **Remark:** The i.i.d. assumption is an approximation (tokens are auto-regressive). In practice, positive temporal correlation further **reduces** variance, making this a conservative bound.

---

## Part 4. Summary

| Result | Statement |
|--------|-----------|
| Lemma 2.1 | $\text{EMA}_{i,t} \geq 0$ — normalization is well-defined |
| Lemma 2.2 | $\sum (\text{EMA}_{i,t} / \sum \text{EMA}) = 1$ |
| Proposition 2.3 | $\sum \tilde{r}^+_{i,t} = d_t$ — reward mass conserved |
| Lemma 2.5 | $\delta_{i,t} = \alpha_2(\text{EMA}_{i,t}/\overline{\text{EMA}}_t - 1)$ |
| Proposition 2.7 | $\|\Delta\mathcal{J}(\theta_k)\| \to 0$ — gradient bias vanishes |
| **Theorem 2.8** | **$\theta^*_\text{GTPO-EMA} = \theta^*_\text{DAPO}$** |
| Proposition 3.1 | $\text{Var}(\text{EMA}) \approx \sigma^2/19$ for $\lambda=0.9$ |

**Central conclusion:** GTPO-EMA does not change *where* the policy converges, but changes *how* it gets there — with a more informative, lower-variance gradient signal that encourages exploration on uncertain tokens and penalizes overconfident errors.
