# RL Post-Training для LLM

## Зачем RL после pre-training?

1. **Alignment**: pre-trained модель оптимизирует NTP, не "помощность"
2. **Verifiable tasks**: для кода/математики есть объективный reward (тест проходит / нет) — проще scoring чем SFT
3. **Exploration**: RL позволяет найти решения, которых нет в обучающих данных
4. **Reasoning**: длинные цепочки рассуждений возникают через RL (DeepSeek-R1)
5. Цитата из транскрипта: "иногда проверить правильность ответа проще, чем построить путь к решению"

---

## Эволюция алгоритмов

### 1. SFT (Supervised Fine-Tuning)
- Просто обучение на пример→ответ парах
- Проблема: **memorizes, doesn't generalize** (Google paper)
- Хорошо для formatting, style; плохо для reasoning

### 2. RLHF + PPO (InstructGPT, 2022)
**Компоненты:**
1. SFT model
2. Reward Model (RM): обучается на human preferences (Bradley-Terry objective)
3. PPO: RL оптимизация SFT под RM с KL constraint

```
L_PPO = E[r_θ(x,y) * A] + α * KL(π_θ || π_ref)
где r_θ = reward от RM - β * KL
    A = advantage (GAE)
```

**Проблемы PPO:**
- Нужно держать 4 модели: actor, critic, reference, reward
- Critic нестабилен при длинных rollouts
- Reward hacking: модель обманывает RM
- Высокий memory и compute overhead

### 3. Rejection Sampling Fine-Tuning (ReST, STaR)
- Генерируем N вариантов, берём те что прошли верификацию → SFT на них
- Итерируем (self-improvement)
- Проще PPO, работает при verifiable rewards

### 4. DPO (Direct Preference Optimization, 2023)
**Ключевая идея:** Implicit reward через log-ratio политик:
```
L_DPO = -E[log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))]
где y_w = preferred, y_l = rejected
```
- Offline, без reward model
- Стабильнее PPO
- Работает на парных предпочтениях

**Проблемы DPO:**
- Sequence-level reward: каждый токен получает одинаковый сигнал (из последнего reward)
- Не работает с online exploration
- Length bias: длинные ответы могут получать лучший сигнал

### 5. GRPO (Group Relative Policy Optimization, DeepSeek-R1, 2024)
**Идея:** Удалить critic, вычислять advantage через группу:
```
Для промпта q: сгенерировать G ответов {o_1,...,o_G}
advantage_i = (r_i - mean(r)) / std(r)

L_GRPO = E[(min(r_θ/r_θ_old * A, clip(r_θ/r_θ_old, 1-ε, 1+ε) * A)) - β*KL]
```
- Нет critic (3 модели вместо 4)
- Contrastive objective: нужны и правильные и неправильные ответы в группе
- Работает с verifiable rewards (pass/fail)

### 6. DAPO (Direct Advantage Policy Optimization, ByteDance/Seed, 2025)
**Улучшения над GRPO:**
1. **Token-level policy gradient** (не sequence-level): каждый токен получает правильный сигнал
2. **Clip-higher**: асимметричный clipping (ε_low=0.2, ε_high=0.28) — стимулирует exploration
3. **Dynamic sampling**: фильтровать промпты где acc=0 или acc=1 (нет контраста)
4. **Entropy bonus**: предотвращение mode collapse

**DAPO — именно то что исследует Alexander Golubev!**

---

## Детальный разбор: PPO для LLM

### Компоненты
- **Actor (π_θ)**: генерируемая модель, обновляем градиентами
- **Critic (V_φ)**: оценивает ожидаемый return из состояния; отдельная модель
- **Reference (π_ref)**: замороженная копия SFT, для KL constraint
- **Reward Model (RM)**: обученный на human preferences

### Generalized Advantage Estimation (GAE)
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = Σ (γλ)^k * δ_{t+k}
```
Стабилизирует high-variance advantage estimates

### Проблема: когда PPO хуже SFT?
- Малый/хороший датасет: SFT лучше (LoRA, P-tuning)
- Нет good reward model: RM сам плохой → reward hacking
- Простые задачи с четким форматом
- Distribution collapse при плохих гиперпараметрах

### Когда PPO всё ещё актуален (2026)?
- Online RL с верификатором (code execution, math checker)
- Process Reward Model (PRM) вместо ORM
- Когда нужен fine-grained token-level сигнал
- Long-horizon reasoning с промежуточными rewards

---

## GRPO vs DAPO vs PPO

| | PPO | GRPO | DAPO |
|---|---|---|---|
| Critic | ✅ Да | ❌ Нет | ❌ Нет |
| Reward granularity | Token (GAE) | Sequence | Token |
| Clipping | Symmetric | Symmetric | Asymmetric |
| Dynamic sampling | ❌ | ❌ | ✅ |
| Entropy | optional | optional | ✅ built-in |
| Models needed | 4 | 3 | 3 |
| Memory overhead | High | Medium | Medium |

---

## Типичные вопросы на интервью

### Q: Почему RLHF а не просто SFT?
**A:** SFT memorizes примеры, но не generalize. RL позволяет исследовать пространство решений и получать reward за правильность, а не за имитацию. Для verifiable tasks (код, математика) reward можно получить без human annotators.

### Q: Что такое reward hacking и как с ним бороться?
**A:** Модель находит способы максимизировать RM без реальной пользы (например, генерирует длинные но бессодержательные ответы). Борьба: KL constraint от reference policy, iterated reward model training, constitutional AI.

### Q: Объясни GRPO в двух предложениях
**A:** Генерируем G ответов на один промпт, нормируем rewards внутри группы (advantage = (r-mean)/std). Обновляем политику PPO-like clipping, но без critic — advantage из группы.

### Q: Что DAPO улучшает над GRPO?
**A:** (1) Token-level gradient вместо sequence-level — более точный сигнал, (2) asymmetric clipping стимулирует exploration, (3) entropy bonus против collapse, (4) dynamic sampling убирает неинформативные промпты.

---

## Частые ошибки
- ❌ Думать что DPO всегда лучше PPO (нет — DPO offline, плохо для online exploration)
- ❌ Путать sequence-level и token-level reward
- ❌ Не знать что GRPO = PPO без critic + group-based advantage
- ❌ Забыть про KL divergence constraint в PPO
