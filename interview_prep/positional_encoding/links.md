# Ресурсы по Positional Encoding

## Основные статьи
- **Attention Is All You Need** (sinusoidal PE): https://arxiv.org/abs/1706.03762
- **RoPE**: Su et al. 2021 — "RoFormer: Enhanced Transformer with Rotary Position Embedding" https://arxiv.org/abs/2104.09864
- **ALiBi**: Press et al. 2021 — "Train Short, Test Long" https://arxiv.org/abs/2108.12409
- **YaRN**: Peng et al. 2023 — https://arxiv.org/abs/2309.00071 ← скачан в этой папке
- **NTK-aware scaling**: блог пост (оригинальный Reddit) → https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
- **LongRoPE**: Ding et al. 2024 — https://arxiv.org/abs/2402.13753

## Обзорные ресурсы
- **Eleuther blog — Rotary Embeddings**: https://blog.eleuther.ai/rotary-embeddings/ (лучшее объяснение RoPE)
- **Lilian Weng — "The Transformer Family"**: https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
- **Кратко про YaRN vs NTK**: https://huggingface.co/blog/yarn

## Практическое
- **LLaMA 3.1 context extension (до 128K)**: https://arxiv.org/abs/2407.21783
- **Mistral 7B long context**: https://mistral.ai/news/mistral-7b/ (использует sliding window + RoPE)
