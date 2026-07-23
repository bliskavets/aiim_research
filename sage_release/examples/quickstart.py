"""Minimal end-to-end example.

Prerequisite: a vLLM server exposing an OpenAI-compatible /v1 endpoint, e.g.

    bash scripts/serve_vllm.sh Qwen/Qwen3-8B-FP8

Then:

    python examples/quickstart.py
"""
import asyncio

from sage import process_query, load_preset


async def main() -> None:
    question = (
        "A rectangle has a perimeter of 24 cm and its length is twice its width. "
        "What is its area in square centimeters? Put the final answer in \\boxed{}."
    )
    kwargs = load_preset("math500")  # best settings + math verification judge
    result = await process_query(
        question,
        model_name="Qwen/Qwen3-8B-FP8",
        base_url="http://localhost:9090/v1",
        **kwargs,
    )
    print("Selected answer:\n", result["output"])
    print(f"\nExplored {len(result['all_answers'])} candidates in total.")


if __name__ == "__main__":
    asyncio.run(main())
