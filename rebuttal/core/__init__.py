"""Core evaluation and utility modules for SAGE rebuttal experiments."""

from core.helpers import (
    ensure_output_dir,
    build_prompt,
    append_jsonl,
    parse_answer,
    equations_are_equal_new,
)
from core.vllm_client import (
    VLLMCompletionsClient,
    AugEngine,
    EngineNoThink,
    get_engine,
)
from core.math500_eval import Math500Eval
from core.alpaca_eval import AlpacaEval
from core.xstest_eval import XSTest
from core.ifeval_eval import IFEval

__all__ = [
    "ensure_output_dir",
    "build_prompt",
    "append_jsonl",
    "parse_answer",
    "equations_are_equal_new",
    "VLLMCompletionsClient",
    "AugEngine",
    "EngineNoThink",
    "get_engine",
    "Math500Eval",
    "AlpacaEval",
    "XSTest",
    "IFEval",
]
