# Experiment copy of the benchmark's SA.py runner (agent_runners/SA.py),
# extended with OpenRouter provider routing via the OPENROUTER_EXTRA_BODY env
# var (JSON merged into each request as extra_body). Lives in the experiment
# folder so that no experiment code enters the benchmark repository.
import argparse
import json
import os
from pathlib import Path

from typing import List, Dict, Any, Optional

from smolagents.models import OpenAIModel
try:
    from smolagents import Tool, ToolCallingAgent
except ImportError:  # pragma: no cover – smolagents expected in eval env
    raise ImportError(
        "The 'smolagents' Python package is required but not installed. "
        "Install it with `pip install smolagents` before running this script."
    )

# The main thing that is being imported here is the TOOLS list, which is a list of tool functions that are used by the agent.
from tool_proxy import TOOLS


import mlflow
mlflow.set_tracking_uri("http://localhost:7777")
mlflow.set_experiment("finqa_agents")
mlflow.autolog()
mlflow.start_run()



def build_agent(model_name: str = "gpt-3.5-turbo-0613", api_key: Optional[str] = None, api_base: Optional[str] = None) -> ToolCallingAgent:
    """Create and return a *smolagents* ToolCallingAgent wired to our tools."""

    tool_objs = TOOLS
    # Distractor-ablation: expose ONLY tools the reference plan actually uses (core),
    # dropping the augmentation's distractor tools, holding the item otherwise fixed.
    import re as _re
    plan_path = "correct_plan_augmented.py"
    if os.path.exists(plan_path):
        plan = open(plan_path).read()
        def _base(nm): return nm[:-5] if nm.endswith("_tool") else nm
        core = [t for t in TOOLS if _re.search(r"\b" + _re.escape(_base(t.name)) + r"\s*\(", plan)]
        if core:
            tool_objs = core
    print(f"[core-only] exposing {len(tool_objs)}/{len(TOOLS)} tools", flush=True)

    extra_body = json.loads(os.environ.get("OPENROUTER_EXTRA_BODY", "{}"))
    model_kwargs = {"extra_body": extra_body} if extra_body else {}
    model = OpenAIModel(model_id=model_name, api_key=api_key, api_base=api_base, **model_kwargs)

    agent = ToolCallingAgent(
        tools=tool_objs,
        model=model,
        max_steps=10,
        verbosity_level=20,
        return_full_result=True,
    )
    return agent


# ---------------------------------------------------------------------------
# Main CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """Run the LLM agent on the provided system prompt and save the outputs."""

    parser = argparse.ArgumentParser(description="Run the financial QA agent.")
    parser.add_argument("--system_prompt_file", required=True, help="Path to the system prompt text file")
    parser.add_argument("--output", required=True, help="File where the agent's final answer will be written")
    parser.add_argument(
        "--output_verbose",
        required=True,
        help="JSON file where the agent's reasoning & tool-call history will be stored",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo-0613",
        help="OpenAI chat-model name passed through to smolagents",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--api_base",
        default=None,
        help="OpenAI base URL",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 1️⃣  Read system prompt
    # ---------------------------------------------------------------------
    system_prompt = Path(args.system_prompt_file).read_text()

    # ---------------------------------------------------------------------
    # 2️⃣  Build & invoke agent
    # ---------------------------------------------------------------------
    agent = build_agent(model_name=args.model_name, api_key=args.api_key, api_base=args.api_base)

    embedded_answer = agent.run(system_prompt)
    final_answer = embedded_answer.output
    embedded_answer = json.dumps({'output': embedded_answer.output, 'messages': embedded_answer.messages})

    # ---------------------------------------------------------------------
    # 3️⃣  Persist results
    # ---------------------------------------------------------------------
    Path(args.output).write_text(str(final_answer).strip())
    Path(args.output_verbose).write_text(embedded_answer)


if __name__ == "__main__":
    main()
