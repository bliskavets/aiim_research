"""Execute a v2 reference plan under a line tracer and emit an annotated copy.

Runs INSIDE an agent_* directory (cwd) so that `from tools_augmented import ...`
and the relative `synthetic_finance.db` resolve exactly as during evaluation.
It actually executes the plan (main()), captures the value of every local
variable right after the line that assigns it, and produces an annotated source
string with a `# <var> = <value>` comment above each assignment plus a one-line
tool-docstring note above calls. Writes a JSON result to argv[2].

    python _plan_trace_runner.py correct_plan_augmented.py /tmp/out.json
"""

import ast
import json
import os
import sys
import tempfile
import traceback


def target_names(node):
    names = []

    def walk(t):
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                walk(e)
        elif isinstance(t, ast.Starred):
            walk(t.value)

    if isinstance(node, ast.Assign):
        for t in node.targets:
            walk(t)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        walk(node.target)
    return names


def tool_docstrings(path="tools_augmented.py"):
    docs = {}
    if os.path.exists(path):
        try:
            tree = ast.parse(open(path).read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    d = ast.get_docstring(node)
                    if d:
                        docs[node.name] = d.strip().splitlines()[0].strip()
        except Exception:
            pass
    return docs


def safe_repr(v):
    try:
        r = repr(v)
    except Exception:
        return "<unrepr>"
    return r if len(r) <= 200 else r[:200] + "…"


def main():
    plan_path, out_json = sys.argv[1], sys.argv[2]
    src = open(plan_path).read()
    plan_abs = os.path.abspath(plan_path)
    # the plan does `from tools_augmented import ...`; make its dir importable
    sys.path.insert(0, os.path.dirname(plan_abs) or os.getcwd())
    result = {"ok": False, "error": None, "computed_answer": None,
              "annotated_source": src, "n_vars_traced": 0}

    tmp_out = tempfile.mktemp()
    sys.argv = [plan_path, "--output", tmp_out]
    code = compile(src, plan_path, "exec")

    post_state = {}          # lineno -> {var: repr} snapshot right after that line ran
    prev = {"line": None}

    def snap(frame):
        return {k: safe_repr(v) for k, v in frame.f_locals.items()}

    def local_trace(frame, event, arg):
        if os.path.abspath(frame.f_code.co_filename) != plan_abs:
            return None
        if event == "line":
            if prev["line"] is not None:
                post_state[prev["line"]] = snap(frame)
            prev["line"] = frame.f_lineno
        elif event == "return":
            if prev["line"] is not None:
                post_state[prev["line"]] = snap(frame)
        return local_trace

    def global_trace(frame, event, arg):
        if event == "call" and os.path.abspath(frame.f_code.co_filename) == plan_abs:
            prev["line"] = None
            return local_trace
        return None

    ns = {"__name__": "__main__", "__file__": plan_path}
    sys.settrace(global_trace)
    try:
        exec(code, ns)
        result["ok"] = True
    except SystemExit:
        result["ok"] = True
    except Exception as e:
        result["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
    finally:
        sys.settrace(None)

    if os.path.exists(tmp_out):
        try:
            result["computed_answer"] = open(tmp_out).read().strip()
        except Exception:
            pass

    # ---- build annotated source ----
    docs = tool_docstrings()
    lines = src.splitlines()
    assigns, calls = {}, {}
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                assigns.setdefault(node.lineno, []).extend(target_names(node))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.setdefault(node.lineno, set()).add(node.func.id)
    except Exception:
        pass

    traced = 0
    out = []
    for i, line in enumerate(lines, start=1):
        indent = line[: len(line) - len(line.lstrip())]
        for fn in sorted(calls.get(i, [])):
            if fn in docs:
                out.append(f"{indent}# ↳ {fn}(): {docs[fn]}")
        for name in assigns.get(i, []):
            val = post_state.get(i, {}).get(name)
            if val is not None:
                out.append(f"{indent}# {name} = {val}")
                traced += 1
        out.append(line)
    result["annotated_source"] = "\n".join(out)
    result["n_vars_traced"] = traced

    open(out_json, "w").write(json.dumps(result))


if __name__ == "__main__":
    main()
