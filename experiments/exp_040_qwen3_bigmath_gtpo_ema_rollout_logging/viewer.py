"""
viewer.py — exp_040 rollout viewer with per-token confidence visualisation.
Usage:  python viewer.py [--port 12000]
Opens:  http://localhost:12000
"""
import os, glob, json, argparse
import numpy as np
from flask import Flask, Response, request, jsonify

app   = Flask(__name__)
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(EXP_DIR, "rollout_logs")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            print("[viewer] Loading Qwen3-4B tokenizer…")
            _tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B", token=HF_TOKEN, trust_remote_code=True)
            print(f"[viewer] Tokenizer ready  vocab={_tokenizer.vocab_size}")
        except Exception as e:
            print(f"[viewer] Tokenizer unavailable: {e}")
            _tokenizer = None
    return _tokenizer

# ── data helpers ──────────────────────────────────────────────────────────────

def step_files():
    return sorted(glob.glob(os.path.join(LOG_DIR, "step_?????.npz")))

def step_id(path):
    return int(os.path.basename(path).replace("step_","").replace(".npz",""))

def load_step_stats(path):
    d = np.load(path, allow_pickle=False)
    ic = d["is_correct"]
    adv = d["advantages"].astype(np.float32)
    mask = d["completion_mask"].astype(np.int8)
    return {
        "step": int(d["step"]),
        "n": len(ic),
        "n_correct": int(ic.sum()),
        "passrate": float(ic.mean()),
        "mean_adv": float(adv.mean()),
        "mean_len": float(mask.sum(axis=1).mean()),
    }

def decode_ids(tok, ids, mask=None):
    """Decode token ids to list of per-token strings (actual text, not BPE symbols)."""
    if mask is not None:
        ids = ids[mask.astype(bool)]
    if tok is None:
        return [str(i) for i in ids.tolist()]
    return [tok.decode([int(i)]) for i in ids.tolist()]

def extract_problem(tok, prompt_ids, prompt_mask):
    """Try to extract the user message from the decoded prompt."""
    if tok is None:
        return None
    try:
        ids = prompt_ids[prompt_mask.astype(bool)].tolist()
        text = tok.decode(ids, skip_special_tokens=False)
        # Qwen chat template: extract between <|im_start|>user\n and <|im_end|>
        import re
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL)
        return m.group(1).strip() if m else text[-500:]
    except Exception:
        return None

def load_step_full(step_id_val):
    path = os.path.join(LOG_DIR, f"step_{step_id_val:05d}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    tok = get_tokenizer()

    B = d["completion_ids"].shape[0]
    is_correct = d["is_correct"].tolist()
    advantages = d["advantages"].astype(np.float32).tolist()
    mask_arr   = d["completion_mask"].astype(np.int8)
    comp_ids   = d["completion_ids"].astype(np.int32)
    topk_lp    = d["topk_log_probs"].astype(np.float32)   # (B, T, 20)
    topk_ids_arr = d["topk_token_ids"].astype(np.int32)   # (B, T, 20)

    # Prompt text (available from step 41+ onwards)
    problem_text = None
    if "prompt_ids" in d.files and "prompt_mask" in d.files:
        problem_text = extract_problem(tok, d["prompt_ids"], d["prompt_mask"])

    rollouts = []
    for i in range(B):
        length = int(mask_arr[i].sum())
        ids_i  = comp_ids[i, :length]
        tokens = decode_ids(tok, ids_i)
        # topk_log_probs for this rollout, valid tokens only
        tlp = topk_lp[i, :length, :].tolist()   # list[T] of list[20]
        rollouts.append({
            "is_correct":  bool(is_correct[i]),
            "advantage":   float(advantages[i]),
            "length":      length,
            "tokens":      tokens,
            "topk_lp":     tlp,          # (T, 20) nested list – floats
        })

    # Sort: correct first
    rollouts.sort(key=lambda r: (0 if r["is_correct"] else 1, -r["advantage"]))

    return {
        "step":         int(d["step"]),
        "problem_text": problem_text,
        "rollouts":     rollouts,
    }

# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/steps")
def api_steps():
    rows = []
    for f in step_files():
        try:
            rows.append(load_step_stats(f))
        except Exception as e:
            rows.append({"step": step_id(f), "error": str(e)})
    return jsonify(rows)

@app.route("/api/step/<int:sid>")
def api_step(sid):
    data = load_step_full(sid)
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)

# ── Main HTML page ────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>exp_040 rollout viewer</title>
<style>
* { box-sizing: border-box; }
body { font-family: monospace; background:#0d1117; color:#c9d1d9; margin:0; padding:16px; }
h1   { color:#58a6ff; margin:0 0 8px; font-size:18px; }
.bar { display:flex; gap:16px; align-items:center; flex-wrap:wrap;
       background:#161b22; padding:10px 14px; border-radius:6px; margin-bottom:12px; }
.bar label { color:#8b949e; font-size:12px; }
.bar select, .bar input[type=range], .bar input[type=number]
  { background:#0d1117; color:#c9d1d9; border:1px solid #30363d;
    border-radius:4px; padding:3px 6px; font-family:monospace; }
.bar select { min-width:260px; }
.bar .val   { color:#58a6ff; font-size:12px; min-width:30px; display:inline-block; }
.problem  { background:#161b22; border-left:3px solid #58a6ff; padding:10px 14px;
            margin-bottom:12px; font-size:12px; white-space:pre-wrap; word-break:break-word;
            max-height:160px; overflow-y:auto; border-radius:0 4px 4px 0; }
.problem.na { color:#8b949e; font-style:italic; }
#rollouts   { display:flex; flex-direction:column; gap:10px; }
.rollout    { background:#161b22; border-radius:6px; padding:10px 14px; }
.rollout-hdr{ display:flex; gap:12px; align-items:center; margin-bottom:6px; font-size:12px; }
.badge      { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; }
.correct    { background:#1a4a1a; color:#3fb950; }
.wrong      { background:#4a1a1a; color:#f85149; }
.tokens     { font-size:13px; line-height:1.9; word-break:break-all; }
.tok        { border-radius:2px; padding:1px 0; cursor:default; }
.stats      { background:#0d1117; padding:6px 10px; border-radius:4px; margin-bottom:10px;
              font-size:12px; color:#8b949e; }
#status     { color:#8b949e; font-size:12px; padding:8px 0; }
</style>
</head>
<body>
<h1>exp_040 — rollout viewer</h1>
<div class="bar">
  <label>Step&nbsp;
    <select id="sel-step" onchange="loadStep()">
      <option value="">Loading…</option>
    </select>
  </label>
  <label>EMA&nbsp;α&nbsp;
    <input type="range" id="ema" min="0" max="1" step="0.05" value="0"
           oninput="document.getElementById('ema-val').textContent=this.value; recolor()">
    <span class="val" id="ema-val">0</span>
  </label>
  <label>top-k&nbsp;
    <input type="range" id="topk" min="1" max="20" step="1" value="20"
           oninput="document.getElementById('topk-val').textContent=this.value; recolor()">
    <span class="val" id="topk-val">20</span>
  </label>
</div>
<div class="problem na" id="problem">Select a step to view rollouts.</div>
<div id="status"></div>
<div id="rollouts"></div>

<script>
// ── Global state ────────────────────────────────────────────────────────────
let currentData = null;   // {step, problem_text, rollouts:[{is_correct,advantage,length,tokens,topk_lp}]}

// ── Populate dropdown ───────────────────────────────────────────────────────
async function loadStepList() {
  const resp = await fetch('/api/steps');
  const steps = await resp.json();
  const sel = document.getElementById('sel-step');
  const prevVal = sel.value;   // '' on first call, step number on refresh
  sel.innerHTML = '';
  if (!steps.length) { sel.innerHTML = '<option>No steps yet</option>'; return; }
  steps.slice().reverse().forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.step;
    const pct = s.passrate != null ? (s.passrate*100).toFixed(0)+'%' : '?';
    opt.textContent = `step ${String(s.step).padStart(5,'0')}  pass=${pct}  adv=${(s.mean_adv||0).toFixed(3)}`;
    sel.appendChild(opt);
  });
  if (prevVal !== '') {
    sel.value = prevVal;   // restore selection without reloading content
  } else {
    loadStep();            // auto-load only on first population
  }
}

// ── Load one step ───────────────────────────────────────────────────────────
async function loadStep() {
  const sid = document.getElementById('sel-step').value;
  if (!sid) return;
  document.getElementById('status').textContent = 'Loading…';
  document.getElementById('rollouts').innerHTML = '';
  try {
    const resp = await fetch('/api/step/' + sid);
    currentData = await resp.json();
    render();
    document.getElementById('status').textContent = '';
  } catch(e) {
    document.getElementById('status').textContent = 'Error: ' + e;
  }
}

// ── Confidence computation ──────────────────────────────────────────────────
function computeConf(topk_lp, k) {
  // topk_lp: T x 20, returns T floats
  return topk_lp.map(row => {
    const slice = row.slice(0, k);
    const mean  = slice.reduce((a,b) => a+b, 0) / k;
    return -mean;   // confidence = -mean_log_prob
  });
}

function applyEMA(conf, alpha) {
  if (alpha === 0) return conf;
  const ema = new Float32Array(conf.length);
  ema[0] = conf[0];
  for (let t = 1; t < conf.length; t++)
    ema[t] = (1 - alpha) * conf[t] + alpha * ema[t-1];
  return Array.from(ema);
}

function normalize(arr) {
  let mn = Infinity, mx = -Infinity;
  arr.forEach(v => { if(v < mn) mn=v; if(v > mx) mx=v; });
  const rng = mx - mn || 1;
  return arr.map(v => (v - mn) / rng);
}

function confToColor(norm) {
  // white → orange → red
  const r = 255;
  const g = Math.round(255 * (1 - norm * 0.85));
  const b = Math.round(255 * (1 - norm));
  const a = 0.15 + 0.75 * norm;
  return `rgba(${r},${g},${b},${a.toFixed(2)})`;
}

// ── Render ──────────────────────────────────────────────────────────────────
function render() {
  if (!currentData) return;
  const probEl = document.getElementById('problem');
  if (currentData.problem_text) {
    probEl.textContent = currentData.problem_text;
    probEl.className = 'problem';
  } else {
    probEl.textContent = 'Prompt not saved in this step\'s data (available from step 41 onwards).';
    probEl.className = 'problem na';
  }

  const container = document.getElementById('rollouts');
  container.innerHTML = '';

  const k     = parseInt(document.getElementById('topk').value);
  const alpha = parseFloat(document.getElementById('ema').value);

  // Stats bar
  const nCorrect = currentData.rollouts.filter(r=>r.is_correct).length;
  const stats = document.createElement('div');
  stats.className = 'stats';
  stats.textContent = `Step ${currentData.step} — ${nCorrect}/${currentData.rollouts.length} correct`;
  container.appendChild(stats);

  currentData.rollouts.forEach((r, ri) => {
    const conf_raw = computeConf(r.topk_lp, k);
    const conf_ema = applyEMA(conf_raw, alpha);
    const conf_n   = normalize(conf_ema);

    const div = document.createElement('div');
    div.className = 'rollout';
    div.dataset.idx = ri;

    const hdr = document.createElement('div');
    hdr.className = 'rollout-hdr';
    const badge = document.createElement('span');
    badge.className = 'badge ' + (r.is_correct ? 'correct' : 'wrong');
    badge.textContent = r.is_correct ? '✓ correct' : '✗ wrong';
    const info = document.createElement('span');
    info.style.color = '#8b949e';
    info.textContent = `adv=${r.advantage.toFixed(3)}  len=${r.length}`;
    hdr.appendChild(badge);
    hdr.appendChild(info);
    div.appendChild(hdr);

    const tokDiv = document.createElement('div');
    tokDiv.className = 'tokens';
    tokDiv.dataset.ri = ri;

    r.tokens.forEach((tok, ti) => {
      const span = document.createElement('span');
      span.className = 'tok';
      span.title = `conf=${conf_ema[ti]!=null ? conf_ema[ti].toFixed(3) : '?'}`;
      span.style.backgroundColor = ti < conf_n.length ? confToColor(conf_n[ti]) : 'transparent';
      // Show token text; handle special tokens
      span.textContent = tok || '';
      tokDiv.appendChild(span);
    });

    div.appendChild(tokDiv);
    container.appendChild(div);
  });
}

// ── Recolor only (sliders changed) ─────────────────────────────────────────
function recolor() {
  if (!currentData) return;
  const k     = parseInt(document.getElementById('topk').value);
  const alpha = parseFloat(document.getElementById('ema').value);
  const tokDivs = document.querySelectorAll('.tokens');
  tokDivs.forEach(tokDiv => {
    const ri  = parseInt(tokDiv.dataset.ri);
    const r   = currentData.rollouts[ri];
    if (!r) return;
    const conf_raw = computeConf(r.topk_lp, k);
    const conf_ema = applyEMA(conf_raw, alpha);
    const conf_n   = normalize(conf_ema);
    const spans = tokDiv.querySelectorAll('.tok');
    spans.forEach((span, ti) => {
      span.style.backgroundColor = ti < conf_n.length ? confToColor(conf_n[ti]) : 'transparent';
      span.title = `conf=${conf_ema[ti]!=null ? conf_ema[ti].toFixed(3) : '?'}`;
    });
  });
}

// ── Stats page (original summary view) ─────────────────────────────────────

// ── Init ────────────────────────────────────────────────────────────────────
loadStepList();
setInterval(loadStepList, 30000);   // refresh dropdown every 30s
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

# ── Summary stats (kept from original viewer) ─────────────────────────────
@app.route("/stats")
def stats_page():
    rows = []
    for f in step_files():
        try:
            rows.append(load_step_stats(f))
        except Exception:
            pass
    total_r = sum(r["n"] for r in rows)
    total_c = sum(r["n_correct"] for r in rows)
    lines = [f"<b>{len(rows)} steps  |  overall passrate: {total_c}/{total_r} = {total_c/max(total_r,1)*100:.1f}%</b>"]
    lines.append("<table border=1 cellpadding=4 style='border-collapse:collapse;font-family:monospace'>")
    lines.append("<tr><th>step</th><th>correct/n</th><th>passrate</th><th>mean_adv</th><th>mean_len</th></tr>")
    for r in reversed(rows[-100:]):
        lines.append(f"<tr><td>{r['step']:05d}</td><td>{r['n_correct']}/{r['n']}</td>"
                     f"<td>{r['passrate']*100:.1f}%</td><td>{r['mean_adv']:+.3f}</td>"
                     f"<td>{r['mean_len']:.0f}</td></tr>")
    lines.append("</table>")
    return Response("<html><body style='font-family:monospace;background:#0d1117;color:#c9d1d9;padding:20px'>"
                   + "\n".join(lines) + "</body></html>", mimetype="text/html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12000)
    args = parser.parse_args()
    print(f"Viewer at http://0.0.0.0:{args.port}  (stats: /stats  api: /api/steps  /api/step/<n>)")
    print(f"Watching: {LOG_DIR}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
