"""Generate HTML report from pre-computed sweep data (JSON).

Reads JSON produced by generate_data.py and writes blog/report.html.
"""

import html
import json
import os


def h(text):
    return html.escape(text)


def build_report(data):
    parts = []
    prompts = data["prompts"]
    baseline = data["baseline"]
    K = data["K"]
    K_coarse = K // 2

    parts.append(HTML_HEAD)

    # --- Intro ---
    parts.append('<h2>Baseline</h2>')
    parts.append('<p>The unmodified Qwen2.5-0.5B-Instruct, no weight edits applied.</p>')
    parts.append('<div class="gen-grid">')
    for i, (prompt, text) in enumerate(zip(prompts, baseline)):
        parts.append(f'<div class="gen-card"><div class="gen-prompt">{h(prompt)}</div><div class="gen-text">{h(text)}</div></div>')
    parts.append('</div>')

    parts.append('<hr>')

    # --- Axis Sweeps ---
    parts.append('<h2>The Discovered Axes</h2>')
    parts.append(
        '<p>Each axis is a learned rank-1 direction in weight space. '
        'We sweep the strength from &minus;3 to +3 and observe how the model&rsquo;s output changes. '
        'The same set of prompts produces different output depending on which direction and how far you push.</p>'
    )

    for axis_id, axis_data in data["sweeps"].items():
        kind = axis_data["kind"]
        results = axis_data["results"]

        parts.append(f'<div class="knob-name">Axis {axis_id} <span class="kind-badge {kind}">{kind}</span></div>')

        # Show probes summary
        probe_summary = []
        for r in results:
            if r["probes"] and abs(r["strength"]) > 0:
                s = r["strength"]
                p = r["probes"]
                probe_summary.append(f'{s:+.1f}: KL={p["KL"]:.2f} Var={p["KL_var"]:.2f} Ent={p["Ent"]:.2f} Coh={p["Coh"]:.3f}')
        if probe_summary:
            parts.append(f'<div class="knob-desc">{" &nbsp;|&nbsp; ".join(probe_summary)}</div>')

        # Show negative vs positive side-by-side for each prompt
        # Pick the most interesting strength pair
        neg_results = {r["strength"]: r for r in results if r["strength"] < 0}
        pos_results = {r["strength"]: r for r in results if r["strength"] > 0}

        for neg_s, pos_s in [(-1.5, +1.5), (-3.0, +3.0)]:
            nr = neg_results.get(neg_s)
            pr = pos_results.get(pos_s)
            if not nr or not pr:
                continue

            parts.append(f'<div class="strength-pair-label">{neg_s:+.1f} vs {pos_s:+.1f}</div>')
            parts.append('<div class="sweep">')
            parts.append('<div class="sweep-header">')
            parts.append(f'<div class="sweep-head low">{neg_s:+.1f}</div>')
            parts.append(f'<div class="sweep-head high">{pos_s:+.1f}</div>')
            parts.append('</div>')

            for pi in range(min(len(prompts), 4)):
                parts.append('<div class="sweep-row">')
                parts.append(f'<div class="sweep-cell"><div class="sweep-prompt">{h(prompts[pi])}</div>{h(nr["texts"][pi])}</div>')
                parts.append(f'<div class="sweep-cell"><div class="sweep-prompt">{h(prompts[pi])}</div>{h(pr["texts"][pi])}</div>')
                parts.append('</div>')

            parts.append('</div>')

    parts.append('<hr>')

    # --- Hierarchy ---
    parts.append('<h2>Hierarchical Interactions</h2>')
    parts.append(
        '<p>The coarse and fine axes aren&rsquo;t independent&mdash;they interact through a learned gating mechanism. '
        'Here we fix a coarse axis value and vary the paired fine axis to see how the modulation changes.</p>'
    )

    for hier in data["hierarchy"]:
        ci = hier["coarse_idx"]
        fi = hier["fine_idx"]
        combos = hier["combos"]

        parts.append(f'<div class="knob-name">Coarse[{ci}] &times; Fine[{fi}]</div>')
        parts.append('<div class="hier-container">')

        coarse_vals = sorted(set(c["coarse_val"] for c in combos))
        for cv in coarse_vals:
            label_class = "neg" if cv < 0 else ("pos" if cv > 0 else "")
            parts.append('<div class="hier-panel">')
            parts.append(f'<div class="hier-title {label_class}">Coarse = {cv:+.0f}</div>')

            for combo in combos:
                if combo["coarse_val"] != cv:
                    continue
                fv = combo["fine_val"]
                # Show just first prompt for compactness
                text = combo["texts"][0] if isinstance(combo["texts"], list) else combo["texts"]
                parts.append('<div class="hier-row">')
                parts.append(f'<div class="hier-fine"><span>Fine = {fv:+.0f}</span></div>')
                parts.append(f'<div class="hier-gen">{h(text[:200])}</div>')
                parts.append('</div>')

            parts.append('</div>')

        parts.append('</div>')

    # --- Training curve summary ---
    parts.append('<hr>')
    parts.append('<h2>How It Was Trained</h2>')
    parts.append(
        '<p>The system uses model-based RL to discover these directions. '
        'An actor policy proposes weight edits, the environment (frozen Qwen 0.5B) evaluates '
        'their behavioral effect via probe signals, and a world model ensemble learns to predict '
        'these effects so the actor can &ldquo;dream&rdquo;&mdash;training on imagined edits without '
        'running the real model.</p>'
        '<p>Key design choices that made it work:</p>'
        '<ul>'
        '<li><strong>Selectivity reward</strong>: The agent is rewarded for finding edits that affect '
        'different prompts differently (high coefficient of variation in per-prompt KL), not just '
        'maximizing raw divergence.</li>'
        '<li><strong>Pessimistic ensemble</strong>: Two world models predict probe signals; the actor '
        'uses mean &minus; std as reward, preventing exploitation of world model errors.</li>'
        '<li><strong>Unified loop with dream ramp</strong>: Instead of separate phases, real and dream '
        'rollouts are interleaved with the dream fraction gradually increasing from 0% to 40%.</li>'
        '<li><strong>Sparse layer targeting</strong>: LoRA basis directions target every 4th layer '
        '(12 modules instead of 48), giving each direction a more distinct behavioral signature.</li>'
        '</ul>'
    )

    parts.append(HTML_FOOT)
    return "\n".join(parts)


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DreamerML: Discovering Behavioral Control Axes in Weight Space</title>
<style>
  :root {
    --bg: #ffffff;
    --fg: #111;
    --muted: #999;
    --dim: #666;
    --border: #e5e5e5;
    --blue: #2563eb;
    --rose: #e11d48;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, 'Inter', 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.8;
    font-size: 17px;
    -webkit-font-smoothing: antialiased;
  }

  .container {
    max-width: 920px;
    margin: 0 auto;
    padding: 100px 40px 140px;
  }

  h1 {
    font-size: 2.6em;
    line-height: 1.1;
    margin-bottom: 16px;
    letter-spacing: -0.03em;
    font-weight: 800;
  }

  .subtitle {
    font-size: 1.1em;
    color: var(--dim);
    margin-bottom: 48px;
    line-height: 1.6;
  }

  h2 {
    font-size: 1.5em;
    margin-top: 72px;
    margin-bottom: 20px;
    letter-spacing: -0.02em;
    font-weight: 700;
  }

  p { margin-bottom: 20px; }
  ul { margin: 0 0 20px 24px; }
  li { margin-bottom: 8px; }
  a { color: var(--blue); text-decoration: none; }
  a:hover { text-decoration: underline; }
  em { font-style: italic; }
  strong { font-weight: 600; }

  code {
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 0.84em;
    background: #f5f5f5;
    padding: 2px 7px;
    border-radius: 4px;
  }

  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 56px 0;
  }

  .gen-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 24px 0;
  }

  .gen-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.85em;
  }

  .gen-prompt {
    font-weight: 700;
    font-size: 0.82em;
    color: var(--muted);
    margin-bottom: 6px;
    font-family: 'SF Mono', monospace;
  }

  .gen-text { color: var(--dim); line-height: 1.5; }

  .knob-name {
    font-size: 1.15em;
    font-weight: 700;
    margin-top: 48px;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
  }

  .kind-badge {
    font-size: 0.65em;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    vertical-align: middle;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .kind-badge.coarse { background: #eef2ff; color: var(--blue); }
  .kind-badge.fine { background: #fff0f3; color: var(--rose); }

  .knob-desc {
    color: var(--dim);
    font-size: 0.78em;
    font-family: 'SF Mono', monospace;
    margin-bottom: 16px;
    line-height: 1.6;
  }

  .strength-pair-label {
    font-family: 'SF Mono', monospace;
    font-size: 0.78em;
    font-weight: 600;
    color: var(--muted);
    margin-top: 12px;
    margin-bottom: 4px;
  }

  .sweep {
    margin: 4px 0 24px;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  .sweep-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
  }

  .sweep-row:last-child { border-bottom: none; }

  .sweep-cell {
    padding: 12px 16px;
    border-right: 1px solid var(--border);
    font-size: 0.84em;
    line-height: 1.5;
  }

  .sweep-cell:last-child { border-right: none; }

  .sweep-prompt {
    font-weight: 600;
    font-size: 0.82em;
    color: var(--muted);
    margin-bottom: 4px;
    font-family: 'SF Mono', monospace;
  }

  .sweep-header {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
  }

  .sweep-head {
    font-family: 'SF Mono', monospace;
    font-size: 0.75em;
    font-weight: 700;
    padding: 8px 16px;
    letter-spacing: 0.02em;
    border-right: 1px solid var(--border);
  }

  .sweep-head:last-child { border-right: none; }
  .sweep-head.low { background: #eef2ff; color: var(--blue); }
  .sweep-head.high { background: #fff0f3; color: var(--rose); }

  .hier-container {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin: 20px 0 36px;
  }

  .hier-panel {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  .hier-title {
    padding: 8px 14px;
    font-family: 'SF Mono', monospace;
    font-size: 0.78em;
    font-weight: 700;
    border-bottom: 1px solid var(--border);
  }

  .hier-title.neg { background: #f0f4ff; color: var(--blue); }
  .hier-title.pos { background: #fff0f3; color: var(--rose); }

  .hier-row {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 0.82em;
  }

  .hier-row:last-child { border-bottom: none; }

  .hier-fine {
    font-family: 'SF Mono', monospace;
    font-size: 0.85em;
    font-weight: 600;
    margin-bottom: 4px;
  }

  .hier-gen { color: var(--dim); line-height: 1.45; }

  @media (max-width: 700px) {
    .container { padding: 60px 20px 100px; }
    .hier-container { grid-template-columns: 1fr; }
    .sweep-row { grid-template-columns: 1fr; }
    .sweep-cell { border-right: none; border-bottom: 1px solid var(--border); }
    .gen-grid { grid-template-columns: 1fr; }
    h1 { font-size: 1.8em; }
  }
</style>
</head>
<body>
<div class="container">

<h1>Discovering Behavioral Control Axes in Weight Space</h1>
<p class="subtitle">We used model-based RL to learn 8 low-dimensional control directions over a frozen Qwen2.5-0.5B. Each direction produces a different behavioral shift&mdash;from subtle stylistic changes to complete personality rewrites. Here&rsquo;s what we found.</p>
"""

HTML_FOOT = """
</div>
</body>
</html>"""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="v2/runs/qwen_v2_selectivity3/report_data.json")
    parser.add_argument("--output", type=str, default="blog/report.html")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    report_html = build_report(data)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report_html)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
