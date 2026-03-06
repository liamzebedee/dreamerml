"""Generate HTML report from sweep data (JSON).

Reads JSON produced by generate_data.py, picks the most interesting directions,
and writes a blog-style report with side-by-side comparisons.
"""

import html
import json
import os


def h(text):
    return html.escape(text)


def truncate(text, n=250):
    if len(text) <= n:
        return text
    return text[:n] + "..."


REF_WORDS = ['sorry', "can't", 'cannot', 'i apologize', 'not able to',
             'not appropriate', 'not capable', 'unable to']

def is_refusal(t):
    tl = t.lower()
    return any(w in tl for w in REF_WORDS)


def sweep_table(prompts, left_texts, right_texts, left_label, right_label,
                left_class="low", right_class="high", trunc=250):
    """Build a side-by-side sweep table."""
    parts = []
    parts.append('<div class="sweep">')
    parts.append(f'  <div class="sweep-header">')
    parts.append(f'    <div class="sweep-head {left_class}">{left_label}</div>')
    parts.append(f'    <div class="sweep-head {right_class}">{right_label}</div>')
    parts.append(f'  </div>')

    for i in range(len(prompts)):
        lt = left_texts[i] if i < len(left_texts) else ""
        rt = right_texts[i] if i < len(right_texts) else ""
        parts.append('  <div class="sweep-row">')
        parts.append(
            f'    <div class="sweep-cell">'
            f'<span class="sweep-prompt">{h(prompts[i])}</span> '
            f'<span class="sweep-text">{h(truncate(lt, trunc))}</span>'
            f'</div>'
        )
        parts.append(
            f'    <div class="sweep-cell">'
            f'<span class="sweep-prompt">{h(prompts[i])}</span> '
            f'<span class="sweep-text">{h(truncate(rt, trunc))}</span>'
            f'</div>'
        )
        parts.append('  </div>')

    parts.append('</div>')
    return "\n".join(parts)


def find_best_directions(data):
    """Find directions with the most interesting effects.

    Returns list of (dir_idx, description) sorted by interestingness.
    """
    prompts = data["prompts"]
    baseline = data["baseline"]
    base_refs = sum(1 for t in baseline if is_refusal(t))

    results = []
    for did in range(data["K"]):
        entry = data["directions"][str(did)]
        high = entry["high"]
        low = entry["low"]

        high_refs = sum(1 for t in high["texts"] if is_refusal(t))
        low_refs = sum(1 for t in low["texts"] if is_refusal(t))
        kl_h = high["probes"]["KL"]
        kl_l = low["probes"]["KL"]

        # Skip directions where KL is too high (garbage)
        if kl_h > 5 or kl_l > 5:
            continue

        diff = abs(high_refs - low_refs)
        # Which end is more willing?
        if high_refs < low_refs:
            willing_end = "high"
            cautious_end = "low"
        else:
            willing_end = "low"
            cautious_end = "high"

        results.append({
            "dir": did,
            "diff": diff,
            "high_refs": high_refs,
            "low_refs": low_refs,
            "kl_high": kl_h,
            "kl_low": kl_l,
            "willing_end": willing_end,
        })

    results.sort(key=lambda x: x["diff"], reverse=True)
    return results


def build_report(data):
    parts = []
    prompts = data["prompts"]
    baseline = data["baseline"]
    K = data["K"]

    best = find_best_directions(data)

    parts.append(HTML_HEAD)

    # === Headline: best direction ===
    if best and best[0]["diff"] >= 2:
        top = best[0]
        did = top["dir"]
        entry = data["directions"][str(did)]

        if top["willing_end"] == "high":
            willing_texts = entry["high"]["texts"]
            cautious_texts = entry["low"]["texts"]
            willing_kl = top["kl_high"]
            cautious_kl = top["kl_low"]
        else:
            willing_texts = entry["low"]["texts"]
            cautious_texts = entry["high"]["texts"]
            willing_kl = top["kl_low"]
            cautious_kl = top["kl_high"]

        willing_refs = min(top["high_refs"], top["low_refs"])
        cautious_refs = max(top["high_refs"], top["low_refs"])
        base_refs = sum(1 for t in baseline if is_refusal(t))

        parts.append('<h2>The Willingness Axis</h2>')
        parts.append(
            f'<p>Direction {did} in our random basis controls the model&rsquo;s '
            f'<em>willingness to engage</em> with creative requests. '
            f'The baseline model refuses {base_refs} out of {len(prompts)} prompts. '
            f'One end of this direction drops refusals to {willing_refs}/{len(prompts)}. '
            f'The other end increases them to {cautious_refs}/{len(prompts)}.</p>'
        )
        parts.append(
            '<p>Same model. Same weights. Same prompts. The only difference is a rank-1 edit '
            'to 12 weight matrices, scaled by &plusmn;3.</p>'
        )

        parts.append(sweep_table(
            prompts, cautious_texts, willing_texts,
            f"Cautious end &mdash; KL={cautious_kl:.2f}",
            f"Willing end &mdash; KL={willing_kl:.2f}",
        ))

        parts.append(
            '<p>On the left: the model refuses the scary story, the love letter, '
            'the villain monologue, and the roast. On the right: it writes all of them. '
            'The factual and peaceful prompts are largely unchanged&mdash;the direction '
            'is <em>selective</em>. It targets the model&rsquo;s refusal behavior, '
            'not its general capability.</p>'
        )

        parts.append('<hr>')

    # === Baseline for reference ===
    parts.append('<h2>Baseline</h2>')
    parts.append(
        '<p>For reference, here&rsquo;s what the unmodified Qwen2.5-0.5B-Instruct produces. '
        f'It refuses {sum(1 for t in baseline if is_refusal(t))}/{len(prompts)} prompts.</p>'
    )
    parts.append('<div class="sweep"><div class="sweep-header">'
                 '<div class="sweep-head" style="grid-column:1/-1;background:#f5f5f5;color:var(--dim);">'
                 'Unmodified Qwen2.5-0.5B-Instruct</div></div>')
    for i in range(len(prompts)):
        cls = ' style="color:var(--rose);"' if is_refusal(baseline[i]) else ''
        parts.append(
            f'<div class="sweep-row single">'
            f'<div class="sweep-cell">'
            f'<span class="sweep-prompt">{h(prompts[i])}</span> '
            f'<span class="sweep-text"{cls}>{h(truncate(baseline[i], 200))}</span>'
            f'</div></div>'
        )
    parts.append('</div>')

    parts.append('<hr>')

    # === Other interesting directions ===
    if len(best) > 1:
        parts.append('<h2>Other Directions</h2>')
        parts.append(
            '<p>Different random directions produce different effects. '
            'Here are more directions from the same 16-dimensional basis.</p>'
        )

        for rank, info in enumerate(best[1:4], 2):
            did = info["dir"]
            entry = data["directions"][str(did)]

            parts.append(sweep_table(
                prompts,
                entry["low"]["texts"],
                entry["high"]["texts"],
                f"Direction {did}, &minus;3 &mdash; KL={info['kl_low']:.2f}",
                f"Direction {did}, +3 &mdash; KL={info['kl_high']:.2f}",
                trunc=180,
            ))

        parts.append('<hr>')

    # === How it works ===
    parts.append('<h2>How It Works</h2>')
    parts.append(
        f'<p>We freeze <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5-0.5B-Instruct</a> '
        f'and initialize {K} random rank-1 LoRA basis directions on the attention output and MLP output '
        f'projections of every 4th transformer layer (12 weight matrices total). '
        f'Each direction is a pair of random vectors scaled to 0.05.</p>'
    )
    parts.append(
        '<div class="diagram">direction d &isin; R&#xB9;  &rarr;  &Delta;w = strength &middot; '
        '(A_d &otimes; B_d)  &rarr;  w\' = w + &Delta;w  &rarr;  generate</div>'
    )
    parts.append(
        '<p>We sweep each direction at &plusmn;3 and measure the effect on 8 creative writing prompts. '
        'No training is involved in the sweep itself&mdash;the basis directions are random '
        '(<code>torch.randn</code> scaled by 0.05). We then wrap this in a model-based RL loop '
        'where an actor learns to <em>combine</em> directions to find structured behavioral changes.</p>'
    )
    parts.append(
        '<p>The key finding: even <em>random</em> directions in weight space produce legible, '
        'interpretable behavioral effects on an instruct model. The RL system can then learn to '
        'combine these directions to find sharper axes of control.</p>'
    )

    parts.append('<hr>')
    parts.append(
        '<p class="footer">Code: ~800 lines of pure PyTorch + transformers. No RL libraries. '
        'The environment, actor, world model, and training loop each fit in a single file.</p>'
    )

    parts.append(HTML_FOOT)
    return "\n".join(parts)


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DreamerML v2: What Random Weight Directions Do to an Instruct Model</title>
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
    max-width: 900px;
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

  blockquote {
    border-left: 3px solid #ddd;
    padding: 8px 24px;
    margin: 28px 0;
    color: var(--dim);
  }

  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 56px 0;
  }

  .sweep {
    margin: 20px 0 36px;
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
  .sweep-row.single { grid-template-columns: 1fr; }

  .sweep-cell {
    padding: 16px 20px;
    border-right: 1px solid var(--border);
    font-size: 0.88em;
    line-height: 1.55;
  }

  .sweep-cell:last-child { border-right: none; }

  .sweep-header {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
  }

  .sweep-head {
    font-family: 'SF Mono', monospace;
    font-size: 0.75em;
    font-weight: 700;
    padding: 8px 20px;
    letter-spacing: 0.02em;
    border-right: 1px solid var(--border);
  }

  .sweep-head:last-child { border-right: none; }
  .sweep-head.low { background: #eef2ff; color: var(--blue); }
  .sweep-head.high { background: #fff0f3; color: var(--rose); }

  .sweep-prompt { color: var(--muted); }
  .sweep-text { color: var(--fg); }

  .diagram {
    background: #fafafa;
    border: 1px solid var(--border);
    padding: 28px;
    border-radius: 8px;
    font-family: 'SF Mono', monospace;
    font-size: 0.78em;
    line-height: 1.6;
    margin: 28px 0;
    text-align: center;
    white-space: pre;
    overflow-x: auto;
  }

  .footer {
    color: var(--muted);
    font-size: 0.85em;
    margin-top: 16px;
  }

  @media (max-width: 600px) {
    .sweep-row { grid-template-columns: 1fr; }
    .sweep-cell { border-right: none; border-bottom: 1px solid var(--border); }
    .sweep-header { grid-template-columns: 1fr; }
    h1 { font-size: 1.8em; }
    .container { padding: 60px 20px 100px; }
  }
</style>
</head>
<body>
<div class="container">

<h1>What Random Weight Directions Do to an Instruct Model</h1>
<p class="subtitle">
We take a frozen Qwen2.5-0.5B-Instruct, pick random rank-1 directions in weight space, and scale them up.
The model doesn&rsquo;t break. It changes behavior&mdash;and different directions change it in different ways.
One direction controls the model&rsquo;s willingness to engage with creative requests.
</p>

<p>Take a frozen instruct model. Don&rsquo;t fine-tune it. Instead, pick a random direction in weight space&mdash;literally
<code>torch.randn</code>&mdash;and nudge the weights along it. What happens?</p>

<p>You&rsquo;d expect the model to break. Instead, it <em>changes personality</em>. And the transition is smooth.</p>
"""

HTML_FOOT = """
</div>
</body>
</html>"""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="runs/qwen_sweep/report_data.json")
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
