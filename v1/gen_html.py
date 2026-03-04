"""Generate sweep HTML rows from examples.json."""
import json, html

d = json.load(open("examples.json"))

for feat, sides in d.items():
    print(f"\n<!-- {feat} -->")
    low_rows = sides["low"]
    high_rows = sides["high"]
    for l, h in zip(low_rows, high_rows):
        lp = html.escape(l["prompt"])
        lc = html.escape(l["completion"])
        hp = html.escape(h["prompt"])
        hc = html.escape(h["completion"])
        print(f'  <div class="sweep-row">')
        print(f'    <div class="sweep-cell"><span class="sweep-prompt">{lp}</span> <span class="sweep-text">{lc}</span></div>')
        print(f'    <div class="sweep-cell"><span class="sweep-prompt">{hp}</span> <span class="sweep-text">{hc}</span></div>')
        print(f'  </div>')
