"""Simple coefficient plot; all intervals come from saved numeric sidecars.

The V3/P31 display convention is estimate with its full HAC interval. This
basic SVG renderer is new presentation code, not an estimator replacement.
"""
from html import escape
import math
from pathlib import Path


def coefficient_svg(path, records, label):
    usable = [r for r in records if all(r.get(k) is not None and math.isfinite(r[k])
                                      for k in ['estimate', 'CI_low', 'CI_high'])]
    if not usable: raise ValueError('NO_FINITE_REPORT_INTERVAL')
    low = min([0.] + [r['CI_low'] for r in usable]); high = max([0.] + [r['CI_high'] for r in usable])
    span = high-low or 1.
    def x(value): return 320+420*(value-low)/span
    height = 80+32*len(usable)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{height}" role="img">',
             f'<title>{escape(label)} — saved estimates and 95% HAC intervals</title>',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="25" font-size="16">{escape(label)}</text>',
             f'<line x1="{x(0)}" x2="{x(0)}" y1="40" y2="{height-20}" stroke="#aaa"/>']
    for i, r in enumerate(usable):
        y=60+32*i
        parts.extend([f'<text x="15" y="{y+4}" font-size="12">{escape(r["model_id"])}</text>',
            f'<line x1="{x(r["CI_low"])}" x2="{x(r["CI_high"])}" y1="{y}" y2="{y}" stroke="#24567a" stroke-width="2"/>',
            f'<circle cx="{x(r["estimate"])}" cy="{y}" r="4" fill="#24567a"/>'])
    parts.append('</svg>'); Path(path).write_text('\n'.join(parts), encoding='utf-8')
