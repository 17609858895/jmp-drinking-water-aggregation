"""E18a: read published regional and group safely managed drinking-water values (rural, urban, total; 2015 and 2024)
from Annex 8 of the JMP 2025 household report (report page 171). Run with Python; the PDF path is an argument.
Output: results/jmp2025_annex8_sm.csv. '-' (estimate not published) is kept as missing; '>99' as 99.5 and '<1' as 0.5."""
import sys, re, csv, os
import fitz

pdf = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'jmp2025_annex8_sm.csv')
text = fitz.open(pdf)[184].get_text()
assert 'Annex 8' in text and 'Safely managed' in text
L = []
for x in text.splitlines():
    x = x.strip()
    m = re.fullmatch(r'(2015|2024) (\S+)', x)         # a year sharing its line with a footnoted first value, e.g. '2015 58,1'
    L.extend([m.group(1), m.group(2)] if m else [x])
i = L.index('SDG regions') + 1
val = re.compile(r'^(-|>99|<1|\d{1,3}(,\d)?)$')
headers = {'SDG regions', 'Other regional groupings', 'Income groupings'}
rows, name, cur = [], [], None
while i < len(L):
    x = L[i]
    if x in ('2015', '2024') and i + 18 < len(L) and all(val.match(v) for v in L[i + 1:i + 19]):
        if name: cur = ' '.join(name); name = []
        v = [re.sub(r',\d$', '', t) for t in L[i + 1:i + 19]]
        conv = lambda t: None if t == '-' else 99.5 if t == '>99' else 0.5 if t == '<1' else float(t)
        rows.append(dict(group=cur, year=int(x), rural=conv(v[0]), urban=conv(v[6]), total=conv(v[12])))
        i += 19; continue
    if x and x not in headers and not val.match(x) and not x.startswith(('Annex', 'Progress')) and not x.isdigit():
        name.append(x)
    i += 1
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['group', 'year', 'rural', 'urban', 'total']); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
