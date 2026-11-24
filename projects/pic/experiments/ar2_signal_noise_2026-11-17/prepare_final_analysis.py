"""Repair a reporting-only schema mismatch, preserving the frozen harness."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
source=ROOT/'analyze.py'
target=ROOT/'analyze_final.py'
old="curv=info.get('ar2',{})"
new="curv=info  # AR2 coefficients are a list, not a nested diagnostics object."
text=source.read_text(encoding='utf-8')
assert text.count(old)==1
target.write_text(text.replace(old,new),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data=dict(reason='Reporting-only schema repair: fit_info.ar2 holds coefficient lists. Optional rho SD is absent; no parameter confidence was calculated.',source_sha256={str(Path(__file__)):sha(Path(__file__))},prior_source_sha256={str(source):sha(source)},generated_sha256={str(target):sha(target)})
(ROOT/'final_analysis_provenance.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
