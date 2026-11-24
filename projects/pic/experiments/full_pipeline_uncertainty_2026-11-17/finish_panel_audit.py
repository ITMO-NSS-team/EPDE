"""Drain results arriving while the live panel auditor processed its last list."""
from bridge import *
import audit_panel as A


def main():
    original=ROOT/'panel_audit_live.json'
    if not original.exists():original.write_bytes((ROOT/'panel_audit.json').read_bytes())
    paths=sorted((ROOT/'panel_results').glob('*.json'));assert len(paths)==69
    rows=[]
    for path in paths:
        record=ROOT/'panel_audit_records'/path.name
        if record.exists():
            r=read(record);assert r['status']=='PASS';assert r['result_sha256']==sha(path)
        else:
            r=A.one(path);dump(record,r);print(path.stem,r['status'],flush=True)
        rows.append(r)
    out=dict(status='PASS',records=len(rows),rows=rows,
             source_sha256=sources('finish_panel_audit.py','audit_panel.py','panel_calibration.py'),
             input_sha256={str(original):sha(original),**{str(ROOT/'panel_audit_records'/p.name):sha(ROOT/'panel_audit_records'/p.name) for p in paths}},
             note='Preserved live watcher report and independently audited every remaining result.')
    dump(ROOT/'panel_audit.json',out);print('PASS',len(rows),flush=True)


if __name__=='__main__':main()
