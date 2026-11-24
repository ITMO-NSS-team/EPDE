from bridge import *

def main():
    src=(ROOT/'full_pipeline.py').read_text();src=src.replace("sources('full_pipeline.py','coverage_decision.txt','run_full.py')","sources('full_pipeline_saved.py','coverage_decision.txt','run_full_saved.py','pilot_roundoff_protocol.txt')")
    src=src.replace("    for b in e.banks:arrays[b+'_observed']", "    for v,y in e.pilot.items():arrays['anchor_pilot_'+v]=y\n    for b in e.banks:arrays[b+'_observed']")
    src=src.replace("        covariance={'fixed_pilot'", "        if j in (0,BOOTSTRAP-1):\n            for v,y in newpilot.items():arrays[f'bootstrap_pilot_{j}_'+v]=y\n        covariance={'fixed_pilot'")
    generated={'full_pipeline_saved.py':src}
    src=(ROOT/'run_full.py').read_text().replace("ROOT/'run_full.py'","ROOT/'run_full_saved.py'").replace('from full_pipeline import run','from full_pipeline_saved import run').replace("ROOT/'preflight/check.json'","ROOT/'preflight_saved/check.json'").replace("sources('run_full.py'","sources('run_full_saved.py'");generated['run_full_saved.py']=src
    src=(ROOT/'preflight.py').read_text().replace('import full_pipeline as X','import full_pipeline_saved as X').replace("ROOT/'preflight'","ROOT/'preflight_saved'").replace("ROOT/'preflight/check.json'","ROOT/'preflight_saved/check.json'").replace("'preflight.py','full_pipeline.py'","'preflight_saved.py','full_pipeline_saved.py'");generated['preflight_saved.py']=src
    src=(ROOT/'audit_factorial.py').read_text();needle="    assert {tag:{v:R.digest(y) for v,y in pil.items()} for tag,pil in pilots.items()}==f['pilot_sha256']"
    insertion="    repro=read(ROOT/'pilot_reproducibility.json');match=next(r for r in repro['comparisons'] if r['seed']==seed and r['matches_original']);path=Path(match['canonical'])\n    with np.load(path) as z:saved={v:z[v] for v in z.files}\n    for v,y in saved.items():np.testing.assert_allclose(pilots['correlated'][v],y,rtol=0.,atol=1e-9*np.max(np.abs(y)))\n    pilots['correlated']=saved\n"
    src=src.replace(needle,insertion+needle).replace("ROOT/'factorial_audit'","ROOT/'factorial_audit_saved'").replace("ROOT/'audit_factorial.json'","ROOT/'audit_factorial_saved.json'").replace("sources('audit_factorial.py')","sources('audit_factorial_saved.py','pilot_roundoff_protocol.txt')");generated['audit_factorial_saved.py']=src
    for name,text in generated.items():compile(text,name,'exec');(ROOT/name).write_text(text,encoding='utf-8')
    dump(ROOT/'saved_provenance.json',dict(source_sha256=sources('prepare_saved.py','full_pipeline.py','run_full.py','preflight.py','audit_factorial.py','pilot_roundoff_protocol.txt'),generated_sha256={str(ROOT/n):sha(ROOT/n) for n in generated}))
if __name__=='__main__':main()
