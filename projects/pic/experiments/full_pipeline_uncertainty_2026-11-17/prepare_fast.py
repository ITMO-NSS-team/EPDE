from bridge import *

def main():
    text=(ROOT/'full_pipeline_saved.py').read_text().replace('import time\n','import time\nimport compressed_time as Compressed\nfrom exact_metric_pair import paired\n').replace('Temporal.estimate(y)','Compressed.estimate(y)')
    first=text.index('    def metric(');last=text.index('\ndef score(',first)
    text=text[:first]+"    def metric(self,pilot,model,mc=MC):\n        return paired(self,{'one':pilot},model,mc=mc)['one']\n"+text[last:]
    text=text.replace("covariance={'fixed_pilot':e.metric(e.pilot,model),'refit_pilot':e.metric(newpilot,model)}","covariance=paired(e,{'fixed_pilot':e.pilot,'refit_pilot':newpilot},model,mc=MC)")
    text=text.replace("'full_pipeline_saved.py','coverage_decision.txt','run_full_saved.py','pilot_roundoff_protocol.txt'","'full_pipeline_fast.py','coverage_decision.txt','run_full_fast.py','pilot_roundoff_protocol.txt','compressed_time.py','exact_metric_pair.py','exact_acceleration_protocol.txt'")
    generated={'full_pipeline_fast.py':text}
    text=(ROOT/'run_full_saved.py').read_text().replace('run_full_saved.py','run_full_fast.py').replace('from full_pipeline_saved import run','from full_pipeline_fast import run').replace("ROOT/'logs'","ROOT/'logs_fast'").replace('preflight_saved/check.json','preflight_fast/check.json').replace('processes.json','processes_fast.json');text=text.replace("assert read(ROOT/'metric_checks.json')", "assert read(ROOT/'acceleration_checks.json')['status']=='PASS';assert read(ROOT/'metric_checks.json')");generated['run_full_fast.py']=text
    text=(ROOT/'preflight_saved.py').read_text().replace('full_pipeline_saved','full_pipeline_fast').replace('preflight_saved','preflight_fast').replace('max_workers=4','max_workers=2');generated['preflight_fast.py']=text
    text=(ROOT/'audit_full.py').read_text().replace('full_pipeline_saved','full_pipeline_fast').replace('preflight_saved','preflight_fast').replace('processes.json','processes_fast.json').replace("sources('audit_full.py'","sources('audit_full_fast.py'");generated['audit_full_fast.py']=text
    text=(ROOT/'analyze_full.py').read_text().replace('processes.json','processes_fast.json').replace("sources('analyze_full.py')","sources('analyze_full_fast.py')");generated['analyze_full_fast.py']=text
    for name,text in generated.items():compile(text,name,'exec');(ROOT/name).write_text(text,encoding='utf-8')
    dump(ROOT/'fast_provenance.json',dict(source_sha256=sources('prepare_fast.py','full_pipeline_saved.py','run_full_saved.py','preflight_saved.py','audit_full.py','analyze_full.py','exact_metric_pair.py','compressed_time.py','exact_acceleration_protocol.txt'),generated_sha256={str(ROOT/n):sha(ROOT/n) for n in generated}))
if __name__=='__main__':main()
