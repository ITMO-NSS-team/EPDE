from common import *
def main():
    t=(ROOT/'studentize.py').read_text();start=t.index('class Studentizer:');end=t.index('def run(',start)
    t=t[:start]+'from patch_studentizer import Patches as Studentizer\n'+t[end:]
    t=t.replace("ROOT/'studentized_frozen'","ROOT/'studentized_all_frozen'").replace("ROOT/'studentized'","ROOT/'studentized_all'")
    t=t.replace("hashes('studentize.py','studentization_protocol.txt'","hashes('studentize_all.py','studentize.py','patch_studentizer.py','studentization_protocol.txt','studentization_all_protocol.txt'")
    t=t.replace('metrics={};regions={};counts={}',"metrics={};regions={};counts={};covariances={tag:{b:[] for b in e.banks} for tag in ('check','actual')}")
    t=t.replace("w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['check']", "covariances['check'][b].append(cov);w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['check']")
    t=t.replace("w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['actual']", "covariances['actual'][b].append(cov);w,mi=S.metric(cov,e.active);assert w is not None,mi;scores['actual']")
    t=t.replace("fp=ROOT/'studentized_all_frozen'", "for b in e.banks:arrays[b+'_check_covariances']=np.stack(covariances['check'][b])\n    fp=ROOT/'studentized_all_frozen'")
    t=t.replace("ap=ROOT/'studentized_all'", "for b in e.banks:arrays[b+'_actual_covariances']=np.stack(covariances['actual'][b])\n    ap=ROOT/'studentized_all'")
    target=ROOT/'studentize_all.py';assert not target.exists();target.write_text(t,encoding='utf-8')
    dump(ROOT/'student_all_provenance.json',dict(source_sha256=hashes('prepare_student_all.py','studentize.py','patch_studentizer.py','studentization_all_protocol.txt'),generated_sha256=hashes('studentize_all.py')))
if __name__=='__main__':main()
