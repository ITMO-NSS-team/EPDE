from common import *
def main():
    t=(ROOT/'robust_fit.py').read_text().replace('np.vander(x,4,','np.vander(x,6,').replace('[:,4:]','[:,6:]').replace('m=n-4','m=n-6').replace('cubic contrasts','quintic contrasts').replace('range(0,n,16)','range(0,n,12)')
    assert not (ROOT/'local6_fit.py').exists();(ROOT/'local6_fit.py').write_text(t,encoding='utf-8')
    for src,dst in [('checks.py','checks6.py'),('screen.py','screen6.py')]:
        t=(ROOT/src).read_text().replace('import robust_fit as F','import local6_fit as F').replace("'robust_fit.py'","'local6_fit.py'").replace("'protocol.txt'","'protocol.txt','local6_protocol.txt'")
        t=t.replace("'checks.json'","'checks6.json'").replace("'checks.py'","'checks6.py'")
        if src=='screen.py':t=t.replace("'screen'","'screen6'").replace("'screen_frozen'","'screen6_frozen'").replace("'screen_summary.json'","'screen6_summary.json'").replace("'screen.py'","'screen6.py'")
        assert not (ROOT/dst).exists();(ROOT/dst).write_text(t,encoding='utf-8')
    dump(ROOT/'local6_provenance.json',dict(source_sha256=hashes('prepare_local6.py','robust_fit.py','checks.py','screen.py','local6_protocol.txt'),generated_sha256=hashes('local6_fit.py','checks6.py','screen6.py')))
if __name__=='__main__':main()
