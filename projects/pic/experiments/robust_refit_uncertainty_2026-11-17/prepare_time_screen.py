from common import *
def main():
    t=(ROOT/'screen.py').read_text().replace('import robust_fit as F','import time_fit as F').replace("'robust_fit.py'","'time_fit.py','state_score.py','repair_fit.py','time_protocol.txt'").replace("'checks.json'","'time_checks.json'").replace("'screen'","'time_screen'").replace("'screen_frozen'","'time_screen_frozen'").replace("'screen_summary.json'","'time_screen_summary.json'").replace("'screen.py'","'time_screen.py'")
    target=ROOT/'time_screen.py';assert not target.exists();target.write_text(t,encoding='utf-8');dump(ROOT/'time_screen_provenance.json',dict(source_sha256=hashes('prepare_time_screen.py','screen.py','time_protocol.txt'),generated_sha256=hashes('time_screen.py')))
if __name__=='__main__':main()
