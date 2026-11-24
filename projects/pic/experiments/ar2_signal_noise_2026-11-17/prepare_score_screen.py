from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    t=(ROOT/'screen.py').read_text(encoding='utf-8').replace('import fit as F','import score_fit as F')
    t=t.replace("ROOT/'fit.py',ROOT/'protocol.txt'","ROOT/'fit.py',ROOT/'score_fit.py',ROOT/'score_core.py',ROOT/'score_protocol.txt',ROOT/'protocol.txt'")
    t=t.replace("ROOT/'screen_frozen'","ROOT/'score_frozen'").replace("ROOT/'screen'","ROOT/'score_screen'").replace("ROOT/'screen_summary.json'","ROOT/'score_screen_summary.json'").replace("ROOT/'checks.json'","ROOT/'checks_score.json'")
    t=t.replace('import sys,json,time,importlib.util','import sys,json,time,importlib.util,argparse')
    t=t.replace("    jobs=[(n,'iid')", "    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=7);args=parser.parse_args()\n    jobs=[(n,'iid')")
    t=t.replace('max_workers=7','max_workers=args.workers').replace('records=len(jobs),summary=summary','records=len(jobs),workers=args.workers,summary=summary')
    p=ROOT/'screen_score.py';assert not p.exists();p.write_text(t,encoding='utf-8')

if __name__=='__main__':main()
