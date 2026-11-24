from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    t=(ROOT/'score_fit.py').read_text(encoding='utf-8')
    t=t.replace('class Contrasts(Old.Contrasts):','class Contrasts(Old.Contrasts):\n    def __init__(self,y,axis):\n        super().__init__(y);self.fields=[self.fields[axis]];self.count=self.fields[0].size\n')
    t=t.replace('def estimate(y,warm=None):','def estimate(y,axis,warm):').replace('cc=Contrasts(y);','cc=Contrasts(y,axis);').replace('dim=y.ndim;','dim=len(cc.fields);')
    t=t.replace("    if warm is None:_,warm=Old.estimate(y)\n",'')
    p=ROOT/'axis_fit.py';assert not p.exists();p.write_text(t,encoding='utf-8')

if __name__=='__main__':main()
