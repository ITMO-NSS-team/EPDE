from bridge import *

def main():
    text=(PREVIOUS/'time_fit.py').read_text().replace('from scipy.optimize import minimize','from scipy.optimize import minimize\nfrom scipy.linalg import qr\nfrom copy import copy')
    needle='    lo=np.array'
    insert="    original_cc=cc;compressed=False\n    data=cc.fields[0]\n    if data.shape[1]>2*data.shape[0]:\n        _,rmat=qr(data.T,mode='economic',check_finite=False);cc=copy(original_cc);cc.fields=[rmat.T*np.sqrt(data.shape[0]/data.shape[1])];cc.count=data.shape[0]**2;compressed=True\n"
    text=text.replace(needle,insert+needle).replace("best=min(fits,key=lambda f:f['value']);u=", "best=min(fits,key=lambda f:f['value']);cc=original_cc;u=")
    text=text.replace("dict(status=status,sigma=float(sigma)","dict(compressed=compressed,status=status,sigma=float(sigma)")
    compile(text,'compressed_time.py','exec');(ROOT/'compressed_time.py').write_text(text,encoding='utf-8');dump(ROOT/'compressed_provenance.json',dict(source_sha256=sources('prepare_compressed.py'),input_sha256={str(PREVIOUS/'time_fit.py'):sha(PREVIOUS/'time_fit.py')},generated_sha256={str(ROOT/'compressed_time.py'):sha(ROOT/'compressed_time.py')}))
if __name__=='__main__':main()
