from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    t=(ROOT/'fit.py').read_text(encoding='utf-8')
    t=t.replace('ab=np.zeros((5,n))','ab=np.zeros((5,n),dtype=np.result_type(rho,kappa,p1,p2,float))')
    t=t.replace('from scipy.optimize import minimize','from scipy.optimize import minimize\nfrom score_core import profile_score')
    t=t.replace("try:q,ld=cc.evaluate(theta[:dim],np.exp(theta[dim:2*dim]),theta[2*dim:3*dim],theta[3*dim:])\n        except (np.linalg.LinAlgError,FloatingPointError):failed+=1;return 1e50\n        return float(np.log(q/cc.count)+ld/cc.count)",
        "try:return profile_score(cc,theta[:dim],np.exp(theta[dim:2*dim]),theta[2*dim:3*dim],theta[3*dim:],covariance_band)\n        except (np.linalg.LinAlgError,FloatingPointError):failed+=1;return 1e50,np.zeros(4*dim)")
    t=t.replace("jac='3-point'","jac=True").replace("'maxiter':300","'maxiter':500").replace(",'finite_diff_rel_step':1e-4",'')
    t=t.replace('iterations=int(opt.nit),message=str(opt.message)','iterations=int(opt.nit),gradient=opt.jac.tolist(),message=str(opt.message)')
    p=ROOT/'score_fit.py';assert not p.exists();p.write_text(t,encoding='utf-8')

if __name__=='__main__':main()
