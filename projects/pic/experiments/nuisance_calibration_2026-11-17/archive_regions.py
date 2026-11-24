"""Descriptive comparator only:12 previously evaluated observed datasets."""
from calibration_context import *
from scipy.linalg import solve


def main():
    rows=[];nominal=-2*np.log(.025)
    for path in sorted((PREVIOUS/'point_frozen').glob('*.json')):
        d=read(path);old=read(PREVIOUS/'results'/path.name);theta=np.array([np.log(d['model']['u'][0]),np.arctanh(d['model']['u'][1][0])]);samples=np.array([[np.log(r['model']['u'][0]),np.arctanh(r['model']['u'][1][0])] for r in d['refits']]);mean=samples.mean(axis=0);cov=np.cov(samples.T);center=2*theta-mean;truth=np.array([np.log(d['model']['u'][0]/old['sigma_truth_ratio']),np.arctanh(d['model']['u'][1][0]-old['rho_error'])]);ss=samples-mean;rad=np.einsum('bi,ib->b',ss,solve(cov,ss.T));cut=float(sorted(rad)[125]);delta=truth-center;basic=float(delta@solve(cov,delta));delta0=truth-theta;uncorrected=float(delta0@solve(cov,delta0));rows.append(dict(key=path.stem,cohort=old['cohort'],profile=d['profile'],center=center,observed_theta=theta,true_theta=truth,covariance=cov,basic_score=basic,uncorrected_score=uncorrected,empirical_cutoff=cut,nominal_cutoff=nominal,basic_empirical_covered=basic<=cut,basic_nominal_covered=basic<=nominal,uncorrected_nominal_covered=uncorrected<=nominal,old_hull_covered=old['nuisance_truth_in_basic_convex_hull'],input_sha256={str(path):sha(path),str(PREVIOUS/'results'/path.name):sha(PREVIOUS/'results'/path.name)}))
    dump(ROOT/'archival_ellipsoids.json',dict(status='COMPLETE',cohort='previously_evaluated_not_fresh',rows=rows,source_sha256=src('archive_regions.py','protocol.txt')));print('ARCHIVAL',[(r['key'],r['basic_empirical_covered'],round(r['basic_score'],2),round(r['empirical_cutoff'],2)) for r in rows],flush=True)


if __name__=='__main__':main()
