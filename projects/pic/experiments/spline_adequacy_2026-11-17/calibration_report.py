from adequacy_context import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def summarize(rows):
    n=len(rows);valid=[r for r in rows if r['numerical_status']=='FROZEN'];finite=[r for r in valid if r['cutoff_kind']=='finite'];return dict(n=n,resolved=len(valid),nominal_covered=sum(r['coverage']['nominal'] is True for r in rows),calibrated_covered=sum(r['coverage']['calibrated'] is True for r in rows),whole_space=sum(r['cutoff_kind']=='infinite' for r in valid),finite_cutoff_cases=len(finite),finite_cutoff_covered=sum(r['coverage']['calibrated'] for r in finite),unbounded_calibrated=sum(r['unbounded_certified']['calibrated'] is True for r in rows),total_bootstrap_abstentions=sum(r['abstentions'] for r in rows),finite_cutoff_median=float(np.median([r['cutoff'] for r in finite])) if finite else None)

def main():
    for name in ('bootstrap_audit','target_audit','target_scan_audit'):assert read(ROOT/(name+'.json'))['status']=='PASS'
    general=[read(p) for p in sorted((ROOT/'bootstrap_results').glob('*.json'))];target=[read(p) for p in sorted((ROOT/'target_results').glob('*.json'))];g=summarize(general);t=summarize(target);profiles={p:summarize([r for r in general if r['job']['profile']==p]) for p in ('iid','positive','negative')};scan=read(ROOT/'target_scan_audit.json');summary=dict(status='COMPLETE',general=g,profiles=profiles,target_conditional=t,target_scan_count=len(scan['rows']),target_eligible_count=scan['eligible'],target_selected=scan['selected'],general_results=general,target_results=target,source_sha256=src('calibration_report.py','bootstrap_protocol.txt','target_protocol.txt'),input_sha256={str(ROOT/(n+'.json')):sha(ROOT/(n+'.json')) for n in ('bootstrap_audit','target_audit','target_scan_audit')});dump(ROOT/'bootstrap_summary.json',summary)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained');labels=['Fresh general\n48 cases',f'Conditional p24\n{t["n"]} cases'];data=[g,t];x=np.arange(2)
    axs[0].bar(x-.18,[r['nominal_covered']/r['n'] for r in data],.35,label='Nominal coverage');axs[0].bar(x+.18,[r['finite_cutoff_covered']/r['n'] for r in data],.35,label='Calibrated, finite cutoff');axs[0].bar(x+.18,[r['whole_space']/r['n'] for r in data],.35,bottom=[r['finite_cutoff_covered']/r['n'] for r in data],label='Whole parameter space',color='#c88445');axs[0].axhline(.975,c='black',ls='--',lw=1);axs[0].set(xticks=x,xticklabels=labels,ylim=(0,1.08),ylabel='Fraction of all outer cases',title='Coverage gained versus information lost');axs[0].legend(fontsize=8,loc='lower left')
    for rr,label in [(general,'General'),(target,'Conditional p24')]:axs[1].scatter([r['abstentions'] for r in rr],[r['limit_lr'] for r in rr],s=35,alpha=.7,label=label)
    axs[1].axvline(2.5,c='black',ls='--',lw=1);axs[1].set(xlabel='Abstentions among 128 bootstrap draws',ylabel='Observed positive-boundary LR',title='Three abstentions make rank126 cutoff infinite');axs[1].legend(fontsize=8)
    fig.savefig(ROOT/'bootstrap_overview.png',dpi=180);fig.savefig(ROOT/'bootstrap_overview.pdf');plt.close(fig)
    text=['Selection-aware bootstrap calibration','====================================','',
    'The primary screening rule is unchanged. Bootstrap repeats all fitting,',
    'CV selection, adequacy and noise-information screening. An abstaining',
    'replicate has an infinite statistic, retained in rank126/128. Three or',
    'more abstentions therefore give an infinite cutoff: the region is the',
    'whole parameter space. Such coverage is not useful precision.','',
    'Fresh general cohort:48 ODE fields,32 positive and8 each iid/negative.',
    'All observed fits selected p32; this cohort does not directly test p24.',
    json.dumps(g,indent=2),json.dumps(profiles,indent=2),'',
    'Separate conditional stress test:256 additional positive ODE fields were',
    'fitted; the first up to8 selecting p24 were chosen BEFORE truth scoring.',
    'All scanned fields and selection ordering were independently replayed.',
    'This is conditional p24 performance, not overall population coverage.',
    f"Eligible p24 fields {scan['eligible']}/256; tested {t['n']}.",
    json.dumps(t,indent=2),json.dumps(target,indent=2),'',
    'The selected-model statistic uses a correlated-GCV pilot and estimated',
    'noise. Repeating selection does not make this an exact bootstrap or',
    'guarantee that the pilot reproduces approximation bias and selection risk.',
    'Finite-cutoff coverage, whole-space regions, and abstentions must be',
    'reported separately. No downstream coefficient-width claim is made.',
    'All draw hashes, selected fits, statistic values, ranks and truth-based',
    'memberships were independently checked. No bootstrap draw was dropped.','']
    (ROOT/'bootstrap_findings.txt').write_text('\n'.join(text),encoding='utf-8');print('CALIBRATION REPORT',g,t,profiles,flush=True)

if __name__=='__main__':main()
