"""Focused checks for independent selection, failure semantics and score algebra."""
import run_outer as S
np=S.np


def main():
    a=dict(bank='local',status='ok',inner=1.)
    b=dict(bank='broad',status='ok',inner=2.)
    assert S.choose([a,b])['bank']=='local'
    assert S.choose([dict(a,inner=2.),b])['bank']=='local'
    assert S.choose([a,dict(b,inner=-1.)])['bank']=='broad'
    failed=dict(a,status='nonconverged')
    assert S.choose([failed,b])['forced']
    assert S.choose([failed,dict(b,inner=None)])['status']=='no_feasible_choice'
    assert S.choose([dict(a,inner=float('nan')),b])['forced']
    # Outer outcomes are neither an argument nor a field consulted by choose.
    assert S.choose([dict(a,outer=-1e8,clean=1e8),dict(b,outer=-1e20,clean=0.)])==S.choose([a,b])
    c=dict(xi=[0,1],yi=2);coef=np.array([1.,2.]);w=np.diag([2.,3.])
    left=np.array([[1.,2.,6.],[1.,3.,8.]])
    right=np.array([[1.,2.,4.],[1.,3.,8.]])
    actual=S.cross([left,right],w,np.array([1.,1.,5.]),c,coef)
    np.testing.assert_allclose(actual,1.)
    # Existing native/constant controls are frozen dependencies, not overwritten.
    import json
    prior=json.loads((S.N.ROOT/'invariant_checks.json').read_text())
    assert prior['all_constants_preserved']
    S.R.dump(S.ROOT/'checks.json',dict(selection_ignores_outer_and_clean=True,
        deterministic_tie_and_failure_rules=True,signed_cross_algebra=True,
        inherited_identity_checks_sha256=S.sha(S.N.ROOT/'invariant_checks.json'),
        source_sha256=S.sources()))
    print('Selection isolation, ties/failures, signed score and inherited identity controls: PASS')


if __name__=='__main__':main()
