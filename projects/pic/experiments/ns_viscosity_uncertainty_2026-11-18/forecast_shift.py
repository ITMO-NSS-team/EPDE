"""Post hoc attribution of between-generator differences in mean crossed margin."""
import csv
from crossed import table,ROOT,np


def load(tag):
    paths=sorted((ROOT/"draws").glob(f"{tag}_*.npz"))
    pieces=[]
    for p in paths:
        with np.load(p) as z:
            pieces.append({k:z[k] for k in ("seeds","vec","scale","cross")})
    data={k:np.concatenate([d[k] for d in pieces]) for k in pieces[0]}
    assert len(data["seeds"])==128 and np.all(np.diff(data["seeds"])==1)
    return data


def mean_margin(coeff,hold):
    means=[]
    for p in (0,1):
        margins=table(coeff["vec"][p::2,:,:2],hold["cross"][1-p::2],hold["scale"][1-p::2])
        means.append(np.mean(margins))
    return float(np.mean(means))


def main():
    fresh=load("fresh")
    ff=mean_margin(fresh,fresh)
    rows=[]
    for anchor in range(73001,73007):
        boot=load(anchor)
        fb=mean_margin(fresh,boot)
        bf=mean_margin(boot,fresh)
        bb=mean_margin(boot,boot)
        # Average the two orders for swapping the source of each input.
        coeff=((bf-ff)+(bb-fb))/2
        hold=((fb-ff)+(bb-bf))/2
        np.testing.assert_allclose(coeff+hold,bb-ff,atol=1.e-16)
        rows.append(dict(anchor=anchor,fresh_both=ff,fresh_coeff_boot_validation=fb,
                         boot_coeff_fresh_validation=bf,boot_both=bb,coefficient_shift=coeff,
                         validation_shift=hold,interaction=bb-bf-fb+ff))
    with (ROOT/"forecast_shift.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(rows)


if __name__=="__main__":
    main()
