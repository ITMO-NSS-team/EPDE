"""Verify the paired measurements and summarize each archive before counting wins."""
from collections import defaultdict
from statistics import median
from frozen import ROOT, read_rows, write_csv


def summarize_ac():
    cells = defaultdict(dict)
    for row in read_rows(ROOT/'ac_transfer.csv'):
        key = (int(row['seed']),row['comparison'],int(row['mode']),float(row['diffusivity']))
        draw = int(row['draw'])
        assert draw not in cells[key]
        margin = float(row['margin'])
        assert abs(margin-(float(row['rival_wape'])-float(row['full_wape']))) < 1e-12
        cells[key][draw] = margin
    assert len(cells)==16*3*2
    rows = []
    for (seed,comparison,mode,diffusivity), draws in cells.items():
        assert set(draws)=={0,*range(927201,927217)}
        noisy = [v for k,v in draws.items() if k]
        rows.append(dict(seed=seed,comparison=comparison,mode=mode,diffusivity=diffusivity,
                         clean_margin=draws[0], median_noisy_margin=median(noisy), noisy_wins=sum(v>0 for v in noisy)))
    write_csv(ROOT/'ac_by_archive.csv', rows)


def summarize_wave():
    pairs = defaultdict(dict)
    for row in read_rows(ROOT/'wave_results.csv'):
        key = (int(row['seed']), int(row['draw']))
        assert row['arm'] not in pairs[key]
        assert int(row['fixed_size'])==len(set(row['fixed_ids'].split()))==6
        pairs[key][row['arm']] = row
    assert set(pairs)=={(s,d) for s in range(923701,923713) for d in (0,*range(927101,927117))}
    arms = ('native','selected')
    assert all(set(p)==set(arms) for p in pairs.values())
    rows = []
    for fresh in (False,True):
        for arm in arms:
            group = [p[arm] for (_,draw),p in pairs.items() if bool(draw)==fresh]
            def metric(key):
                return median(float(r[key]) for r in group if r[key]!='')
            rows.append(dict(cohort='fresh' if fresh else 'original',arm=arm,checks=len(group),
                             fixed_hits=sum(int(r['fixed_hit']) for r in group),
                             front_hits=sum(int(r['front_hit']) for r in group),median_front_size=metric('front_size'),
                             median_physical_wape=metric('physical_wape'),median_actual_wape=metric('best_actual_wape'),
                             median_rival_wape=metric('rival_wape')))
    write_csv(ROOT/'wave_summary.csv', rows)


if __name__=='__main__':
    summarize_ac()
    summarize_wave()
