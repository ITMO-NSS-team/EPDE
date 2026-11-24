"""Find observed reference terms and near misses after each completed search."""
import ast
from functools import lru_cache
import re

from assess import ARMS, DATASETS, ROOT, SEEDS, csv, folder, gzip, json, label_key, load_data, registry


@lru_cache(None)
def references(dataset):
    return registry(dataset,load_data(dataset))


def peak_grid(log):
    marker = 'Frequencies of trigonometric found in the data'
    for line in log.splitlines():
        if line.startswith(marker):
            values = ast.literal_eval(line.split(':',1)[1].strip())
            return values if isinstance(values,dict) else {0:values}
    return {}


def inspect(name, seed, arm):
    dest = folder(name,seed,arm)
    with gzip.open(dest/'archive.json.gz','rt') as stream:
        rec = json.load(stream)
    peaks = peak_grid((dest/'search.log').read_text(errors='replace'))
    requirements, frequencies = references(name)
    labels = [label_key(t,frequencies) for t in rec['labels']]
    points = {frozenset(labels[j] for j,w in terms) for slot,target,terms,a,b,first in rec['points']}
    seen = frozenset().union(*points)
    results = []
    for i in range(len(requirements[0])):
        forms = {row[i] for row in requirements}
        def distance(left,right):
            # One insertion, deletion or replacement costs one structural edit.
            return max(len(left-right),len(right-left))
        distance_to_law = min(distance(form,p) for form in forms for p in points)
        edits = {(len(form-p),len(p-form)) for form in forms for p in points if distance(form,p)==1}
        def family(term):
            return re.sub(r'~[^@*]+@','~*@',term)
        shape_forms = {frozenset(family(t) for t in form) for form in forms}
        topology = any(frozenset(family(t) for t in p) in shape_forms for p in points)
        def within_grid(form):
            for term in form:
                for freq,dim in re.findall(r'~([^@*]+)@(\d+)',term):
                    f = float(freq)
                    if not any(abs(k-f)<=.01*abs(f) for k in peaks.get(int(dim),[])):
                        return False
            return True
        results.append(dict(system=name,seed=seed,arm=arm,law=i,edits=distance_to_law,
                            all_terms_scored=int(any(form<=seen for form in forms)),
                            frequency_grid_covers_form=int(any(within_grid(form) for form in forms)),
                            support_ignoring_frequency=int(topology),one_add=int((1,0) in edits),
                            one_drop=int((0,1) in edits),one_swap=int((1,1) in edits)))
    return results


if __name__=='__main__':
    rows = [r for name in DATASETS for seed in SEEDS for arm in ARMS for r in inspect(name,seed,arm)]
    with (ROOT/'reachability.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
