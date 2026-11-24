"""Complete-system identity credit used alongside the existing structural registry."""
from functools import lru_cache
import json
import sys

from epoch_settings import live
from historical_lorenz import historical_label


@lru_cache(maxsize=1)
def lorenz_classifier():
    sys.path.insert(0, str(live.operators.HISTORICAL))
    from identity_hits import Classifier
    return Classifier('lorenz', use_cache=False, progress=False)


def lorenz_identity(archive):
    classifier = lorenz_classifier()
    labels = [historical_label(term) for term in archive['labels']]

    @lru_cache(maxsize=None)
    def equation(point_id):
        _, target, pairs, *_ = archive['points'][point_id]
        terms = [j for j, w in pairs if j != target and w != 0.]
        key = labels[target], tuple(sorted(labels[j] for j in terms if j != 0)), 0 in terms
        return key, classifier.classify(key)['kind'] in ('law', 'identity')

    def hit(front):
        for candidate in front:
            keys = []
            for i in candidate:
                key, accepted = equation(i)
                if not accepted:
                    break
                keys.append(key)
            else:
                # The historical full-system test checks that the law vectors span every law.
                if classifier.full(tuple(keys))['full']:
                    return True
        return False
    return hit


def catalog_diagnostics(dest, rows):
    path = dest/'catalog_matches.json'
    catalog = json.loads(path.read_text()) if path.exists() else {}
    return [dict(r, catalog_front_hit=catalog.get(r['method'], r.get('catalog_front_hit', r['front_hit'])),
                 catalog_generated=catalog.get('generated', r.get('catalog_generated', r['generated']))) for r in rows]
