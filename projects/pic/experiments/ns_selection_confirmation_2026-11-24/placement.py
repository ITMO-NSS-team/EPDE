"""Trace the ordering of complete systems for seeds gained by joint reassignment."""
import json
import sys

from score import OLD_SEEDS, OPTIONS, ROOT, compare_arms, directory, live, read_archive


def trace(seed):
    dest = ROOT/'results'/str(seed)
    baseline = json.loads((dest/'comparison.json').read_text())['arms']
    assert baseline['exchange_reassignment']['complete_nondominated'] and not baseline['exchange_local_target']['complete_nondominated']
    old = seed in OLD_SEEDS
    archive = read_archive(directory('ns', seed, 0) if old else dest)
    measured = compare_arms(live.load_data('ns', **OPTIONS['ns']), archive, old, placement_counts=True)
    result = {}
    for arm, row in measured['arms'].items():
        for key in ('total_fits', 'complete_generated', 'complete_nondominated'):
            assert row[key] == baseline[arm][key]
        if 'complete_placements' in row:
            result[arm] = row['complete_placements']
    path = dest/'placement.json'
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(result, indent=2), encoding='utf8')
    temp.replace(path)
    print(seed, result['exchange_reassignment'], flush=True)


if __name__ == '__main__':
    trace(int(sys.argv[1]))
