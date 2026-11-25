"""Replay the same seed at two assembly epochs under its original total fit budget."""
import argparse
import json

from donor_history import read_archive, score_archive
from discover import CHECKPOINT, DONOR_CONFIRMATION, ROOT


def compare(seed):
    folder = ROOT / 'results' / str(seed)
    archive = read_archive(folder)
    operator = archive['operator']
    budget = operator['full_cost'][-1]
    assert archive['seed'] == seed and archive['input_shape'] == [200, 50, 100]
    assert operator['fill_settings']['fill_beam_width'] == 1000
    assert budget == archive['calls'] + operator['extra_fits'] == operator['epoch_clock'][-1]['cost']
    baseline = json.loads((folder / 'donor_history.json').read_text())
    result = {}
    for epoch in (CHECKPOINT, 125):
        arms = score_archive(archive, epoch)
        for name, row in arms.items():
            assert row['native_fits'] + row['new_fits'] == row['total_fits'] == budget
            if epoch == CHECKPOINT:
                for key, value in baseline[name].items():
                    if key != 'seconds' and row[key] != value:
                        raise ValueError(f'Seed {seed} epoch {epoch} {name} {key}: {row[key]} vs archived {value}')
        result[epoch] = arms
    output = folder / 'assembly_timing.json'
    pending = output.with_suffix('.pending')
    pending.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    pending.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, choices=DONOR_CONFIRMATION)
    compare(parser.parse_args().seed)
