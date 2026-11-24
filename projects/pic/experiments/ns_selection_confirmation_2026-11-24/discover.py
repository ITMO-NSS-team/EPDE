"""Record native search once per paired seed, including exact epoch work clocks."""
from contextlib import redirect_stdout
from pathlib import Path
import sys
from time import monotonic
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'epoch_full_record_2026-11-23'))
from epoch_settings import OPTIONS, epochs, live, trial, whole_record
from epde.interface.search_config import resolve_plain_params

SEEDS = (24112401, 24112402, 24112403)
DONOR_CONFIRMATION = (24112404, 24112405, 24112406)
CHECKPOINT = 100


def discover(seed):
    dest = ROOT / 'results' / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'archive.json.gz').is_file():
        return
    clock = []
    observer = None
    started = monotonic()
    fit_milliseconds = []
    fill_settings = {}

    class Observer(trial.Trial):
        def __init__(self, *args):
            nonlocal observer
            super().__init__(*args)
            observer = self

        def evaluated(self, equation):
            super().evaluated(equation)
            fit_milliseconds.append(round(1000 * (monotonic()-started)))

        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            assert len(fit_milliseconds) == self.calls
            return dict(result, epoch_clock=clock, fit_milliseconds=fit_milliseconds,
                        fill_settings=fill_settings)

    original_fit = live.U.EpdeSearch.fit

    def fit(search, *args, **kwargs):
        fill_settings.update(resolve_plain_params(search._config))
        assert fill_settings['fill_beam_width'] == 1000 and search._config.search_space.equation_terms_max_number == 7
        native_callback = search.optimizer_exec_params['early_stopping_callback']
        def record(snapshot, index):
            stopped = native_callback(snapshot, index)
            clock.append(dict(epoch=int(index)+1, cost=observer.cost, calls=observer.calls,
                              points=len(observer.points), labels=len(observer.labels)))
            print('Search clock', clock[-1], flush=True)
            return stopped
        search.optimizer_exec_params['early_stopping_callback'] = record
        return original_fit(search, *args, **kwargs)

    with (
        (dest/'search.log').open('w', buffering=1) as log,
        redirect_stdout(log),
        patch.object(live.U.EpdeSearch, 'fit', fit),
        patch.object(trial, 'Trial', Observer),
        patch.object(live, 'split', whole_record),
        patch.object(trial, 'folder', return_value=dest),
    ):
        trial.worker('ns', seed, 'candidate', noise_level=0, epoch_budget=epochs('ns'),
                     loader_options=OPTIONS['ns'])


if __name__ == '__main__':
    seed = int(sys.argv[1])
    if seed not in SEEDS + DONOR_CONFIRMATION:
        raise ValueError(seed)
    discover(seed)
