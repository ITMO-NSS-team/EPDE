"""Exercise the same persisted record consumed by notebooks, the CLI and campaign reports."""
import json
import sys
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIC))

from epde import DiscoveryResult
from epde_bench.metrics import score_run
from epde_bench.reconstruction import parse_equation
from epde_bench.report import load_runs
from epde_bench.runner import front_values, record_front, run_one


def test_scored_campaign_record_roundtrip(tmp_path):
    overrides = {'search': {'preprocessing': {'max_deriv_order': [2], 'default_preprocessor_type': 'FD'},
                            'search_space': {'tokens': [], 'system_rps_mode': 'plain', 'equation_terms_max_number': 3,
                                             'equation_factors_max_number': 1, 'data_fun_pow': 2},
                            'evolution': {'engine': 'nsga2', 'population_size': 4, 'training_epochs': 1, 'neighbors_number': 2}}}
    record = run_one('ode', seed=8, overrides=overrides)
    assert record['status'] == 'ok', record.get('traceback')
    result = DiscoveryResult.from_dict(json.loads(json.dumps(record['discovery'])))
    texts, objectives = front_values(result)
    assert texts and objectives and all(texts)
    assert texts == record_front(record) == record['front']
    for candidate, lines in zip(result.front, texts):
        for equation, line in zip(candidate['equations'], lines):
            assert parse_equation(equation) == parse_equation(line)
    truth = [record['problem']['truth']]
    assert score_run(texts, objectives, truth) == record['metrics']
    folder = tmp_path / 'runs' / 'ode'
    folder.mkdir(parents=True)
    output = folder / 'run.json'
    for key in ('front', 'selected'):
        record.pop(key)
    output.write_text(json.dumps(record), encoding='utf-8')
    summary = load_runs(tmp_path).iloc[0]
    i = record['metrics']['selected_index']
    assert summary.selected == (' | '.join(texts[i]) if i is not None else '')


def test_stored_problem_inputs_match_live_compatibility_path():
    import numpy as np
    from epde_bench.problem import Problem, mesh
    from epde_bench.runner import _inputs, build_search

    t = np.linspace(0, 5, 90)
    p = Problem('sample', 'sample', 'ode', 'synthetic', mesh(t), {'u': np.sin(t)}, ('t',),
                token_groups=[('forcing', {'input': np.cos(t)}, True)])
    cfg = {'preprocessing': {'default_preprocessor_type': 'FD', 'max_deriv_order': [1]}}
    with_inputs, sample, declared = _inputs(p, cfg)
    assert declared[0].token_tensors is None
    with_inputs.create_pool(sample, additional_tokens=declared)
    observed = with_inputs._pool_data[0].raw_token('input').copy()
    legacy, trajectory, families = build_search(p, cfg)
    legacy.create_pool([trajectory], additional_tokens=families)
    np.testing.assert_equal(trajectory.raw_token('input'), observed)
    np.testing.assert_equal(observed, np.cos(t))
    legacy.close()
    with_inputs.close()
