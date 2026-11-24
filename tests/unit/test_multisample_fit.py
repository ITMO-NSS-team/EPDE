"""Check joint fits on records with different scales, ranks and slopes."""
import numpy as np
import pytest

from epde.operators.common.objectives import Discrepancy, FitContext


def _table(samples, masks, metric='wape', weighting='samples'):
    from epde.operators.common.multisample import joint_subset_table
    return joint_subset_table(samples, subsets=masks, discrepancy=metric,
                              weighting=weighting, penalty_coeff=0.2)


def _conflicting_slopes(repeats=1, amplitude=1.0):
    x = np.array([1.0, 2.0, 3.0])
    x2 = np.tile(amplitude * x, repeats)
    return [(31, x[:, None], x, np.ones(x.size)),
            (7, x2[:, None], 3.0 * x2, np.ones(x2.size))]


class TestJointPhysicalFit:
    def test_different_experiments_identify_a_locally_singular_law(self):
        a = np.linspace(0.3, 2.0, 53)
        b = np.linspace(0.7, 1.6, 79)
        samples = [(31, np.column_stack([a, -a]), -0.3 * a, np.ones(a.size)),
                   (7, np.column_stack([b, -4.0 * b]), -0.6 * b, np.ones(b.size))]
        disc, _, width, coefficients, full_rank = _table(samples, [3])
        assert width == 3
        assert full_rank.tolist() == [True]
        np.testing.assert_allclose(coefficients[0], [-0.2, 0.1, 0.0], atol=1e-13)
        assert disc[0] < 1e-12

    def test_a_common_law_reports_the_residual_of_contradictory_samples(self):
        disc, _, _, coefficients, _ = _table(_conflicting_slopes(), [1])
        np.testing.assert_allclose(coefficients[0], [2.0, 0.0], atol=1e-14)
        assert disc[0] == pytest.approx(2.0 / 3.0)

    def test_sample_weighting_does_not_count_repeated_grid_rows_as_new_experiments(self):
        _, _, _, coarse, _ = _table(_conflicting_slopes(), [1])
        _, _, _, repeated, _ = _table(_conflicting_slopes(repeats=4), [1])
        np.testing.assert_allclose(repeated, coarse, rtol=1e-13, atol=1e-14)

    def test_point_weighting_counts_all_rows(self):
        _, _, _, coefficients, _ = _table(_conflicting_slopes(repeats=4), [1],
                                          weighting='points')
        np.testing.assert_allclose(coefficients[0], [2.6, 0.0], atol=1e-14)

    def test_target_and_feature_amplitudes_keep_their_physical_units(self):
        _, _, _, coefficients, _ = _table(_conflicting_slopes(amplitude=10.0), [1])
        # Equal experiment mass does not min-max normalise either experiment:
        # (1*1 + 10*30) / (1**2 + 10**2), with the same x-shape in both.
        np.testing.assert_allclose(coefficients[0], [301.0 / 101.0, 0.0], atol=1e-14)

    def test_reordering_trajectory_blocks_preserves_the_fit_and_scores(self):
        samples = _conflicting_slopes(repeats=4, amplitude=3.0)
        forward = _table(samples, [1, 3])
        backward = _table(samples[::-1], [1, 3])
        for first, second in zip(forward, backward):
            np.testing.assert_array_equal(first, second)

    def test_globally_dependent_columns_are_still_refused(self):
        samples = [(k, np.column_stack([x, 2.0 * x]), 3.0 * x, np.ones(x.size))
                   for k, x in [(31, np.arange(1.0, 8.0)), (7, np.arange(2.0, 11.0))]]
        _, _, _, _, full_rank = _table(samples, [1, 3])
        assert full_rank.tolist() == [True, False]

    def test_an_intercept_is_only_present_when_selected(self):
        samples = [(k, x[:, None], 2.0 * x + 0.7, np.ones(x.size))
                   for k, x in [(31, np.arange(1.0, 8.0)), (7, np.arange(2.0, 11.0))]]
        _, _, _, coefficients, full_rank = _table(samples, [1, 3])
        assert full_rank.all()
        assert coefficients[0][-1] == 0.0
        np.testing.assert_allclose(coefficients[1], [2.0, 0.7], atol=1e-13)

    @pytest.mark.parametrize('bad', [np.nan, -1.0, np.inf, 0.0])
    def test_invalid_weight_mass_fails_loudly(self, bad):
        samples = _conflicting_slopes()
        key, X, y, w = samples[1]
        samples[1] = (key, X, y, np.full_like(w, bad))
        with pytest.raises(ValueError, match='weights'):
            _table(samples, [1])

    def test_a_bad_trajectory_is_not_silently_omitted(self):
        samples = _conflicting_slopes()
        samples[1][1][0, 0] = np.nan
        with pytest.raises(ValueError, match='finite'):
            _table(samples, [1])


class _EquationColumns:
    """The public evaluation protocol consumed by the real discrepancy filler."""
    def __init__(self, samples, coefficients):
        self.targets = {key: y for key, _X, y, _w in samples}
        self.features = {key: X for key, X, _y, _w in samples}
        self.weights_internal = self.weights_final = np.asarray(coefficients)
        self.active_mask = self.weights_internal[:-1] != 0

    def evaluate(self, *, active_only=False):
        if active_only:
            return self.targets, {key: X[:, self.active_mask] for key, X in self.features.items()}
        return self.targets, self.features


class TestJointTableFitnessParity:
    @pytest.mark.parametrize('metric', ['wape', 'l2', 'l2_relative', 'scale_invariant'])
    @pytest.mark.parametrize('mask', [1, 3])
    def test_table_and_fitness_score_the_same_concrete_vector(self, metric, mask):
        samples = _conflicting_slopes(amplitude=3.0)
        disc, _, _, coefficients, _ = _table(samples, [mask], metric=metric)
        equation = _EquationColumns(samples, coefficients[0])
        context = FitContext(g_fun_vals={key: w for key, _X, _y, w in samples},
                             data_shape={key: y.shape for key, _X, y, _w in samples},
                             penalty_coeff=0.2, for_rps=False)
        assert disc[0] == pytest.approx(Discrepancy(metric).compute(equation, context),
                                      rel=1e-13, abs=1e-14)

    def test_the_amplification_guard_uses_the_shared_coefficients(self):
        _, amp, _, _, _ = _table(_conflicting_slopes(), [1])
        # c=2; A is 2/1 on sample one and 2/3 on sample two.
        assert amp[0] == pytest.approx(4.0 / 3.0)

    def test_instability_keeps_each_trajectory_grid_and_detects_local_drift(self):
        from epde.operators.common.multisample import joint_instability_table
        from epde.operators.common.survival import dyad_scores
        from conftest import using_config
        samples, shapes = [], {}
        for key, shape in [(31, (24, 12)), (7, (20, 16))]:
            time, space = np.meshgrid(np.linspace(0, 1, shape[0]),
                                      np.linspace(0, 1, shape[1]), indexing='ij')
            x = (2.0 + np.sin(4 * space) + 0.2 * np.cos(3 * time)).ravel()
            y = (1.0 + 0.8 * time.ravel()) * x
            w = np.ones_like(x)
            samples.append((key, x[:, None], y, w))
            shapes[key] = shape
        with using_config(instability_metric='dyad', equation_terms_max_number=3):
            _, _, _, coefs, _ = _table(samples, [1])
            score = joint_instability_table(samples, coefs, shapes, main_var='u')[0]
            expected = np.mean([np.sum(dyad_scores(X, y, w, shapes[key], fit_intercept=False))
                                for key, X, y, w in samples])
        assert expected > 0.01
        assert score == pytest.approx(expected, rel=1e-13)


def _two_experiment_equation(mode, full_rank_first=False, first_amplitude=1.0, second_points=160):
    import epde
    from epde.interface.equation_translator import translate_equation
    search = epde.EpdeSearch(use_solver=False, multiobjective_mode=True,
                             system_rps_mode=mode, multisample_mode='joint',
                             discrepancy_metric='wape', instability_metric='dyad',
                             equation_terms_max_number=3, device='cpu')
    trajectories = []
    for key, k, n in [(31, 1, 120), (7, 2, second_points)]:
        t = np.linspace(0.0, 2.0, n)
        u = np.exp((-0.2 - 0.1 * k ** 2) * t)
        if key == 31:
            u = first_amplitude * u
        v = -k ** 2 * u
        if full_rank_first and key == 31:
            v = v + 0.3 * np.sin(3 * t)
        du = -0.2 * u + 0.1 * v
        _, domain = search.createDomain((t,), ID=key, boundary_width=0)
        _, trajectory = search.createTrajectory({'u': u, 'v': v}, domain, cache_id=key,
                                                derivs={'u': du[:, None],
                                                        'v': (-k ** 2 * du)[:, None]})
        trajectories.append(trajectory)
    search.create_pool(data=trajectories, max_deriv_order=(1,), data_fun_pow=1)
    system = translate_equation({'u': '1.0 * u{power: 1.0} + 1.0 * v{power: 1.0} '
                                      '+ 0.0 = du/dx0{power: 1.0}',
                                 'v': '1.0 * u{power: 1.0} + 0.0 = dv/dx0{power: 1.0}'},
                                search.pool, all_vars=['u', 'v'])
    equation = system.vals['u']
    equation.reset_state(True)
    equation.target_idx = len(equation.structure) - 1
    return search, equation


def _fit_versions(equation, mode):
    from test_assignment_rps import _soeq_selector
    from epde.operators.common.plain_fit import _versions
    from epde.operators.common.right_part_selection import _equation_variants
    selector = _soeq_selector().suboperators['eq_right_part_selector']
    host, coeff_calc = selector.suboperators['fitness_calculation'], selector.suboperators['coeff_calc']
    if mode == 'plain':
        return _versions(equation, 2, coeff_calc, {}, host, {})
    candidates = _equation_variants(equation, equation.target, frozenset(),
                                   coeff_calc, {}, host, {})
    versions = [candidate.materialize() for candidate in candidates]
    return [version for version in versions if version is not None]


class TestJointFitPaths:
    @pytest.mark.parametrize('weighting, expected_feature', [('samples', 'u'), ('points', 'v')])
    def test_assignment_screen_uses_the_same_sample_mass_as_the_fit(self, weighting, expected_feature):
        from conftest import using_config
        search, equation = _two_experiment_equation('assignment', first_amplitude=3.0,
                                                    second_points=640)
        try:
            with using_config(system_rps_mode='assignment', multisample_mode='joint',
                              multisample_weighting=weighting, instability_metric='dyad',
                              equation_terms_max_number=3,
                              system_rps_params={'selected_terms_max_number': 1, 'screen_size': 1}):
                versions = _fit_versions(equation, 'assignment')
                assert len(versions) == 1
                features = [term.structure[0].label for term in versions[0].structure
                            if term is not versions[0].target]
                assert features == [expected_feature]
        finally:
            search.close()

    @pytest.mark.parametrize('metric', ['wape', 'l2', 'l2_relative', 'scale_invariant'])
    def test_assignment_candidates_report_the_materialised_common_law_error(self, metric):
        from test_assignment_rps import _soeq_selector
        from epde.operators.common.right_part_selection import _equation_variants
        from conftest import using_config
        search, equation = _two_experiment_equation('assignment')
        equation.structure = [equation.structure[0], equation.target]
        equation._invalidate_label_cache()
        try:
            with using_config(system_rps_mode='assignment', multisample_mode='joint',
                              discrepancy_metric=metric, instability_metric='dyad',
                              equation_terms_max_number=3):
                selector = _soeq_selector().suboperators['eq_right_part_selector']
                candidates = _equation_variants(
                    equation, equation.target, frozenset(), selector.suboperators['coeff_calc'], {},
                    selector.suboperators['fitness_calculation'], {})
                built = [(candidate, candidate.materialize()) for candidate in candidates]
                built = [(candidate, version) for candidate, version in built if version is not None]
                assert built
                for candidate, version in built:
                    # Each trajectory alone admits u_t=c*u exactly, but the
                    # two trajectories have different decay rates.
                    assert version.fitness_value > 1e-3
                    assert candidate.fitness_value == pytest.approx(version.fitness_value, rel=1e-12)
                    assert candidate.coefficients_stability == pytest.approx(
                        version.coefficients_stability, rel=1e-12, abs=1e-14)
        finally:
            search.close()

    @pytest.mark.parametrize('mode', ['plain', 'assignment'])
    @pytest.mark.parametrize('reverse', [False, True])
    def test_a_support_identifiable_across_experiments_reaches_the_real_fitness_host(self, mode, reverse):
        import epde.globals as global_var
        from conftest import using_config
        search, equation = _two_experiment_equation(mode)
        try:
            if reverse:
                search._register_trajectories(list(global_var.samples_manager._traj.values())[::-1])
            with using_config(system_rps_mode=mode, multisample_mode='joint',
                              instability_metric='dyad', equation_terms_max_number=3):
                versions = _fit_versions(equation, mode)
                fitted = [version for version in versions
                          if len(version.structure) == 3 and version.weights_final[-1] == 0.0]
                assert fitted, 'the common rank-two support must not be rejected for local rank one'
                for version in fitted:
                    np.testing.assert_allclose(version.weights_final, [-0.2, 0.1, 0.0], atol=1e-11)
                    assert version.fitness_value < 1e-11
                    assert np.isfinite(version.coefficients_stability)
        finally:
            search.close()

    @pytest.mark.parametrize('mode', ['plain', 'assignment'])
    def test_single_sample_preserves_the_existing_fit_exactly(self, mode):
        import epde.globals as global_var
        from conftest import using_config
        search, equation = _two_experiment_equation(mode, full_rank_first=True)
        try:
            search._register_trajectories([global_var.samples_manager._traj[31]])
            results = []
            for fit_mode in ('mean', 'joint'):
                with using_config(system_rps_mode=mode, multisample_mode=fit_mode,
                                  instability_metric='dyad', equation_terms_max_number=3):
                    results.append(_fit_versions(equation, mode))
            assert results[0] and len(results[0]) == len(results[1])
            for original, joint in zip(*results):
                np.testing.assert_array_equal(original.weights_final, joint.weights_final)
                assert original.fitness_value == joint.fitness_value
                assert original.coefficients_stability == joint.coefficients_stability
        finally:
            search.close()

    @pytest.mark.parametrize('mode', ['plain', 'assignment'])
    def test_a_nonfinite_candidate_column_rejects_the_whole_joint_fit(self, mode, monkeypatch):
        from conftest import using_config
        from epde.structure.main_structures import Term
        search, equation = _two_experiment_equation(mode, full_rank_first=True)
        bad_label = equation.structure[0].factors_labels
        evaluate = Term.evaluate

        def one_bad_experiment(term, *args, **kwargs):
            values = evaluate(term, *args, **kwargs)
            if term.factors_labels == bad_label:
                values = dict(values)
                values[7] = np.full_like(values[7], np.inf)
            return values

        monkeypatch.setattr(Term, 'evaluate', one_bad_experiment)
        try:
            with using_config(system_rps_mode=mode, multisample_mode='joint',
                              instability_metric='dyad', equation_terms_max_number=3):
                assert _fit_versions(equation, mode) == []
        finally:
            search.close()
