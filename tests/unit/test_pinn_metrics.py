#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contracts for the evaluation layer in ``projects/pic/data/pinn_common.py``.

This layer is a transplant of the reporting protocol in
``~/PycharmProjects/NN`` (``online_snr_pruning.ipynb``): per-seed results
kept unaggregated, arms compared PAIRED over shared seeds, a noise
yardstick measured from the within-arm seed range, and every verdict
stating its own bar.

It is worth pinning down in tests because it decides what gets CALLED a
result. A sign error in ``contrast`` would print "A WINS" for the arm that
lost, and a comparison layer that is confidently backwards is worse than
no comparison layer at all -- the numbers it reports look exactly as
authoritative either way.

The hold-out guard is here too, for the same reason: ``assert_train_only``
is only worth having if it actually FIRES. A guard that silently passes
everything reads identically to a guard that works, right up until the
held-out tail is in the training loss.
"""

import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir))
_DATA_DIR = os.path.join(_REPO_ROOT, 'projects', 'pic', 'data')
_DP_DIR = os.path.join(_DATA_DIR, 'dp')
for _d in (_DATA_DIR, _DP_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# Optional testbed module living under ``projects/``, not in the package.
# Skip rather than error at COLLECTION when it is absent -- a collection error
# takes the whole suite down and forces an --ignore flag. ``importorskip`` is
# not enough: these testbed modules load their siblings by explicit path, so a
# missing one surfaces as FileNotFoundError rather than ImportError.
try:
    import pinn_common as _pinn_common
except (ImportError, OSError) as _exc:      # OSError covers FileNotFoundError
    pytest.skip(f"projects/pic/data/pinn_common.py is unavailable: {_exc}",
                allow_module_level=True)
arm_table = _pinn_common.arm_table
assert_train_only = _pinn_common.assert_train_only
contrast = _pinn_common.contrast
contrast_table = _pinn_common.contrast_table
mde = _pinn_common.mde
paired_t = _pinn_common.paired_t
seeds_needed = _pinn_common.seeds_needed
sign_p = _pinn_common.sign_p
t_crit = _pinn_common.t_crit


# ============================================================ paired t
class TestPairedT:
    """``paired_t`` is the whole reason the arms are run on shared seeds."""

    def test_matches_the_definition(self):
        d = np.array([0.4, -0.1, 0.3, 0.2, 0.0])
        expect = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        assert paired_t(d) == pytest.approx(expect)

    def test_is_nan_when_it_cannot_be_formed(self):
        # One observation has no spread to divide by, and a constant
        # difference has none either. Returning inf here would print as a
        # spectacular result; nan prints as "not resolved", which is the
        # honest reading.
        assert math.isnan(paired_t([0.3]))
        assert math.isnan(paired_t([0.3, 0.3, 0.3]))

    def test_scales_with_sqrt_n_not_with_n(self):
        """Duplicating the seeds must move t by sqrt(2), not by 2 -- this
        is the property that stops a small effect from being talked into
        significance by rerunning the same draws."""
        d = [0.4, -0.1, 0.3, 0.2]
        t1 = paired_t(d)
        t2 = paired_t(d + d)
        # ddof=1 on 8 points instead of 4 shifts sd slightly, so this is
        # a band rather than an equality.
        assert t2 / t1 == pytest.approx(math.sqrt(2), rel=0.12)

    def test_sign_follows_the_mean(self):
        assert paired_t([-0.2, -0.3, -0.1]) < 0
        assert paired_t([0.2, 0.3, 0.1]) > 0


# ============================================================ sign test
class TestSignP:
    def test_unanimous_and_even_splits(self):
        assert sign_p(4, 4) == pytest.approx(2 * (1 / 16.0))
        assert sign_p(0, 4) == pytest.approx(2 * (1 / 16.0))
        assert sign_p(2, 4) == pytest.approx(1.0)

    def test_is_symmetric_in_wins_and_losses(self):
        for n in (3, 5, 6, 10):
            for k in range(n + 1):
                assert sign_p(k, n) == pytest.approx(sign_p(n - k, n))

    def test_never_exceeds_one(self):
        for n in range(1, 12):
            for k in range(n + 1):
                assert 0.0 <= sign_p(k, n) <= 1.0


# ============================================================ the bar
class TestTCrit:
    def test_known_table_values(self):
        # df = n - 1, two-sided 5%.
        assert t_crit(2) == pytest.approx(12.706)
        assert t_crit(4) == pytest.approx(3.182)
        assert t_crit(6) == pytest.approx(2.571)
        assert t_crit(11) == pytest.approx(2.228)

    def test_falls_back_to_normal_for_large_n(self):
        assert t_crit(200) == pytest.approx(1.960)

    def test_is_monotone_decreasing_in_n(self):
        vals = [t_crit(n) for n in range(2, 31)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))


class TestMdeAndSeedsNeeded:
    """These two exist so an unresolved contrast reports what the DESIGN
    could have seen, instead of being written up as 'no difference'."""

    def test_mde_is_the_effect_that_would_exactly_clear_the_bar(self):
        sd, n = 0.02, 6
        eff = mde(sd, n)
        # An effect of exactly `eff` with this sd gives |t| == t_crit.
        t = eff / (sd / math.sqrt(n))
        assert t == pytest.approx(t_crit(n))

    def test_seeds_needed_delivers_what_it_promises(self):
        """At the n it recommends, the SAME effect and spread clear the
        bar that is printed on the same row -- t_crit(n), not 2.

        Checked against the definition rather than against a fresh random
        draw: one draw at the recommended n clears the bar only about half
        the time, so a sampling test here would be measuring the draw.
        """
        for d in ([0.05, 0.4, -0.2, 0.3, 0.1],
                  [1.0, 1.4, 0.6, 1.2],
                  [-0.02, -0.05, 0.01, -0.04, -0.03, -0.01]):
            d = np.asarray(d, dtype=float)
            n = seeds_needed(d)
            assert n >= len(d)            # never advises FEWER than were run
            t_at_n = abs(d.mean()) / (d.std(ddof=1) / math.sqrt(n))
            assert t_at_n >= t_crit(n)
            # and it is minimal -- one seed fewer would fall short, except
            # where the clamp binds because the effect was already big
            # enough and simply had not cleared t_crit at the n run.
            if n > len(d):
                t_below = abs(d.mean()) / (d.std(ddof=1) / math.sqrt(n - 1))
                assert t_below < t_crit(n - 1)

    def test_seeds_needed_agrees_with_the_bar_on_its_own_row(self):
        """DELIBERATE DEVIATION from NN, pinned so it cannot drift back.

        NN uses the 2-sigma approximation ``(2*sd/|mean|)^2``. NN runs
        10-24 seeds, where t_crit is about 2.1 and that is harmless. This
        project runs 3-6, where t_crit is 2.571 to 4.303 -- so NN's
        formula would print "needs ~N seeds" on a row whose own stated bar
        N seeds do not reach. Two numbers on one line that disagree is
        worse than either alone.
        """
        d = np.array([0.05, 0.4, -0.2, 0.3, 0.1])
        n_ours = seeds_needed(d)
        n_nn = max(len(d), int(math.ceil(
            (2 * d.std(ddof=1) / abs(d.mean())) ** 2)))
        assert n_ours >= n_nn          # ours is never the more optimistic
        # NN's number would NOT have cleared the bar it was printed beside.
        t_at_nn = abs(d.mean()) / (d.std(ddof=1) / math.sqrt(n_nn))
        assert t_at_nn < t_crit(n_nn)
        # ours does.
        t_at_ours = abs(d.mean()) / (d.std(ddof=1) / math.sqrt(n_ours))
        assert t_at_ours >= t_crit(n_ours)

    def test_seeds_needed_is_infinite_for_a_zero_effect(self):
        assert seeds_needed([0.1, -0.1]) == float('inf')

    def test_seeds_needed_never_advises_fewer_than_were_run(self):
        """It is only ever printed when the contrast did NOT resolve, so a
        recommendation below the n already run would read as 'this was
        already enough' -- the opposite of what happened."""
        d = [1.0, 1.4, 0.6, 1.2]            # large effect, tiny spread
        assert seeds_needed(d) >= len(d)


# ============================================================ contrast
class TestContrast:
    A_BETTER = [1.0, 1.1, 0.9, 1.2]      # errors: lower is better
    B_WORSE = [2.0, 2.1, 1.9, 2.2]

    def test_difference_is_formed_a_minus_b(self):
        c = contrast("A - B", self.A_BETTER, self.B_WORSE)
        assert c["mean"] == pytest.approx(-1.0)

    def test_lower_is_better_counts_wins_for_a(self):
        c = contrast("A - B", self.A_BETTER, self.B_WORSE)
        assert c["wins"] == 4
        assert c["resolved"] and c["better"]

    def test_the_losing_arm_is_not_reported_as_winning(self):
        """The failure this class exists for: the same two arms, named the
        other way round, must flip the verdict and nothing else."""
        c = contrast("B - A", self.B_WORSE, self.A_BETTER)
        assert c["mean"] == pytest.approx(1.0)
        assert c["wins"] == 0
        assert c["resolved"] and not c["better"]

    def test_higher_is_better_flips_the_win_direction(self):
        c = contrast("A - B", self.A_BETTER, self.B_WORSE,
                     lower_is_better=False)
        assert c["wins"] == 0
        assert not c["better"]

    def test_unpaired_arms_are_refused(self):
        with pytest.raises(ValueError, match="unpaired"):
            contrast("A - B", [1.0, 2.0], [1.0])

    def test_an_unresolved_contrast_is_not_marked_better(self):
        a = [1.0, 3.0, 0.5, 2.5]
        b = [1.1, 2.8, 0.6, 2.7]          # nearly identical, tiny effect
        c = contrast("A - B", a, b)
        if not c["resolved"]:
            assert not c["better"]
            assert c["needed"] > c["n"] or math.isinf(c["needed"])

    def test_wins_and_sign_p_agree_with_each_other(self):
        c = contrast("A - B", self.A_BETTER, self.B_WORSE)
        assert c["sign_p"] == pytest.approx(sign_p(c["wins"], c["n"]))

    def test_carries_its_own_bar(self):
        c = contrast("A - B", self.A_BETTER, self.B_WORSE)
        assert c["t_crit"] == pytest.approx(t_crit(4))
        assert c["resolved"] == (abs(c["t"]) >= c["t_crit"])


# ============================================================ tables
class TestArmTable:
    RESULTS = {"A": [1.0, 1.4, 1.2], "B": [2.0, 2.1, 2.05],
               "probe": [0.0, 9.0, 4.0]}

    def test_yardstick_is_the_mean_within_arm_range(self, capsys):
        noise = arm_table(self.RESULTS, ["A", "B"], noise_arms=["A", "B"])
        capsys.readouterr()
        assert noise == pytest.approx(np.mean([0.4, 0.1]))

    def test_a_deliberately_varied_probe_arm_can_be_excluded(self, capsys):
        """NN excludes recipe probes from the yardstick on purpose: an arm
        that differs from the others by construction would otherwise widen
        the bar every real comparison is judged against."""
        with_probe = arm_table(self.RESULTS, ["A", "B", "probe"])
        without = arm_table(self.RESULTS, ["A", "B", "probe"],
                            noise_arms=["A", "B"])
        capsys.readouterr()
        assert with_probe > without

    def test_prints_the_per_seed_values_not_only_the_mean(self, capsys):
        arm_table(self.RESULTS, ["A", "B"], fmt="%.2f")
        out = capsys.readouterr().out
        for v in ("1.00", "1.40", "1.20"):
            assert v in out

    def test_contrast_table_renders_both_verdict_branches(self, capsys):
        rows = [contrast("A - B", self.RESULTS["A"], self.RESULTS["B"]),
                contrast("A - probe", self.RESULTS["A"],
                         self.RESULTS["probe"])]
        contrast_table(rows)
        out = capsys.readouterr().out
        assert "A WINS" in out
        assert "not resolved" in out
        assert "wins" in out


# ============================================================ hold-out
class TestAssertTrainOnly:
    """The guard is the reason a future edit that widens one slice stops
    the run instead of quietly improving the numbers."""

    def test_fires_on_a_leak(self):
        coords = torch.linspace(0.0, 1.0, 11)
        with pytest.raises(AssertionError, match="TRAIN/TEST LEAK"):
            assert_train_only("collocation", coords, 0.8)

    def test_names_the_offending_tensor_and_both_numbers(self):
        with pytest.raises(AssertionError) as e:
            assert_train_only("bc", torch.tensor([0.0, 0.95]), 0.8)
        msg = str(e.value)
        assert "'bc'" in msg and "0.95" in msg and "0.8" in msg

    def test_passes_at_exactly_t_split(self):
        coords = torch.tensor([0.0, 0.4, 0.8])
        assert assert_train_only("data", coords, 0.8) is coords

    def test_tolerance_absorbs_float32_representation_only(self):
        t_split = 0.8
        # float32 round-trip of t_split itself must not trip the guard...
        c = torch.tensor([np.float32(t_split)], dtype=torch.float32)
        assert_train_only("fd", c, t_split)
        # ...but a real test point must, even a nearby one.
        with pytest.raises(AssertionError):
            assert_train_only("fd", torch.tensor([t_split + 1e-3]), t_split)

    def test_accepts_numpy_as_well_as_tensors(self):
        with pytest.raises(AssertionError):
            assert_train_only("windows", np.array([0.0, 0.9]), 0.8)
