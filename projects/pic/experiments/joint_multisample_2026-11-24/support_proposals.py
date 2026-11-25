"""Catalog-blind predictor proposals for the joint Pareto pilot."""
from copy import deepcopy
from itertools import combinations
from unittest.mock import patch

from epde.operators.common import fill_prune, plain_fit
from epde.operators.common.right_part_selection import _related_to_target


class SupportProposals:
    def __init__(self, native_versions, refiner):
        self.native_versions = native_versions
        self.refiner = refiner
        self.streams = {}
        self.rotation = 0
        self.counts = {'deletions': 0, 'additive': 0}

    def predictors(self, eq, cap, partners):
        target = eq.target
        current = tuple(term for term in eq.structure if term is not target)
        if len(current) > 1:
            for index in range(len(current)):
                yield 'deletions', current[:index] + current[index+1:]

        # Sample every allowed size on each visit. Traversing the discrete
        # library in its own order avoids a residual-based cutoff for proposals.
        if len(target.structure) != 1:
            return
        setting = fill_prune.factors_setting(eq)
        key = (eq.main_var_to_explain, target.factors_labels, partners, cap, repr(setting))
        if key not in self.streams:
            atoms = [term for term in fill_prune.library_terms(eq.pool, setting)
                     if len(term.structure) == 1 and term.factors_labels != target.factors_labels
                     and term.factors_labels not in partners and not _related_to_target(term, target)]
            self.streams[key] = [combinations(atoms, n) for n in range(1, min(len(atoms), cap)+1)]
        for stream in self.streams[key]:
            terms = next(stream, None)
            if terms is not None:
                yield 'additive', terms

    def fit(self, eq, args, partners):
        versions = self.refiner.refine(self.native_versions(eq, *args), args)
        seen_points = {self.refiner.point(v) for v in versions}
        seen_terms = {frozenset(t.factors_labels for t in eq.structure if t is not eq.target)}
        cap = min(args[0], int(eq.metaparameters['max_terms_number']['value']) - 1)
        for kind, predictors in self.predictors(eq, cap, partners):
            labels = frozenset(term.factors_labels for term in predictors)
            if labels in seen_terms:
                continue
            seen_terms.add(labels)
            self.counts[kind] += 1
            variant = eq.clone_shell()
            variant.structure = [deepcopy(t) for t in (eq.target, *predictors)]
            variant.reset_state()
            variant._invalidate_label_cache()
            plain_fit._install_target(variant, variant.structure[0])
            alternatives = self.native_versions(variant, *args)
            for fitted in self.refiner.refine(alternatives, args):
                pid = self.refiner.point(fitted)
                if pid not in seen_points:
                    versions.append(fitted)
                    seen_points.add(pid)
        return versions

    def apply(self, native_apply, system, *args):
        # apply_plain installs all targets before requesting equation versions.
        # Expand one structural neighborhood per system visit to avoid taking
        # a Cartesian product of simultaneous library sweeps in coupled systems.
        selected = None
        def fit_slot(eq, *fit_args):
            nonlocal selected
            if selected is None:
                pending = [other for other in system.vals if other.target is not None and not plain_fit._fitted(other)]
                selected = pending[self.rotation % len(pending)] if pending else eq
                if len(pending) > 1:
                    self.rotation += 1
            if eq is not selected:
                return self.refiner.refine(self.native_versions(eq, *fit_args), fit_args)
            partners = frozenset(other.target.factors_labels for other in system.vals
                                 if other is not eq and other.target is not None)
            return self.fit(eq, fit_args, partners)

        with patch.object(plain_fit, '_versions', fit_slot):
            return native_apply(system, *args)
