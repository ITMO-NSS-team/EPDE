"""``evolution.engine='nsga2'``: NSGA-II on EPDE's systems of equations.

K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist
multiobjective genetic algorithm: NSGA-II", IEEE Trans. Evol. Comput. 6(2),
2002. One generation: binary tournaments on (level, crowding) pick the
parents, crossover and mutation make about as many offspring, and parents and
offspring together are sorted into non-dominated levels and cut back to the
population size -- whole levels first, the level that overflows by crowding.

With per-record objectives, dominance compares all record/equation errors and
instabilities, including for fits with the same terms. The archive keeps
distinct fits at equal objective values. With the default mean discrepancy,
dominance is ``check_dominance`` (the same
structure of an equation is the same equation), crowding is measured in each
equation's own plane (``operators.plane_crowding``), and the sort never loses a
member of a dominance cycle (``supplementary.cycle_safe_nds``). Under
``evolution.niching='per_equation'`` the truncation keeps the holders of every
equation's own front first (``NSGAOptimizer._protected_first``) and the
tournament compares per-equation plane ranks (``operators.plane_ranks``).
Under ``evolution.niching='per_variable'`` (the default of plain mode) every
equation is ranked in its own plane only -- truncation, tournaments, the
assembly crossover and the initial population (``per_variable``).

An archive keeps every evaluated system that no archived system dominates,
and after the fit the archive is sorted in with the last generation, so a
system found once cannot be crowded out later. For one equation the first
Pareto level is then exactly the archive. For several equations dominance is
not transitive (the same structure of one equation is never compared), so an
archived system can be evicted by one that a third later evicts, and the
first level is the archive's sort rather than the archive itself. One epoch runs as many generations as
make the offspring of one MOEA/DD epoch at the same population size, so
``training_epochs`` -- and ``early_stopping_patience`` -- mean the same budget
for both engines.
"""
import warnings
from copy import deepcopy

import numpy as np

import epde.globals as global_var
from epde import _loop_stats
from epde.optimizers.moeadd.moeadd import MOEADDOptimizer
from epde.optimizers.moeadd.supplementary import (DominanceRows, _dominance_view, check_dominance,
                                                  cycle_safe_ndl_update, cycle_safe_nds)
from epde.optimizers.nsga.operators import plane_crowding


def _values_dominate(system, other) -> bool:
    """``system``'s objective values are no worse than ``other``'s and better
    in one, compared value by value (no structure is skipped)."""
    new, old = np.asarray(system.obj_fun, dtype=float), np.asarray(other.obj_fun, dtype=float)
    return bool(new.shape == old.shape and np.all(new <= old) and np.any(new < old))


def _fitted_identity(system):
    """Target, unrounded active terms, physical coefficients and intercept in equation order."""
    def term_key(term):
        return tuple(sorted((f.ftype, f.label, tuple(f.params)) for f in term.structure))
    result = []
    for eq in system.vals:
        terms = tuple(sorted((term_key(term), float(eq.weights_final[eq.weight_index(i, eq.target_idx)]))
                             for i, term in enumerate(eq.structure) if i != eq.target_idx
                             and eq.weights_final[eq.weight_index(i, eq.target_idx)] != 0))
        result.append((eq.main_var_to_explain, term_key(eq.target), terms, float(eq.weights_final[-1])))
    return tuple(result)


def moeadd_parents_per_sector(population_size: int, parents_fraction: float = 0.2) -> int:
    """How many offspring one MOEA/DD sector makes (``MOEADDSelection``'s
    parent count): the fraction of the population, rounded up to even, at
    least 2 and at most the population."""
    counted = int(population_size * parents_fraction)
    counted = counted if not counted % 2 else counted + 1
    return min(max(2, counted), population_size)


class NSGAOptimizer(MOEADDOptimizer):
    """NSGA-II with MOEA/DD's population build, initial sorting and post-fit
    interface (``pareto_levels``, ``_pareto_history``, ``plot_pareto``).

    The weight vectors the parent class builds are used only by the initial
    sorting's 'auto' population cut; the epochs never read them.
    """
    # The archive's DominanceRows, the archive list they were built for, and
    # the insert count (which member the loop-stats check reads).
    _rows = None
    _rows_of = None
    _inserts = 0

    def __init__(self, population_instruct, pop_size, solution_params, H=None,
                 neighbors_number=3, nds_method=cycle_safe_nds, ndl_update=cycle_safe_ndl_update,
                 passed_population=None, best_sol_vals=None, weights_assigner='marriage',
                 population_floor=None):
        super().__init__(population_instruct, pop_size, solution_params,
                         pop_size - 1 if H is None else H, neighbors_number,
                         nds_method=nds_method, ndl_update=ndl_update,
                         passed_population=passed_population, best_sol_vals=best_sol_vals,
                         weights_assigner=weights_assigner, population_floor=population_floor)
        self._archive = []
        self._record_archive_values = None
        self._mu = None

    # ------------------------------------------------------------------ archive
    def _archive_insert(self, system) -> bool:
        """Add ``system`` unless an archived one dominates it, dropping those it
        dominates. True when it entered.

        One arrangement per structure: a system whose structures are archived
        already (the same terms, maybe solved for other targets) replaces the
        archived one only when its own objectives dominate that one's, value by
        value (``check_dominance`` never compares the same structure), and no
        other archived system dominates it.

        The members are compared all at once on their cached dominance rows
        (``DominanceRows``; archived systems are not changed in place), and one
        by one only when a view does not fit the rows."""
        self._inserts += 1
        label = system.equations_labels
        view = _dominance_view(system)
        if view[2] is None:
            return self._insert_record_fit(system, view[1])
        rows = self._archive_rows(view)
        if rows is None:
            return self._archive_insert_scalar(system, label)
        obj, label_ids, system_id = rows.row(view, label)
        twins = rows.twins(system_id)
        if twins.any() and not _values_dominate(system, self._archive[int(np.flatnonzero(twins)[0])]):
            return False
        if np.any(rows.dominating(obj, label_ids) & ~twins):
            return False
        keep = ~twins & ~rows.dominated(obj, label_ids)
        self._archive = [member for member, kept in zip(self._archive, keep) if kept] + [system]
        rows.keep(keep)
        rows.append(view, label)
        self._rows_of = self._archive
        return True

    def _insert_record_fit(self, system, values):
        """Strict Pareto archive of distinct fitted systems, including objective ties."""
        from epde.operators.common.objectives import LOSS_NAN_VAL
        if not np.isfinite(values).all() or np.any(values >= LOSS_NAN_VAL):
            return False
        rows = self._record_archive_values
        if rows is None:
            rows = np.empty((0, len(values)))
        no_worse = np.all(rows <= values, axis=1)
        if np.any(no_worse & np.any(rows < values, axis=1)):
            return False
        ties = np.flatnonzero(no_worse)
        if len(ties):
            key = _fitted_identity(system)
            if any(_fitted_identity(self._archive[idx]) == key for idx in ties):
                return False
        keep = ~(np.all(values <= rows, axis=1) & np.any(values < rows, axis=1))
        self._archive = [member for member, retain in zip(self._archive, keep) if retain]
        self._archive.append(deepcopy(system))
        self._record_archive_values = np.vstack((rows[keep], values))
        return True

    def _archive_insert_scalar(self, system, label) -> bool:
        twins = [member for member in self._archive if member.equations_labels == label]
        if twins and not _values_dominate(system, twins[0]):
            return False
        others = [member for member in self._archive if member.equations_labels != label]
        if any(check_dominance(member, system) for member in others):
            return False
        self._archive = [member for member in others if not check_dominance(system, member)] + [system]
        return True

    def _archive_rows(self, view):
        """The archive's ``DominanceRows``, built again when ``self._archive``
        is not the list they were built for, when the id tables have grown well
        past the archive (evicted members' labels), or when they no longer fit;
        None when a view does not fit them. With loop stats on, one member's row
        is checked against its current view per insert."""
        rows = self._rows
        if (rows is None or self._rows_of is not self._archive or len(rows) != len(self._archive)
                or rows.interned() > 4 * len(rows) + 1024 or not rows.fits(view)):
            rows = DominanceRows(view[0], len(view[1]))
            for member in self._archive:
                member_view = _dominance_view(member)
                if not rows.fits(member_view):
                    self._rows = None
                    return None
                rows.append(member_view, member.equations_labels)
            if not rows.fits(view):
                self._rows = None
                return None
            self._rows, self._rows_of = rows, self._archive
        elif _loop_stats.enabled() and len(rows):
            r = self._inserts % len(rows)
            member = self._archive[r]
            if not rows.holds(r, _dominance_view(member), member.equations_labels):
                _loop_stats.record('NSGA.archive_rows.stale', 1, 1)
        return rows

    def reported_front(self) -> list:
        """The archive: the evaluated systems no archived system dominates
        (``EpdeSearch``'s 'auto' epochs watch it)."""
        return list(self._archive)

    # --------------------------------------------------------------- truncation
    def _protected_first(self, population, levels) -> list:
        """``evolution.niching='per_equation'``: the survivors when the systems
        holding a structure on some equation's own front come first
        (``equation_front_holders``, one holder per structure, the copy on the
        shallowest system level). When they alone exceed the population size,
        those holding more fronts are kept, then the less crowded in their own
        planes; otherwise the rest is filled by level and crowding as usual."""
        from epde.operators.multiobjective.moeadd_specific import (
            _crowding_distances, equation_front_holders, equation_planes)
        level_of = {id(member): k for k, level in enumerate(levels) for member in level}
        holders = equation_front_holders(population, lambda i: level_of.get(id(population[i]), 0))
        counts = [sum(i in held for held in holders) for i in range(len(population))]
        protected = [i for i, count in enumerate(counts) if count]
        if len(protected) >= self._mu:
            _loop_stats.record('NSGA.per_equation.protected_overflow', 1, 1)
            planes, _ = equation_planes(population)
            crowding = {}
            for e, held in enumerate(holders):
                members = sorted(held)
                for i, distance in zip(members, _crowding_distances(planes[e][members])):
                    crowding[i] = min(crowding.get(i, np.inf), distance)
            order = sorted(protected, key=lambda i: (-counts[i], -crowding.get(i, np.inf), i))
            keep = set(order[:self._mu])
            return [population[i] for i in range(len(population)) if i in keep]
        survivors = [population[i] for i in protected]
        kept = {id(member) for member in survivors}
        for level in levels:
            room = self._mu - len(survivors)
            if room <= 0:
                break
            rest = [k for k, member in enumerate(level) if id(member) not in kept]
            if not rest:
                continue
            if len(rest) > room:
                crowding = plane_crowding(level)
                rest = sorted(sorted(rest, key=lambda k: -crowding[k])[:room])
            survivors.extend(level[k] for k in rest)
        return survivors

    def _truncate(self) -> None:
        from epde.operators.multiobjective.moeadd_specific import niching_per_equation
        from epde.optimizers.nsga.per_variable import niching_per_variable, truncate
        levels_obj = self.pareto_levels
        if niching_per_variable(levels_obj.population):
            truncate(self)
            return
        levels = cycle_safe_nds(levels_obj.population)
        if niching_per_equation(levels_obj.population):
            survivors = self._protected_first(levels_obj.population, levels)
            levels_obj.population = survivors
            levels_obj.levels = cycle_safe_nds(survivors) if survivors else [[]]
            return
        survivors, kept = [], []
        for level in levels:
            if not level:
                continue
            room = self._mu - len(survivors)
            if room <= 0:
                break
            if len(level) <= room:
                chosen = list(level)
            else:
                crowding = plane_crowding(level)
                # Largest crowding first; ties keep their order.
                order = sorted(range(len(level)), key=lambda i: -crowding[i])
                chosen = [level[i] for i in sorted(order[:room])]
            survivors.extend(chosen)
            kept.append(chosen)
        levels_obj.population = survivors
        levels_obj.levels = kept if kept else [[]]

    def _report(self) -> None:
        """After the epochs: the population is the archive plus the last
        generation, sorted, so the first level is the archive."""
        levels_obj = self.pareto_levels
        from epde.interface.search_config import active_config
        if active_config().objectives.multisample_discrepancy == 'per_record':
            from epde.operators.common.objectives import LOSS_NAN_VAL
            # The archive holds all nondominated feasible fits. Only final
            # population fits missing from that first level need sorting.
            seen = {(tuple(member.obj_fun), _fitted_identity(member)) for member in self._archive}
            rest = []
            for member in levels_obj.population:
                values = member.obj_fun
                if not np.isfinite(values).all() or np.any(values >= LOSS_NAN_VAL):
                    continue
                key = (tuple(values), _fitted_identity(member))
                if key not in seen:
                    seen.add(key)
                    rest.append(member)
            levels_obj.population = self._archive + rest
            levels_obj.levels = [list(self._archive)] + (cycle_safe_nds(rest) if rest else [])
            return
        archived = {id(member) for member in self._archive}
        labels = {member.equations_labels for member in self._archive}
        # A candidate is listed once: a copy of an archived one (the duplicates
        # filling a small pool's initial population) stays out.
        union = list(self._archive) + [member for member in levels_obj.population
                                       if id(member) not in archived
                                       and member.equations_labels not in labels]
        levels_obj.population = union
        levels_obj.levels = cycle_safe_nds(union) if union else [[]]

    def _snapshot(self) -> list:
        snapshot = []
        for sol in self._archive:
            try:
                obj = sol.obj_fun.tolist() if hasattr(sol.obj_fun, 'tolist') else list(sol.obj_fun)
            except Exception:
                obj = None
            snapshot.append({'text_form': sol.text_form, 'obj_fun': obj})
        return snapshot

    # ----------------------------------------------------------------- optimize
    def optimize(self, epochs, early_stopping_callback=None, *, start_epoch=0, epoch_observer=None):
        from epde.interface.search_config import active_config
        from epde.operators.common.plain_fit import evaluated_system_observer
        token = None
        if active_config().objectives.multisample_discrepancy == 'per_record':
            token = evaluated_system_observer.set(self._archive_insert)
        try:
            return self._optimize(epochs, early_stopping_callback, start_epoch=start_epoch, epoch_observer=epoch_observer)
        finally:
            if token is not None:
                evaluated_system_observer.reset(token)

    def _optimize(self, epochs, early_stopping_callback=None, *, start_epoch=0, epoch_observer=None):
        """Run ``epochs`` NSGA-II epochs (see the module docstring). The
        callback gets ``(snapshot, epoch_idx)`` after every epoch, the snapshot
        listing the archive, and a truthy return stops the run."""
        assert self.best_obj is not None
        levels_obj = self.pareto_levels
        if len(levels_obj.population) == 0:
            linked = self.sector_processer.linked_blocks
            init_block = linked.blocks_labeled['initial_sorter']
            linked.reset_traversal_cond()
            linked.initial[0][1].set_output(levels_obj)
            init_block.apply(self.form_processer_args(0))
            self._sync_weights()
        from epde.interface.search_config import active_config
        if getattr(levels_obj, '_init_collapsed', False):
            if active_config().objectives.multisample_discrepancy == 'per_record':
                for member in levels_obj.population:
                    self._archive_insert(member)
                self._report()
            else:
                self._archive = list(levels_obj.levels[0])
            if not self._pareto_history:
                self._pareto_history.append(self._snapshot())
            if epoch_observer:
                epoch_observer(0, True)
            return
        if self._mu is None:
            self._mu = len(levels_obj.population)
            config = active_config()
            if (config.objectives.multisample_discrepancy == 'per_record'
                    and isinstance(config.evolution.population_size, int)):
                # A fit can produce more siblings than there are initialization
                # slots. Archive them all, then apply the requested survivor cap.
                self._mu = min(self._mu, config.evolution.population_size)
        if not start_epoch:
            for member in levels_obj.population:
                self._archive_insert(member)
            if len(levels_obj.population) > self._mu:
                self._truncate()
        fraction = active_config().evolution.operators.get('MOEADDSelection', {}).get('parents_fraction', 0.2)
        generations = moeadd_parents_per_sector(self._mu, fraction)

        for epoch_idx in range(start_epoch, epochs):
            with _loop_stats.timer('NSGA.epoch'):
                if global_var.verbose.show_iter_idx:
                    print(f'\n----- NSGA-II : {epoch_idx + 1}-th epoch ({generations} generations) -----')
                for _ in range(generations):
                    # The incumbents stay referenced until the inserts are done: one the
                    # generation drops (it need not be archived) would otherwise free its
                    # id(), a new member could take it over and never be archived.
                    incumbents = list(levels_obj.population)
                    before = {id(member) for member in incumbents}
                    self.sector_processer.run(population_subset=levels_obj,
                                              EA_kwargs=self.form_processer_args(0))
                    for member in levels_obj.population:
                        if id(member) not in before:
                            self._archive_insert(member)
                    del incumbents
                    self._truncate()
            self._hist.append(levels_obj.get_stats())
            snapshot = self._snapshot()
            self._pareto_history.append(snapshot)
            should_stop = False
            if early_stopping_callback is not None:
                try:
                    should_stop = bool(early_stopping_callback(snapshot, int(epoch_idx)))
                except Exception as exc:
                    warnings.warn(f'early_stopping_callback raised {exc!r}; the run goes on.',
                                  global_var.EPDEUsageWarning)
                    should_stop = False

            if epoch_observer is not None:
                epoch_observer(epoch_idx + 1, should_stop)
            if should_stop:
                if global_var.verbose.show_iter_idx:
                    print(f'Early stopping at epoch {int(epoch_idx) + 1}/{int(epochs)} '
                          '(callback returned True).')
                break
        self._report()
