"""Within-parent exchange with the retained equations fixed in place."""
from functools import cache
from itertools import product

from anonymous_exchange import archive_terms, exchanges, fit_pairs, rename_factor, swapped, target_slot


def fixed_edits(parents, vocabulary, nfields, naxes, retarget_copy):
    maps = list(product(exchanges(nfields), exchanges(naxes)))

    @cache
    def variants(donor, slot):
        support, target = donor
        result = set()
        for fm, am in maps:
            changed = swapped(support, fm, am)
            if changed == support or any(f not in vocabulary for term in changed for f in term):
                continue
            inherited = tuple(sorted(rename_factor(f, fm, am) for f in target))
            pivots = changed if retarget_copy else (inherited,)
            for pivot in pivots:
                if target_slot(pivot) == slot:
                    result.add((changed, pivot))
        return result

    edits = set()
    for i, parent in enumerate(parents):
        for slot in range(nfields):
            kept = parent[:slot] + parent[slot+1:]
            kept_supports = {s for s,_ in kept}
            if len(kept_supports) != nfields-1:
                continue
            for donor in kept:
                edits.update((i, slot, support, pivot) for support, pivot in variants(donor, slot)
                             if support not in kept_supports)
    return edits


def generate_fixed(archive, data, bank, *, retarget_copy, max_new_fits):
    labels, supports, vocabulary = archive_terms(archive)
    front = archive['native_front']
    parents = [tuple((supports[i], labels[archive['points'][i][1]]) for i in row) for row in front]
    nfields, naxes = len(archive['variables']), len(data.shape)
    edits = fixed_edits(parents, vocabulary, nfields, naxes, retarget_copy)

    # A relabelling permutes the destination slot and carries the pivot with its field.
    fm = tuple((i+1) % nfields for i in range(nfields))
    am = tuple((i-1) % naxes for i in range(naxes))

    @cache
    def renamed(s, t):
        return swapped(s, fm, am), tuple(sorted(rename_factor(f, fm, am) for f in t))

    reordered = [tuple(renamed(*p[fm.index(i)]) for i in range(nfields)) for p in parents]
    vocab = {rename_factor(f, fm, am) for f in vocabulary}
    check = fixed_edits(reordered, vocab, nfields, naxes, retarget_copy)
    assert check == {(i, fm[slot], *renamed(s, t)) for i,slot,s,t in edits}
    needed = {(s, t) for _,_,s,t in edits}
    fits, scores, cost = fit_pairs(archive, data, bank, labels, supports, needed, max_new_fits)
    proposals = set()
    for i, slot, support, pivot in edits:
        if (support, pivot) in fits:
            row = list(front[i])
            row[slot] = fits[support, pivot]
            proposals.add(tuple(row))
    cost.update(parent_front=len(front), parent_slot_edits=len(edits), relabelled_edits_verified=len(check))
    return sorted(proposals), supports, scores, cost
