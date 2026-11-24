"""Access the training rows of the current fit from search operators."""
import epde.globals as global_var


def current_rows():
    manager = getattr(global_var, 'samples_manager', None)
    return getattr(manager, 'evaluation_rows', None)


def active():
    return current_rows() is not None


def slot_of(equation):
    """Pointwise strong evaluation shares rows across equation slots."""
    return equation.main_var_to_explain if active() else None


def term_columns(term, slot):
    return term.evaluate() if slot is None else term.evaluate(slot=slot)
