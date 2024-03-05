from sympy import Symbol, Mul


def hash_term(term):
    total_term = 0
    for token in term:
        total_token = 1
        if type(token) == tuple:
            token = token[0]
        for char in token:
            total_token += ord(char)
        total_term += total_token * total_token
    return total_term


coefficients_direct2 = {('u',): 0.,
                ('du/dx1',): -1.,
                ('du/dx2',): 0.,
                ('d^2u/dx2^2',): 0.,
                ('d^3u/dx2^3',): -1.,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -6.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                }

coefficients_direct1 = {('u',): 0.,
                ('du/dx1',): -0.16666666666667,
                ('du/dx2',): 0.,
                ('d^2u/dx2^2',): 0.,
                ('d^3u/dx2^3',): -0.16666666666667,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -1.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                }
coefficients_prob = {Symbol('u'): 3.,
                     Symbol('du/dx1'): 10,
                     Symbol('du/dx2'): 2.,
                     Symbol('d^2u/dx2^2'): 3.,
                     Symbol('d^3u/dx2^3'): 9,
                     Mul(Symbol('u'), Symbol('du/dx1')): 2.,
                     Mul(Symbol('u'), Symbol('du/dx2')): 9,
                     Mul(Symbol('u'), Symbol('d^2u/dx2^2')): 2.,
                     Mul(Symbol('u'), Symbol('d^3u/dx2^3')): 2.,
                     Mul(Symbol('du/dx1'), Symbol('du/dx2')): 3.,
                     Mul(Symbol('du/dx1'), Symbol('d^2u/dx2^2')): 4.,
                     Mul(Symbol('du/dx1'), Symbol('d^3u/dx2^3')): 2.,
                     Mul(Symbol('du/dx2'), Symbol('d^2u/dx2^2')): 2.,
                     Mul(Symbol('du/dx2'), Symbol('d^3u/dx2^3')): 3.,
                     Mul(Symbol('d^2u/dx2^2'), Symbol('d^3u/dx2^3')): 2.,
                     }
term_ls = list(coefficients_direct1.keys())
values = list(coefficients_direct1.values())
hashed_ls = [hash_term(term) for term in term_ls]

values2 = list(coefficients_direct2.values())

coefficients1 = dict(zip(hashed_ls, values))
coefficients1[1] = 0.
coefficients2 = dict(zip(hashed_ls, values2))
coefficients2[1] = 0.
