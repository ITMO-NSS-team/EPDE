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
                ('cos(t)sin(x)',): 1.,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -6.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('u', 'cos(t)sin(x)'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'cos(t)sin(x)'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'cos(t)sin(x)'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'cos(t)sin(x)'): 0.,
                ('d^3u/dx2^3', 'cos(t)sin(x)'): 0.,
                }

coefficients_direct1 = {('u',): 0.,
                ('du/dx1',): -0.16666666666667,
                ('du/dx2',): 0.,
                ('d^2u/dx2^2',): 0.,
                ('d^3u/dx2^3',): -0.16666666666667,
                ('cos(t)sin(x)',): 0.16666666666667,
                ('u', 'du/dx1'): 0.,
                ('u', 'du/dx2'): -1.,
                ('u', 'd^2u/dx2^2'): 0.,
                ('u', 'd^3u/dx2^3'): 0.,
                ('u', 'cos(t)sin(x)'): 0.,
                ('du/dx1', 'du/dx2'): 0.,
                ('du/dx1', 'd^2u/dx2^2'): 0.,
                ('du/dx1', 'd^3u/dx2^3'): 0.,
                ('du/dx1', 'cos(t)sin(x)'): 0.,
                ('du/dx2', 'd^2u/dx2^2'): 0.,
                ('du/dx2', 'd^3u/dx2^3'): 0.,
                ('du/dx2', 'cos(t)sin(x)'): 0.,
                ('d^2u/dx2^2', 'd^3u/dx2^3'): 0.,
                ('d^2u/dx2^2', 'cos(t)sin(x)'): 0.,
                ('d^3u/dx2^3', 'cos(t)sin(x)'): 0.,
                }
coefficients_prob = {Symbol('u'): 2.,
                     Symbol('du/dx1'): 9,
                     Symbol('du/dx2'): 2.,
                     Symbol('d^2u/dx2^2'): 2.,
                     Symbol('d^3u/dx2^3'): 8,
                     Symbol('cos(t)sin(x)'): 8,
                     Mul(Symbol('u'), Symbol('du/dx1')): 2.,
                     Mul(Symbol('u'), Symbol('du/dx2')): 10.,
                     Mul(Symbol('u'), Symbol('d^2u/dx2^2')): 2.,
                     Mul(Symbol('u'), Symbol('d^3u/dx2^3')): 2.,
                     Mul(Symbol('u'), Symbol('cos(t)sin(x)')): 3.,
                     Mul(Symbol('du/dx1'), Symbol('du/dx2')): 2.,
                     Mul(Symbol('du/dx1'), Symbol('d^2u/dx2^2')): 3.,
                     Mul(Symbol('du/dx1'), Symbol('d^3u/dx2^3')): 2.,
                     Mul(Symbol('du/dx1'), Symbol('cos(t)sin(x)')): 3.,
                     Mul(Symbol('du/dx2'), Symbol('d^2u/dx2^2')): 2.,
                     Mul(Symbol('du/dx2'), Symbol('d^3u/dx2^3')): 3,
                     Mul(Symbol('du/dx2'), Symbol('cos(t)sin(x)')): 2.,
                     Mul(Symbol('d^2u/dx2^2'), Symbol('d^3u/dx2^3')): 2.,
                     Mul(Symbol('d^2u/dx2^2'), Symbol('cos(t)sin(x)')): 4.,
                     Mul(Symbol('d^3u/dx2^3'), Symbol('cos(t)sin(x)')): 3.,
                     }
term_ls = list(coefficients_direct1.keys())
values = list(coefficients_direct1.values())
hashed_ls = [hash_term(term) for term in term_ls]

values2 = list(coefficients_direct2.values())

coefficients1 = dict(zip(hashed_ls, values))
coefficients1[1] = 0.
coefficients2 = dict(zip(hashed_ls, values2))
coefficients2[1] = 0.
