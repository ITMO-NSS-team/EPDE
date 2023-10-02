# from sympy import Mul, Symbol, Pow
#
# list0 = ("u", "u", "du/dx1")
# list1 = list(map(lambda u: Symbol(u), list0))
# d1 = Mul(*list1)
# d2 = Mul(Symbol("du/dx2"), Symbol("u"))
# prob_ls = [0.3, 0.7]
# ddd = {d1, d2}
# rand = np.random.choice(a=[d1, d2], p=prob_ls)
# dict1 = {d1: 21, d2: 56}
# keys = list(dict1.keys())
# if d1 == d2:
#     print('they are!')
# tsym = {1, 15, 208, 196}
# pool_set = {7, 1, 15}
# tsym_dict = {208: 'n/a', 1: 'y', 15: 'n', 196: 'not def'}
# pool_ls = list(pool_set)
#
# pool_set_intersection_tsym = pool_set.intersection(tsym)
# coeffs = []
# for el in pool_ls:
#     coeffs.append(tsym_dict.get(el))
# print()