import re


def get_terms_ls(eq: str):
    terms = eq.split("+")
    coefs = []
    names = []
    for term in terms:
        coefs.append(float(re.findall(r"-?\d+\.\d*", term)[0]))
        names.append(re.findall(r"x\S*", term)[0])
    return coefs, names


def calc_difference(eq: str, true_coef: list, true_names: list):
    coeffs, names = get_terms_ls(eq)
    mae = 0
    term_found = 0
    for i in range(len(names)):
        if names[i] in true_names:
            term_found += 1
            idx = true_names.index(names[i])
            mae += abs(coeffs[i] - true_coef[idx])
        else:
            mae += abs(coeffs[i])
    mae /= (len(names) + 1)
    if term_found == len(true_names):
        return mae, 1
    return None, 0
