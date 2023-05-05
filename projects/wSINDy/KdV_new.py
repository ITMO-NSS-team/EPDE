import numpy as np
# from scipy.io import loadmat

import pandas as pd
import pysindy as ps

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import re


def sindy_out_format(model, max_deriv_order):
    def pure_derivs_format(string, max_deriv_order, deriv_symbol='\d'):
        # производная переменной x2 или x1
        if deriv_symbol == '\d':
            deriv_ind = 'x2'
        else:
            deriv_ind = 'x1'

        for i in range(max_deriv_order, 0, -1):

            derivs_str = deriv_symbol
            regex1 = r'x0_' + derivs_str * i + '[^1,t]'
            match1 = re.search(regex1, string)
            regex2 = r'x0_' + derivs_str * i + '$'
            match2 = re.search(regex2, string)

            # конец строки вынесен в отдельный паттерн regex2
            if match2 != None:
                start, end = match2.regs[0][0], match2.regs[0][1]
                if i != 1:
                    string = string[:start] + 'd^' + str(i) + 'u/d' + deriv_ind + '^' + str(i) + '{power: 1}' \
                             + string[end:]
                else:
                    string = string[:start] + 'du/d' + deriv_ind + '{power: 1}' + string[end:]

            if match1 == None:
                continue
            else:
                start, end = match1.regs[0][0], match1.regs[0][1]
                if i == 1:
                    string = string[:start] + 'du/d' + deriv_ind + '{power: 1} ' + string[end:]
                else:
                    string = string[:start] + 'd^' + str(i) + 'u/d' + deriv_ind + '^' + str(i) + '{power: 1} ' \
                             + string[end:]
        return string

    def mix_derivs_format(string, match):
        if match != None:
            start, end = match.regs[0][0], match.regs[0][1]

            # считаем число производных по x2 и x1 в найденном кусочке
            number_1 = string.count('1', start, end)
            number_t = i - number_1

            if number_1 == 1:
                string_x_der = ' du/dx2{power: 1}'
            else:
                string_x_der = ' d^' + str(number_1) + 'u/dx2^' + str(number_1) + '{power: 1}'

            if number_t == 1:
                string_t_der = '* du/dx1{power: 1} *'
            else:
                string_t_der = '* d^' + str(number_t) + 'u/dx1^' + str(number_t) + '{power: 1} *'

            string = string[:start] + string_t_der + string_x_der + string[end:]
        return string

    string_in = model.equations()[0]

    # заменяем нулевую степень
    string = string_in.replace(" 1 ", " ")

    # заменим все чистые производные
    string = pure_derivs_format(string, max_deriv_order)
    string = pure_derivs_format(string, max_deriv_order, 't')

    # заменим смешанные производные
    for i in range(max_deriv_order, 1, -1):
        derivs_str = '[1,t]'
        regex2 = r'x0_' + derivs_str * i + '$'
        match2 = re.search(regex2, string)
        string = mix_derivs_format(string, match2)

        regex1 = r'x0_' + derivs_str * i
        match1 = re.search(regex1, string)
        string = mix_derivs_format(string, match1)

    # проставим пробелы и * после числовых коэфф-в
    while (True):
        regex = r'\d [a-zA-Z]'  # + '[^1]'
        match = re.search(regex, string)
        if match == None:
            break
        else:
            start = match.regs[0][0]
            string = string[:start + 1] + ' *' + string[start + 1:]

    # заменим x0 на u{power: 1} (нулевая степень производной)
    for j in range(10, 0, -1):
        asterixes = j - 1
        insert_string = 'u{power: 1} ' + '* u{power: 1} ' * asterixes
        string = string.replace('x0' * j, insert_string)

    # проставим недостающие * и пробелы
    while (True):
        regex = r'power: 1} [a-zA-Z]'
        match = re.search(regex, string)
        if match == None:
            break
        else:
            start = match.regs[0][0]
            string = string[:start + 9] + ' *' + string[start + 9:]

    string_out = 'du/dx1 = ' + string
    return string_out




if __name__ == "__main__":
    np.random.seed(100)
    
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['method'] = 'LSODA'
    integrator_keywords['atol'] = 1e-12
    
    
    
    # kdV = loadmat('kdv.mat')
    # t = np.ravel(kdV['t'])
    # x = np.ravel(kdV['x'])
    # u = np.real(kdV['usol'])
    
    df = pd.read_csv('projects/wSINDy/data/KdV/KdV_sln_100.csv', header=None)
    u = df.values
    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])
    
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    
    u = u.reshape(len(x), len(t), 1)
    
    # задаем свои токены через лямбда выражения
    library_functions = [lambda x: np.cos(x)*np.cos(x),]# lambda x: x, lambda x: x * x]#, lambda x: 1/x]
    library_function_names = [lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')',]# lambda x: x, lambda x: x + x,]#, lambda x: '1/'+x]
    
    # ? проблема с использованием multiindices
    # multiindices=np.array([[0,1],[1,1],[2,0],[3,0]])
    
    pde_lib = ps.PDELibrary(library_functions=library_functions,
                            function_names=library_function_names,
                            derivative_order=3, spatial_grid=x,
                            # multiindices=multiindices,
                            implicit_terms=False,# temporal_grid=t,
                            include_bias=True, is_uniform=True)#, include_interaction=True)
    # feature_library = ps.feature_library.PolynomialLibrary(degree=3)
    # optimizer = ps.SR3(threshold=7, max_iter=10000, tol=1e-15, nu=1e2,
    #                    thresholder='l0', normalize_columns=True)
    # model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    # model.fit(u, t=dt)
    # model.print()
    
    # второй оптимизатор
    optimizer = ps.STLSQ(threshold=5, alpha=1e-3, normalize_columns=True)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    model.print()
    
    string = model.equations() # вернет правую часть уравнения в виде списка строки
    # print(string)
    
    max_deriv_order = pde_lib.derivative_order # вытаскиваем параметр derivative_order, чтобы передать дальше
    string1 = sindy_out_format(model, max_deriv_order) # форматированная строка
    print(string1)