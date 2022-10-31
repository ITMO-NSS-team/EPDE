#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:51:34 2021

@author: mike_ubuntu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:45:14 2021

@author: mike_ubuntu
"""

import numpy as np
import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import TrigonometricTokens

if __name__ == '__main__':

    t = np.linspace(0, 4*np.pi, 1000) # setting time axis, corresonding to the solution of ODE
    u = np.load('/media/mike_ubuntu/DATA/EPDE_publication/tests/system/Test_data/fill366.npy') # loading data with the solution of ODE
    
    # Trying to create population for mulit-objective optimization with only 
    # derivatives as allowed tokens. Spoiler: only one equation structure will be 
    # discovered, thus MOO algorithm will not be launched.
    
    epde_search_obj = epde_alg.epde_search()
    
    trig_tokens = TrigonometricTokens(freq = (0.95, 1.05))
    
    epde_search_obj.fit(data = u, equation_factors_max_number = 2, coordinate_tensors = [t,], 
                        additional_tokens = trig_tokens, field_smooth = False)
    
    epde_search_obj.equation_search_results(only_print = True, level_num = 1) # showing the Pareto-optimal set of discovered equations 