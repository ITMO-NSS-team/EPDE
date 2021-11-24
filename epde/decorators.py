#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:56:10 2021

@author: mike_ubuntu
"""

from functools import wraps

changelog_entry_templates = {}

class Reset_equation_status():
    def __init__(self, reset_input : bool = True, reset_output : bool = False, 
                 reset_right_part : bool = True):
        self.reset_input = reset_input; self.reset_output = reset_output
        self.reset_right_part = reset_right_part

    def __call__(self, method):
        @wraps(method)
        def wrapper(obj, *args, **kwargs):
            result = method(obj, *args, **kwargs)
            
            if self.reset_input:        
                for element in [obj,] + list(args):
                    if isinstance(element, (list, tuple, set)):
                        for subelement in element:
                            try: 
                                subelement.reset_state(self.reset_right_part)
                            except AttributeError:
                                pass
                    else:
                        try: 
                            element.reset_state(self.reset_right_part )
                        except AttributeError:
                            pass
            if self.reset_output:
                try:
                    for equation in result:
                        try: 
                            equation.reset_state(self.reset_right_part )
                        except AttributeError:
                            pass
                except TypeError:
                    try: 
                        result.reset_state(self.reset_right_part )
                    except AttributeError:
                        pass                    
            return result
        return wrapper
                

class History_Extender():
    '''
    
    Extend histroy log of the complex structure
    
    '''
    def __init__(self, action_log_entry : str = '', state_writing_points = 'n'):
        assert (state_writing_points == 'n' or state_writing_points == 'ba' or 
                state_writing_points == 'b' or state_writing_points == 'a')
        self.action_log_entry = action_log_entry
        self.state_writing_points = state_writing_points

    def __call__(self, method):
        @wraps(method)
        def wrapper(obj, *args, **kwargs):
            def historized(h_obj):
                res = hasattr(h_obj, '_history') and hasattr(h_obj, 'add_history')
#                print(f'called object of the type {type(h_obj)} is historized {res}')
                return res #hasattr(h_obj, '_history') and hasattr(h_obj, 'add_history')

            for element in [obj,] + list(args):
                if historized(element):
                    element.add_history(self.action_log_entry)
                    
            if 'b' in self.state_writing_points:
                ender = ' ' if 'a' in self.state_writing_points else ' || \n'
                for element in [obj,] + list(args):
                    if historized(element):
                        element.add_history(' || before operation: ' + element.state + ender)
                        
            result = method(obj, *args, **kwargs)
            if 'a' in self.state_writing_points:
                beginner = ' | ' if 'b' in self.state_writing_points else ' || '                
                for element in [obj,] + list(args):
                    if historized(element):
                        element.add_history(beginner + 'after operation: ' + element.state + ' || \n')
            return result
        return wrapper
    
# class Parallelize_method():
#     def __init__(self, num_cpus = 1):
#         self.num_cpus = num_cpus
        
#     def __call__(self, method):
#         @wraps(method)
#         @ray.remote(num_cpus = self.num_cpus)
#         def wrapper(*args, **kwargs):
#             return method(*args, **kwargs)
    
    
#class Ray_parallelizer():
#    def __init__(self):
#        
