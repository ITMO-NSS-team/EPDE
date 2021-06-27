#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:56:10 2021

@author: mike_ubuntu
"""

from functools import wraps

changelog_entry_templates = {}

class Reset_equation_status():
    def __init__(self, reset_input : bool = True, reset_output : bool = False):
        self.reset_input = reset_input; self.reset_output = reset_output
        
    def __call__(self, method):
        @wraps(method)
        def wrapper(obj, *args, **kwargs):
            result = method(obj, *args, **kwargs)
#            print('method output:', method,  method(obj, *args, **kwargs))
            if self.reset_input:        
                for element in [obj,] + list(args):
                    if isinstance(element, (list, tuple, set)):
                        for subelement in element:
                            try: 
                                subelement.reset_eval()
                            except AttributeError:
                                pass
                    else:
                        try: 
                            element.reset_eval()
                        except AttributeError:
                            pass
            if self.reset_output:
#                print('From decorator Reset_equation_status:', type(result), result)
                try:
                    for equation in result:
                        try: 
                            equation.reset_eval()
                        except AttributeError:
                            pass
                except TypeError:
                    try: 
                        result.reset_eval()
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


#            if historized(obj):
#                if 'b' in self.state_writing_points:
#                    log_entry += ' || before operation: ' + obj.state
#                method(obj, *args, **kwargs)
#                if 'a' in self.state_writing_points:
#                    log_entry += ' | after operation: ' + obj.state
#                obj.add_history(log_entry + '|| \n')
#            elif historized(args[0]):
#                if 'b' in self.state_writing_points:
#                    log_entry += ' || before operation: ' + obj.state
#                args[0].add_history(self.action_log_entry)
#                if 'a' in self.state_writing_points:
#                    log_entry += ' | after operation: ' + obj.state
#                obj.add_history(log_entry + '|| \n')                    
#            else:
#                raise ValueError('Attempting to write an entry into changelog of an object without changelog')
