from email.policy import default
from typing import Union, Optional
import json


def read_config(name: str) -> json:
    """
    Read some config

    Args:
        name: config name.

    Returns:
        json config.

    """
    with open(name, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

DEFAULT_CONFIG = """
{
"Optimizer": {
"learning_rate":1e-4,
"lambda_bound":10,
"optimizer":"Adam"
},
"Cache":{
"use_cache":true,
"cache_dir":"../cache/",
"cache_verbose":false,
"save_always":false,
"model_randomize_parameter":0
},
"NN":{
"batch_size":null,
"lp_par":null,
"grid_point_subset":["central"],
"h":0.001
},
"Verbose":{
	"verbose":true,
	"print_every":null
},
"StopCriterion":{
"eps":1e-5,
"tmin":1000,
"tmax":1e5 ,
"patience":5,
"loss_oscillation_window":100,
"no_improvement_patience":1000   	
},
"Matrix":{
"lp_par":null,
"cache_model":null
}
}
"""

default_config = json.loads(DEFAULT_CONFIG)


def check_module_name(module_name: str) -> bool:
    """
    Check correctness of the 'first' level of config parameter name
    we call it module.

    Args:
        module_name: first level of a parameter of a custom config.

    Returns:
        if module presents in 'default' config.
    """
    if module_name in default_config.keys():
        return True
    else:
        return False


def check_param_name(module_name: str, param_name: str) -> bool:
    """
    Check correctness of the 'first' level of config parameter name
    we call it module.

    Args:
        module_name: first level of a parameter of a custom config.
        param_name: specific parameter name.

    Returns:
       true if module presents in 'default' config.
    """
    if param_name in default_config[module_name].keys():
        return True
    else:
        return False

class Config:
    def __init__(self, *args):
        """
        We initialize config with default one

        If there is passed path to a custom config, we try to load it and change
        default parameters

        Args:
            config_path: path to a custom config

        Returns:
            config used in solver.optimization_solver function

        """

        self.params = default_config
        if len(args) == 1:
            try:
                custom_config = read_config(args[0])
            except Exception:
                print('Error reading config. Default config assumed.')
                custom_config = default_config
            for module_name in custom_config.keys():
                if check_module_name(module_name):
                    for param in custom_config[module_name].keys():
                        if check_param_name(module_name, param):
                            self.params[module_name][param] = custom_config[module_name][param]
                        else:
                            print('Wrong parameter name: ok.wrong for {}.{}. Defalut parameters assumed.'.format(
                                module_name, param))
                else:
                    print(
                        'Wrong module name: wrong.maybeok for {}.smth. Defalut parameters assumed.'.format(module_name))

        elif len(args) > 1:
            print('Too much initialization args, using default config')

    def set_parameter(self, parameter_string: str, value: Union[bool, float, int, None]):
        """
        We may want to just change default config parameters manually, without loading
        the .json

        We run checks to see we set them correctly

        Args:
            parameter_string: string in format 'module.parameter'.
            value: value for the parameter.

        """

        module_name, param = parameter_string.split('.')
        if check_module_name(module_name):
            if check_param_name(module_name, param):
                self.params[module_name][param] = value
            else:
                print(
                    'Wrong parameter name: ok.wrong for {}.{}. Defalut parameters assumed.'.format(module_name, param))
        else:
            print('Wrong module name: wrong.maybeok for {}.smth. Defalut parameters assumed.'.format(module_name))
