from email.policy import default
from typing import Union, Optional
import json


def read_config(name: str) -> json:
    """
    Reads a configuration file containing settings for the equation discovery process.
    
    This function is essential for setting up and customizing the search for differential equations.
    It allows users to define parameters such as evolutionary algorithm settings, data preprocessing 
    options, and custom token definitions, ensuring that the equation discovery process is tailored 
    to the specific problem and dataset.
    
    Args:
        name (str): The path to the JSON configuration file.
    
    Returns:
        json: A dictionary containing the configuration parameters loaded from the file.
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
    Validates the module name against a predefined configuration.
    
    This function ensures that the provided module name is a valid top-level
    configuration element, which is crucial for correct configuration processing.
    By verifying the module name, the system can properly route and apply
    configuration settings, preventing errors and ensuring consistent behavior.
    
    Args:
        module_name: The module name to validate (first level config parameter).
    
    Returns:
        True if the module name is found in the default configuration, False otherwise.
    """
    if module_name in default_config.keys():
        return True
    else:
        return False


def check_param_name(module_name: str, param_name: str) -> bool:
    """
    Checks if a given parameter name exists within a specified module in the default configuration.
    
    This function verifies that a parameter is valid for a particular module, ensuring that only recognized parameters are used during the equation discovery process.
    
    Args:
        module_name (str): The name of the module to check within the default configuration.
        param_name (str): The name of the parameter to validate.
    
    Returns:
        bool: True if the parameter name is found within the specified module in the default configuration, False otherwise.
    """
    if param_name in default_config[module_name].keys():
        return True
    else:
        return False

class Config:
    """
    Represents a configuration object for the solver.
    
        The configuration can be initialized with default values and updated
        from a custom configuration file.
    
        Attributes:
            config_path: Path to a custom configuration file.
    """

    def __init__(self, *args):
        """
        Initializes the configuration, prioritizing user-defined settings.
        
        The configuration is initialized with default parameters. If a path to a
        custom configuration file is provided, the method attempts to load it and
        override the default settings. This allows users to tailor the equation
        discovery process to their specific problem and dataset. The method validates
        the structure of the custom configuration to ensure compatibility.
        
        Args:
            config_path (str, optional): Path to a custom configuration file. If None, the default configuration is used.
        
        Returns:
            dict: The configuration dictionary used by the equation discovery process.
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
        Modifies a specific configuration parameter directly.
        
                This allows for fine-grained control over the configuration without needing to load a complete configuration file.
                Input is validated to ensure that only existing modules and parameters are modified, preventing unintended configuration errors.
        
                Args:
                    parameter_string: A string specifying the parameter to modify, in the format 'module.parameter'.
                    value: The new value for the specified parameter.  Can be a boolean, float, integer, or None.
        
                Returns:
                    None. The method modifies the configuration in place.
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
