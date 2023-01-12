import torch


class SolverAdapter(object):
    def __init__(self, model = None, use_cache : bool = True):
        if model is None:
            model = torch.nn.Sequential(
               torch.nn.Linear(1, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 1),
            )
        self.default_model = model


        self._solver_params = {'model' : self.default_model, 'learning_rate' : 1e-3, 'eps' : 1e-5, 'tmin' : 1000,
                               'tmax' : 1e5, 'use_cache' : True, 'cache_verbose' : True, 
                               'save_always' : False, 'print_every' : False, 
                               'model_randomize_parameter' : 1e-6, 'step_plot_print' : False, 
                               'step_plot_save' : False, 'image_save_dir' : None}

        self.set_solver_params()
        self.use_cache = use_cache
        self.prev_solution = None

    # def use_default(self, key, vals, base_vals):
    #     self._params = key


    # def set_solver_params(self, params = {'model' : None, 'learning_rate' : None, 'eps' : None, 'tmin' : None, 
    #                                       'tmax' : None, 'use_cache' : None, 'cache_verbose' : None, 
    #                                       'save_always' : None, 'print_every' : None, 
    #                                       'model_randomize_parameter' : None, 'step_plot_print' : None, 
    #                                       'step_plot_save' : None, 'image_save_dir' : None}):

    def set_solver_params(self, model = None, learning_rate : float = None, eps : float = None, 
                          tmin : int = None, tmax : int = None, use_cache : bool = None, cache_verbose : bool = None, 
                          save_always : bool = None, print_every : bool = None, 
                          model_randomize_parameter : bool = None, step_plot_print : bool = None, 
                          step_plot_save : bool = None, image_save_dir : str = None):
        params = {'model' : model, 'learning_rate' : learning_rate, 'eps' : eps, 'tmin' : tmin,
                  'tmax' : tmax, 'use_cache' : use_cache, 'cache_verbose' : cache_verbose, 
                  'save_always' : save_always, 'print_every' : print_every, 
                  'model_randomize_parameter' : model_randomize_parameter, 'step_plot_print' : step_plot_print, 
                  'step_plot_save' : step_plot_save, 'image_save_dir' : image_save_dir}
        
        if model is None:
            model = self.default_model 
        for param_key, param_vals in params.items():
            if params is not None:
                try:
                    self._solver_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')

    def set_param(self, param_key, value):
        self._solver_params[param_key] = value

    def solve_epde_system(self, system : SoEq, grid = None, boundary_conditions = None):
        system_interface = SystemSolverInterface(system_to_adapt = system)

        solver_form, grid, bconds = 

    def solve(self, system_form = None, grid = None, boundary_conditions = None):
        if system_form is None and grid is None and boundary_conditions is None:
            self.equation = SolverEquation(grid, system_form, boundary_conditions).set_strategy('NN')
        self.prev_solution = solver.Solver(grid, self.equation, self.model, 'NN').solver(**self._solver_params)
        return self.prev_solution