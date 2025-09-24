import torch
from typing import Any
from epde.solver.device import device_type

class Closure():
    """
    A class that manages different closure types for optimization.
    
        Class Methods:
        - __init__:
    """

    def __init__(self,
        mixed_precision: bool,
        model):
        """
        Initializes the Trainer, preparing it for equation discovery.
        
                This setup configures the training environment based on the provided model and mixed precision settings.
                It determines the appropriate device (CPU or CUDA), data type, and initializes the optimizer.
                The trainer is initialized with the necessary components to search for the best equation structure.
        
                Args:
                    mixed_precision (bool): Whether to use mixed precision training for faster computations.
                    model: The model to train, containing the equation structure and parameters.
        
                Returns:
                    None
        
                Fields:
                    mixed_precision (bool): Whether mixed precision training is enabled.
                    model (torch.nn.Module): The model representing the equation to be discovered.
                    optimizer (torch.optim.Optimizer): The optimizer used for training the model parameters.
                    normalized_loss_stop (float): The normalized loss value at which training should stop.
                    device (str): The device ('cuda' or 'cpu') used for training.
                    cuda_flag (bool): A flag indicating whether CUDA is used.
                    dtype (torch.dtype): The data type (torch.float16 or torch.bfloat16) used for training.
        """

        self.mixed_precision = mixed_precision
        self.set_model(model)
        self.optimizer = self.model.optimizer
        self.normalized_loss_stop = self.model.normalized_loss_stop
        self.device = device_type()
        self.cuda_flag = True if self.device == 'cuda' and self.mixed_precision else False
        self.dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16
        if self.mixed_precision:
            self._amp_mixed()


    def set_model(self, model):
        """
        Sets the internal equation model.
        
        This model represents the discovered differential equation. Setting it allows the system to evaluate and utilize the identified equation for further analysis or simulation.
        
        Args:
            model: The equation model to be set. This model encapsulates the structure and parameters of the discovered differential equation.
        
        Returns:
            None.
        """
        self._model = model

    @property
    def model(self):
        """
        Gets the symbolic representation of the discovered equation.
        
        This representation allows for further analysis, manipulation,
        and integration with other symbolic computation tools.
        
        Args:
            None
        
        Returns:
            str: The symbolic representation of the discovered equation.
        """
        return self._model

    def _amp_mixed(self):
        """
        Configures mixed precision training using torch.cuda.amp.
        
        This setup is crucial for leveraging mixed precision, which can significantly accelerate training on CUDA-enabled devices.
        It initializes a GradScaler if mixed precision is enabled, which is essential for preventing underflow issues during backpropagation with lower precision data types.
        It also performs a check to ensure compatibility with the LBFGS optimizer, as it is known to be incompatible with mixed precision training.
        
        Args:
            mixed_precision (bool): A flag indicating whether to use mixed precision training.
        
        Raises:
            NotImplementedError: If the LBFGS optimizer is used in conjunction with mixed precision.
        
        Returns:
            scaler (torch.cuda.amp.GradScaler): A GradScaler instance for managing gradients during mixed precision training.
            cuda_flag (bool): True if CUDA is active and mixed_precision is enabled, False otherwise.
            dtype (torch.dtype): The data type to be used for operations (torch.float16 or torch.float32).
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            print(f'Mixed precision enabled. The device is {self.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")
        

    def _closure(self):
        """
        Performs a closure step for optimization. This step is crucial for refining the equation discovery process by iteratively adjusting model parameters based on the loss.
        
                This method performs the core optimization step, including zeroing gradients,
                evaluating the loss, performing backpropagation, and updating the optimizer.
                It also handles mixed precision training if enabled. The loss is calculated and backpropagated to refine the equation's coefficients, guiding the search towards a better fit with the observed data.
        
                Args:
                    self: The object instance.
        
                Returns:
                    torch.Tensor: The computed loss value, representing the discrepancy between the model's predictions and the data.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        if self.cuda_flag:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        return loss
    
    def _closure_nncg(self):
        """
        Compute the loss and gradients to refine the equation discovery process.
        
                This method evaluates the loss based on how well the current equation
                fits the data, computes the gradients of the loss with respect to the
                equation's parameters, and updates the model's current loss. It leverages
                mixed precision training when enabled for faster and more efficient
                computation. This is a crucial step in guiding the search for the
                optimal equation structure by quantifying the error and providing
                direction for improvement.
        
                Args:
                    self: The object instance.
        
                Returns:
                    tuple: A tuple containing the loss and the gradients.
                        - loss: The computed loss, representing the error between the
                          equation's predictions and the observed data.
                        - grads: The computed gradients, indicating the direction to
                          adjust the equation's parameters to reduce the loss.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        
        # if self.optimizer.use_grad:
        grads = self.optimizer.gradient(loss)
        grads = torch.where(grads != grads, torch.zeros_like(grads), grads)
        # else:
        #     grads = torch.tensor([0.])
        # if self.cuda_flag:
        #     self.scaler.scale(loss).backward()
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        # else:
        #     loss.backward()
        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss
        return loss, grads

    def _closure_pso(self):
        """
        Performs a Particle Swarm Optimization (PSO) iteration to refine equation candidates.
        
                This method evaluates the loss and gradients for each particle (equation candidate) in the swarm,
                updates the swarm's losses and gradients based on how well each candidate fits the data,
                and sets the current loss of the model to the best performing candidate's loss. This process
                iteratively improves the swarm's ability to represent the underlying differential equation.
        
                Args:
                    self: The instance of the class containing the PSO optimizer and model.
        
                Returns:
                    tuple: A tuple containing:
                        - losses (torch.Tensor): A tensor of losses for each particle in the swarm,
                          representing the error between the equation candidate's predictions and the data.
                        - gradients (torch.Tensor): A tensor of gradients for each particle in the swarm,
                          indicating the direction to adjust the equation candidate to reduce the error.
        """
        def loss_grads():
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device,
                                dtype=self.dtype,
                                enabled=self.mixed_precision):
                loss, loss_normalized = self.model.solution_cls.evaluate()

            if self.optimizer.use_grad:
                grads = self.optimizer.gradient(loss)
                grads = torch.where(grads == float('nan'), torch.zeros_like(grads), grads)
            else:
                grads = torch.tensor([0.])

            return loss, grads

        loss_swarm = []
        grads_swarm = []
        for particle in self.optimizer.swarm:
            self.optimizer.vec_to_params(particle)
            loss_particle, grads = loss_grads()
            loss_swarm.append(loss_particle)
            grads_swarm.append(grads.reshape(1, -1))

        losses = torch.stack(loss_swarm).reshape(-1)
        gradients = torch.vstack(grads_swarm)

        self.model.cur_loss = min(loss_swarm)

        return losses, gradients
    
    def _closure_ngd(self):
        """
        Performs a single optimization step using the NGD optimizer to refine the solution.
        
                This method computes the loss, performs backpropagation to adjust model parameters, and updates these parameters using the optimizer.
                It also calculates interior residuals and boundary values to assess how well the current solution satisfies the differential equation and boundary conditions.
                This is crucial for guiding the search towards a solution that accurately represents the underlying physical system described by the data.
        
                Args:
                    self: The object instance.
        
                Returns:
                    tuple: A tuple containing:
                        - int_res: Interior residuals computed by the PDE operator, indicating the equation's satisfaction within the domain.
                        - bval: Boundary values obtained after applying boundary conditions, representing the solution's behavior at the domain's edges.
                        - true_bval: True boundary values, the target values for the boundary conditions.
                        - loss: The computed loss value, quantifying the discrepancy between the model's predictions and the observed data and boundary conditions.
                        - self.model.solution_cls.evaluate: The evaluation function of the solution class, used for subsequent evaluations.
        """
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        if self.cuda_flag:
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward(retain_graph=True)

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        int_res = self.model.solution_cls.operator._pde_compute()
        bval, true_bval, _, _ = self.model.solution_cls.boundary.apply_bcs()

        return int_res, bval, true_bval, loss, self.model.solution_cls.evaluate

    def get_closure(self, _type: str):
        """
        Retrieves a specific optimization closure based on the provided type.
        
        This method allows to switch between different optimization strategies 
        during the equation discovery process. Different closures encapsulate different
        optimization algorithms, enabling the framework to explore various search
        directions in the equation space.
        
        Args:
            _type (str): The type of closure to retrieve ('PSO', 'NGD', 'NNCG', or default).
        
        Returns:
            callable: The requested closure function. Returns a default closure if the type is not recognized.
        """
        if _type == 'PSO':
            return self._closure_pso
        elif _type == 'NGD':
            return self._closure_ngd
        elif _type == 'NNCG':
            return self._closure_nncg        
        else:
            return self._closure
