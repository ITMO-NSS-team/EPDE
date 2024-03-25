import torch
from typing import Any
from epde.solver.device import device_type

class Closure():
    def __init__(self,
        mixed_precision: bool,
        model):

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
        self._model = model

    @property
    def model(self):
        return self._model

    def _amp_mixed(self):
        """ Preparation for mixed precsion operations.

        Args:
            mixed_precision (bool): use or not torch.amp.

        Raises:
            NotImplementedError: AMP and the LBFGS optimizer are not compatible.

        Returns:
            scaler: GradScaler for CUDA.
            cuda_flag (bool): True, if CUDA is activated and mixed_precision=True.
            dtype (dtype): operations dtype.
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            print(f'Mixed precision enabled. The device is {self.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")
        

    def _closure(self):
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

    def _closure_pso(self):
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

    def get_closure(self, _type: str):
        if _type == 'PSO':
            return self._closure_pso
        else:
            return self._closure
