# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:50:12 2021

@author: user
"""
import pickle
import datetime
import torch
import os 
import glob
import numpy as np

from epde.solver.metrics import Solution
from epde.solver.input_preprocessing import Equation, EquationMixin

class Model_prepare(Solution):
    def __init__(self, grid, equal_cls, model, mode):
        super().__init__(grid, equal_cls, model, mode)
        self.equal_cls = equal_cls    
    
    @staticmethod
    def create_random_fn(eps):
        def randomize_params(m):
            if type(m)==torch.nn.Linear or type(m)==torch.nn.Conv2d:
                m.weight.data=m.weight.data+(2*torch.randn(m.weight.size())-1)*eps#Random weight initialisation
                m.bias.data=m.bias.data+(2*torch.randn(m.bias.size())-1)*eps
        return randomize_params


    def cache_lookup(self, lambda_bound=0.001, cache_dir='../cache/', nmodels=None, cache_verbose=False):
        
        files=glob.glob(cache_dir+'*.tar')
        # if files not found
        if len(files)==0:
            best_checkpoint=None
            min_loss=np.inf
            return best_checkpoint, min_loss
        # at some point we may want to reduce the number of models that are
        # checked for the best in the cache
        if nmodels==None:
            # here we take all files that are in cache
            cache_n=np.arange(len(files))
        else:
            # here we take random nmodels from the cache
            cache_n=np.random.choice(len(files), nmodels, replace=False)
        cache_same_architecture=[]
        min_loss=np.inf
        best_model=0
        best_checkpoint={}
        var = []
        n_vars = []
        for eqn in self.prepared_operator:
                for term in eqn:
                    if self.mode == 'NN':
                        var.append(term[4])
                    elif self.mode == 'autograd':
                         var.append(term[3])
        for elt in var:
            nrm = np.sqrt((np.array([-1]) - elt) ** 2)
            for elem in nrm:
                n_vars.append(elem)
        n_vars = int(max(n_vars))
        
        for i in cache_n:
            file=files[i]
            checkpoint = torch.load(file)
            model=checkpoint['model']
            model.load_state_dict(checkpoint['model_state_dict'])
            # this one for the input shape fix if needed
            # it is taken from the grid shape
            if model[0].in_features != self.grid.shape[-1]:
                continue
            try:
                if model[-1].out_features != n_vars:
                    continue
            except Exception:
                continue
            # model[0] = torch.nn.Linear(prepared_grid.shape[-1], model[0].out_features)
            # model.eval()
            l=self.loss_evaluation(lambda_bound=lambda_bound)      
            if l<min_loss:
                min_loss=l
                best_checkpoint['model']=model
                best_checkpoint['model_state_dict']=model.state_dict()
                best_checkpoint['optimizer_state_dict']=checkpoint['optimizer_state_dict']
                if cache_verbose:
                    print('best_model_num={} , loss={}'.format(i,l))
        if best_checkpoint=={}:
            best_checkpoint=None
            min_loss=np.inf
        return best_checkpoint,min_loss


    def save_model(self, prep_model, state, optimizer_state, cache_dir='../cache/', name=None):
        if name==None:
            name=str(datetime.datetime.now().timestamp())
        if os.path.isdir(cache_dir):
            torch.save({'model':prep_model, 'model_state_dict': state,
                    'optimizer_state_dict': optimizer_state}, cache_dir+name+'.tar')
        else:
            os.mkdir(cache_dir)
            torch.save({'model':prep_model, 'model_state_dict': state,
                    'optimizer_state_dict': optimizer_state}, cache_dir+name+'.tar')
        
    def save_model_mat(self, cache_dir='../cache/', name=None, cache_model=None):
        NN_grid=torch.from_numpy(np.vstack([self.grid[i].reshape(-1) for i in range(self.grid.shape[0])]).T).float()
        if cache_model==None:
            cache_model = torch.nn.Sequential(
                torch.nn.Linear(self.grid.shape[0], 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 1)
            )
        optimizer = torch.optim.Adam(cache_model.parameters(), lr=0.001)
        model_res=self.model.reshape(-1,1)
    
        def closure():
            optimizer.zero_grad()
            loss = torch.mean((cache_model(NN_grid)-model_res)**2)
            loss.backward()
            return loss

        loss=np.inf
        t=1
        while loss>1e-5 and t<1e5:
            loss = optimizer.step(closure)
            t+=1

        self.save_model(cache_model,cache_model.state_dict(),optimizer.state_dict(),cache_dir=cache_dir, name=name)

    def scheme_interp(self, trained_model, cache_verbose=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        loss = torch.mean(torch.square(trained_model(self.grid)-self.model(self.grid)))

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((trained_model(self.grid) - self.model(self.grid))**2)
            loss.backward()
            return loss
        t=1
        while loss>1e-5 and t<1e5:
            optimizer.step(closure)
            loss = torch.mean(torch.square(trained_model(self.grid) - self.model(self.grid)))
            t+=1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(t,loss))
        
        return self.model, optimizer.state_dict()


    def cache_retrain(self, cache_checkpoint, cache_verbose=False):
        # do nothing if cache is empty
        if cache_checkpoint==None:
            optimizer_state = None
            return self.model,optimizer_state
        # if models have the same structure use the cache model state
        if str(cache_checkpoint['model']) == str(self.model):
            self.model = cache_checkpoint['model']
            self.model.load_state_dict(cache_checkpoint['model_state_dict'])
            self.model.eval()
            optimizer_state=cache_checkpoint['optimizer_state_dict']
            if cache_verbose:
                print('Using model from cache')
        # else retrain the input model using the cache model 
        else:
            optimizer_state = None
            model_state = None
            cache_model=cache_checkpoint['model']
            cache_model.load_state_dict(cache_checkpoint['model_state_dict'])
            cache_model.eval()
            self.model, optimizer_state = self.scheme_interp(cache_model, cache_verbose=cache_verbose)
        return self.model, optimizer_state

    def cache(self, cache_dir, nmodels, lambda_bound, cache_verbose, model_randomize_parameter, cache_model):
        r =self.create_random_fn(model_randomize_parameter) 
        if self.mode == 'NN' or self.mode == 'autograd':
            cache_checkpoint, min_loss=self.cache_lookup(cache_dir=cache_dir, nmodels=nmodels, cache_verbose=cache_verbose, lambda_bound=lambda_bound)
            self.model, optimizer_state = self.cache_retrain(cache_checkpoint, cache_verbose=cache_verbose)

            self.model.apply(r)

            return self.model, min_loss

        elif self.mode == 'mat':
            NN_grid=torch.from_numpy(np.vstack([self.grid[i].reshape(-1) for i in range(self.grid.shape[0])]).T).float()
            if cache_model == None:
                cache_model = torch.nn.Sequential(
                    torch.nn.Linear(NN_grid.shape[-1], 100),
                    torch.nn.Tanh(),
                    torch.nn.Linear(100, 100),
                    torch.nn.Tanh(),
                    torch.nn.Linear(100, 100),
                    torch.nn.Tanh(),
                    torch.nn.Linear(100, 1)
                )
            operator_NN = EquationMixin.op_dict_to_list(self.equal_cls.operator)
            for term in operator_NN:
                if type(term[0])==torch.Tensor:
                    term[0]=term[0].reshape(-1)
                if callable(term[0]):
                    print("Warning: coefficient is callable, it may lead to wrong cache item choice")
            
            equal = Equation(NN_grid, operator_NN, self.equal_cls.bconds).set_strategy('NN')
            model_cls = Model_prepare(NN_grid, equal, cache_model, 'NN')
            cache_checkpoint, min_loss = model_cls.cache_lookup(cache_dir=cache_dir, nmodels=nmodels, cache_verbose=cache_verbose, lambda_bound=lambda_bound)
            prepared_model, optimizer_state = model_cls.cache_retrain(cache_checkpoint, cache_verbose=cache_verbose)

            prepared_model.apply(r)

            if len(self.grid.shape)==2:
                self.model = prepared_model(NN_grid).reshape(self.grid.shape).detach()
            else:
                self.model = prepared_model(NN_grid).reshape(self.grid[0].shape).detach()

            min_loss = self.loss_evaluation(lambda_bound=lambda_bound)

            return self.model, min_loss


