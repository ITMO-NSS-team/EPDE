
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from epde.solver.cache import Model_prepare
import os
import sys
import datetime

def grid_format_prepare(coord_list, mode='NN'):
    if type(coord_list)==torch.Tensor:
        print('Grid is a tensor, assuming old format, no action performed')
        return coord_list
    if mode=='NN' or mode =='autograd':
        if len(coord_list)==1:
            coord_list=torch.tensor(coord_list)
            grid=coord_list.reshape(-1,1).float()
        else:
            coord_list_tensor=[]
            for item in coord_list:
                if isinstance(item,(np.ndarray)):
                    coord_list_tensor.append(torch.from_numpy(item))
                else:
                    coord_list_tensor.append(item)
            grid=torch.cartesian_prod(*coord_list_tensor).float()
    elif mode=='mat':
        grid = np.meshgrid(*coord_list)
        grid = torch.tensor(np.array(grid))
    return grid

class Solver(Model_prepare):
    def __init__(self, grid, equal_cls, model, mode):
        super().__init__(grid, equal_cls, model, mode)


    def optimizer_choice(self, optimizer, learning_rate):
        if optimizer=='Adam':
            if self.mode =='NN' or self.mode == 'autograd':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            elif self.mode =='mat':
                optimizer = torch.optim.Adam([self.model.requires_grad_()], lr=learning_rate)
        
        elif optimizer=='SGD':
            if self.mode =='NN' or self.mode == 'autograd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
            elif self.mode =='mat':
                optimizer = torch.optim.SGD([self.model.requires_grad_()], lr=learning_rate)
        
        elif optimizer=='LBFGS':
            if self.mode =='NN' or self.mode == 'autograd':
                optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate)
            elif self.mode =='mat':
                optimizer = torch.optim.LBFGS([self.model.requires_grad_()], lr=learning_rate)
        
        else:
            print('Wrong optimizer chosen, optimization was not performed')
            return self.model
        
        return optimizer


    def solution_print(self,title=None,solution_print=False,solution_save=True,save_dir=None):
        if save_dir==None:
            img_dir=os.path.join(os.path.dirname( __file__ ), 'img')
            if not(os.path.isdir(img_dir)):
                os.mkdir(img_dir)
            directory=os.path.abspath(os.path.join(img_dir,str(datetime.datetime.now().timestamp())+'.png'))
        else:
            directory=os.path.join(save_dir, str(datetime.datetime.now().timestamp())+'.png')
        if self.mode == 'NN' or self.mode == 'autograd':
            nvars_model = self.model(self.grid).shape[-1]
            nparams = self.grid.shape[1]
            fig = plt.figure()
            for i in range(nvars_model):
                if nparams == 1:
                    ax1 = fig.add_subplot(1,nvars_model,i+1)
                    if title!=None:
                        ax1.set_title(title+' variable {}'.format(i))
                    ax1.scatter(self.grid.detach().numpy().reshape(-1), self.model(self.grid).detach().numpy().reshape(-1))
                else:
                    ax1 = fig.add_subplot(1,nvars_model,i+1,projection='3d')

                    if title!=None:
                        ax1.set_title(title+' variable {}'.format(i))

                    ax1.plot_trisurf(self.grid[:, 0].detach().numpy().reshape(-1), self.grid[:, 1].detach().numpy().reshape(-1),
                                self.model(self.grid)[:,i].detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
                    ax1.set_xlabel("x1")
                    ax1.set_ylabel("x2")
            if solution_print:
                plt.show()
            if solution_save:
                plt.savefig(directory)
            plt.close()
        elif self.mode == 'mat':
            nparams = self.grid.shape[0]

            if nparams == 1:
                fig = plt.figure()
                plt.scatter(self.grid.reshape(-1), self.model.detach().numpy().reshape(-1))
                if solution_print:
                    plt.show()
                if solution_save:
                    plt.savefig(directory)
                plt.close()
            elif nparams == 2:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                if title!=None:
                    ax.set_title(title)
                ax.plot_trisurf(self.grid[0].reshape(-1), self.grid[1].reshape(-1),
                                self.model.reshape(-1).detach().numpy(), cmap=cm.jet, linewidth=0.2, alpha=1)
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                if solution_print:
                    plt.show()
                if solution_save:
                    plt.savefig(directory)
                plt.close()


    def solve(self, lambda_bound=10, verbose=False, learning_rate=1e-4, eps=1e-5, tmin=1000,
                            tmax=1e5,nmodels=None,name=None, abs_loss=None,
                            use_cache=True,cache_dir='../cache/',cache_verbose=False,
                            save_always=False,print_every=100,cache_model=None,
                            patience=5,loss_oscillation_window=100,no_improvement_patience=1000,
                            model_randomize_parameter=0, optimizer_mode='Adam',step_plot_print=False,step_plot_save=False,image_save_dir=None):
        # prepare input data to uniform format 
        r = self.create_random_fn(model_randomize_parameter)
        #  use cache if needed
        if use_cache:
            self.model, min_loss = self.cache(cache_dir=cache_dir,
                                              nmodels=nmodels,
                                              lambda_bound=lambda_bound,
                                              cache_verbose=cache_verbose,
                                              model_randomize_parameter=model_randomize_parameter,
                                              cache_model=cache_model)

        optimizer = self.optimizer_choice(optimizer_mode, learning_rate)
        
        if True:
        #if not use_cache:
            min_loss = self.loss_evaluation(lambda_bound=lambda_bound)   
    
        save_cache=False
    
        if min_loss >0.1 or save_always: # why 0.1?
            save_cache=True
    
    
        # standard NN stuff
        if verbose:
            print('[{}] initial (min) loss is {}'.format(datetime.datetime.now(),min_loss))
    
        t = 0
    
        last_loss=np.zeros(loss_oscillation_window)+float(min_loss)
        line=np.polyfit(range(loss_oscillation_window),last_loss,1)
    

        def closure():
            nonlocal cur_loss
            optimizer.zero_grad()
            loss = self.loss_evaluation(lambda_bound=lambda_bound)
            
            loss.backward()
            cur_loss = loss.item()
            return loss
    
        stop_dings=0
        t_imp_start=0
        # to stop train proceduce we fit the line in the loss data
        #if line is flat enough 5 times, we stop the procedure
        cur_loss=min_loss
        while stop_dings<=patience:
            optimizer.step(closure)

            last_loss[t%loss_oscillation_window]=cur_loss

        
            if cur_loss<min_loss:
                min_loss=cur_loss
                t_imp_start=t

            if verbose:
                info_string='Step = {} loss = {:.6f} normalized loss line = {:.6f}x+{:.6f}. There was {} stop dings already.'.format(t, cur_loss, line[0]/cur_loss,line[1]/cur_loss, stop_dings+1)

            if t%loss_oscillation_window==0:
                line=np.polyfit(range(loss_oscillation_window),last_loss,1)
                if abs(line[0]/cur_loss) < eps and t>0:
                    stop_dings+=1
                    if self.mode =='NN' or self.mode =='autograd':
                        self.model.apply(r)
                    if verbose:
                        print('[{}] Oscillation near the same loss'.format(datetime.datetime.now()))
                        print(info_string)
                        if step_plot_print or step_plot_save:
                            self.solution_print(title='Iteration = ' + str(t),solution_print=step_plot_print,solution_save=step_plot_save,save_dir=image_save_dir)
        
            if (t-t_imp_start==no_improvement_patience):
                if verbose:
                    print('[{}] No improvement in {} steps'.format(datetime.datetime.now(),no_improvement_patience))
                    print(info_string)
                    if step_plot_print or step_plot_save:
                        self.solution_print(title='Iteration = ' + str(t),solution_print=step_plot_print,solution_save=step_plot_save,save_dir=image_save_dir)
                t_imp_start=t
                stop_dings+=1
                if self.mode =='NN' or self.mode =='autograd':
                        self.model.apply(r)

            
            if abs_loss!=None and cur_loss<abs_loss:
                if verbose:
                    print('[{}] Absolute value of loss is lower than threshold'.format(datetime.datetime.now()))
                    print(info_string)
                    if step_plot_print or step_plot_save:
                        self.solution_print(title='Iteration = ' + str(t),solution_print=step_plot_print,solution_save=step_plot_save,save_dir=image_save_dir)
                stop_dings+=1


            if print_every!=None and (t % print_every == 0) and verbose:
                print('[{}] Print every {} step'.format(datetime.datetime.now(),print_every))
                print(info_string)
                if step_plot_print or step_plot_save:
                    self.solution_print(title='Iteration = ' + str(t),solution_print=step_plot_print,solution_save=step_plot_save,save_dir=image_save_dir)

            t += 1
            if t > tmax:
                break
        if (save_cache and use_cache) or save_always:
            if self.mode=='mat':
                self.save_model_mat(cache_dir=cache_dir,name=name)
            else:
                self.save_model(self.model, self.model.state_dict(),optimizer.state_dict(),cache_dir=cache_dir,name=name)
        return self.model





