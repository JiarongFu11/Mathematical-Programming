import numpy as np
from abc import ABC

class BacktrackLineSearch(ABC):
    def __init__(self, num_vars:int, mode:str, delta:float=0.001, u:float=0.1, num_iteration:int=50):
        if mode not in ['Maximize', 'Minimize']:
            raise ValueError('mode can only be Maximize or Minimize')
        self.mode = mode
        self.direction = 1 if self.mode == 'Maximize' else -1
        self.num_vars = num_vars
        self.delta = delta
        self.u = u
        self.num_iteration = num_iteration
        
        self.init_params()
        self.iter_process()
    
    def check_constraints(self, vars_values:np.ndarray) -> bool:
        return True
    
    def cal_objective(self, var_values:np.ndarray) -> float:
        raise  NotImplementedError
    
    def cal_grad(self, ori_obj:float) -> np.ndarray:
        grads = np.zeros(self.num_vars)
        for i in range(len(self.vars_lst)):
            var_values = self.vars_lst.copy()
            var_values[i] = var_values[i] + self.delta
            grads[i] = (self.cal_objective(var_values) - ori_obj) / self.delta            
            
        print(f'compute the gradients: {grads}')
        
        return grads
    
    def update_params(self, ori_obj:float, grad:np.ndarray) -> None:
        lambda_ = 1
        search_dir = grad * self.direction
        print(f'direction for {self.mode} is {search_dir}')
        while True:
            if lambda_ < 1e-8:
                print("Warning: step size is too small so stop search lambda")
                break
            
            print(f'test lambda_ = {lambda_}')
            
            new_vars = self.vars_lst + lambda_ * search_dir
            new_obj = self.cal_objective(new_vars)
            print(f'trial point {new_vars} and the objective value is {new_obj}')
            
            expected_obj = ori_obj + self.u * lambda_ * grad @ search_dir.T
            print(f'the expected objective is {expected_obj}')
            
            if not self.check_constraints(new_vars):
                lambda_ /= 2
                continue
            else:
                print('constraints satisfy')
            
            if self.mode == 'Maximize' and new_obj >= expected_obj:
                self.vars_lst = new_vars
                print(f'lambda = {lambda_} satisfied')
                print(f'objective value = {new_obj} is larger than {expected_obj}')
                print(f'the parameters is updated to {self.vars_lst}')
                print('\n')
                break
            elif self.mode == 'Minimize' and new_obj <= expected_obj:
                self.vars_lst = new_vars
                print(f'objective value = {new_obj} is less than {expected_obj}')
                print(f'the parameters is updated to {self.vars_lst}')
                print('\n')
                break
            else:
                print(f'reject lambda {lambda_}')
                print('\n')
                lambda_ /= 2
    
    def init_params(self,) -> None:
        is_feasibility = False
        while not is_feasibility:
            self.vars_lst = np.random.rand(self.num_vars)
            is_feasibility = self.check_constraints(self.vars_lst)
            print(f'satisfy feasibility : {self.vars_lst}')
        
        obj = self.cal_objective(self.vars_lst)
        print(f'initial objective {obj}')
    
    def iter_process(self,) -> None:
        is_stationary_point = False
        iteration_ = 0
        while (not is_stationary_point) and (iteration_ <= self.num_iteration):
            ori_obj = self.cal_objective(self.vars_lst)
            grads = self.cal_grad(ori_obj)
            if grads @ grads.T < 1e-5:
                print('find the stationary point: iteration process end')
                break
            self.update_params(ori_obj, grads)
            iteration_ += 1

class TestBLS1(BacktrackLineSearch):
    def __init__(self):
        super().__init__(num_vars=1, mode='Maximize')
        
    def check_constraints(self, vars_values:np.ndarray) -> None:
        return True
    
    def cal_objective(self, vars_values:np.ndarray) -> float:
        return -vars_values[0] ** 2 + 10 * vars_values[0]
    
    def init_params(self,) -> None:
        self.vars_lst = np.array([3.0])
        obj = self.cal_objective(self.vars_lst)

class TestBLS2(BacktrackLineSearch):
    def __init__(self):
        super().__init__(num_vars=1, mode='Minimize')
        
    def check_constraints(self, vars_values:np.ndarray) -> None:
        return True
    
    def cal_objective(self, vars_values:np.ndarray) -> float:
        return vars_values[0] ** 2
    
    def init_params(self,) -> None:
        self.vars_lst = np.array([2.0])
        obj = self.cal_objective(self.vars_lst)
        
if __name__ == '__main__':
    t = TestBLS2()
        