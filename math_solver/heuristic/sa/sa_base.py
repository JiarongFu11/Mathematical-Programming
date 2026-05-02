import numpy as np
import random
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from math_solver.heuristic.ga.mutate import Mutate
from math_solver.heuristic.ga.generate_population import GeneratePopulation
class SimulatedAnnealing():
    def __init__(self, 
                 population_num:int, 
                 iteration_num:int, 
                 temperature:float, 
                 cooling_rate:float,
                 obj_direction:str):
        self.direction = 1 if obj_direction == 'Maximize' else -1
        self.select_op = np.argmax if obj_direction == 'Maximize' else np.argmin
        self.iteration_num = iteration_num
        self.population_num = population_num
        self.cooling_rate = cooling_rate
        self.temperature = temperature
        self.population_generator = GeneratePopulation(population_num=self.population_num)
    
    def generate_population(self, population_num:int):
        raise NotImplementedError
    
    def selection(self, population: np.ndarray):
        obj_values = [self.cal_obj(ind) for ind in population]
        best_idx = self.select_op(obj_values)
        return population[best_idx]
    
    def decide_acceptance(self, origin_obj:float, new_obj:float, temp:float) -> bool:
        delta = self.direction * (new_obj - origin_obj)
        
        if delta >= 0:
            return True
            
        accptance_prob = np.exp(delta / temp)
        if random.random() < accptance_prob:
            return True
        return False
    
    def perturb(self, origin_subject:int):
        raise NotImplementedError

    
    def cal_obj(self, subject):
        raise NotImplementedError
    
    def check_constraints(self, subject:np.ndarray) -> bool:
        raise NotImplementedError
    
    def termination(self, subject:np.ndarray) -> bool:
        return False
    
    def iterative(self):
        population = self.generate_population(self.population_num)
        subject = self.selection(population)
        
        ori_obj = self.cal_obj(subject) 
        iter_i = 0
        
        while iter_i <= self.iteration_num:
            new_subject = self.perturb(subject)
            iter_i += 1
            
            if not self.check_constraints(new_subject): 
                continue
            
            new_obj = self.cal_obj(new_subject)
            self.temperature *= self.cooling_rate
            is_accept = self.decide_acceptance(ori_obj, new_obj, self.temperature)
            
            if is_accept:
                subject = new_subject
                ori_obj = new_obj 
                
            if self.termination(subject):
                break
            
        return subject
