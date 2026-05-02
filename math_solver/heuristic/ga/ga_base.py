import numpy as np
import sys
import os
from typing import List, Tuple


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from math_solver.heuristic.ga.crossover import Crossover
from math_solver.heuristic.ga.generate_population import GeneratePopulation
from math_solver.heuristic.ga.ga_selection import Selection
from math_solver.heuristic.ga.mutate import Mutate

class GeneticAlgo():
    def __init__(self, 
                 population_size:int, 
                 crossover_pro:float,
                 mutate_pro:float,
                 elitism:bool,
                 obj_type:str,
                 iteration_num:int = 3,
                 seletion_num:int = 100
                 ):
        
        self.population_size = population_size
        self.crossover_pro = crossover_pro
        self.mutate_pro = mutate_pro
        self.elitism = elitism
        self.iteration_num = iteration_num
        self.obj_type = obj_type
        self.optim_operator = np.min if self.obj_type == 'Minimize' else np.max
        
        self.population_generator = GeneratePopulation(population_size)
        self.selector = Selection(elitism, seletion_num, obj_type)
        self.crossover_engine = Crossover(crossover_pro)
        self.mutate_engine = Mutate(mutate_pro)
        
        self.iteration_process()
        
    def generate_population(self, population_size:int) -> np.ndarray:
        raise NotImplementedError
    
    def crossover(self, population:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def mutate(self, population:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def check_constraints(self, population:np.ndarray) -> np.ndarray[bool]:
        raise NotImplementedError
        
    def remove_infeasible_chromosome(self, population:np.ndarray, feasibility_array:np.ndarray[bool]) -> np.ndarray:
        return population[feasibility_array]
            
    
    def cal_obj_value(self, population:np.ndarray) -> np.ndarray[float]:
        raise NotImplementedError
    
    def select_chromosomes(self, population:np.ndarray, obj_values_array: np.ndarray, elitism:bool) -> np.ndarray:
        raise NotImplementedError
    
    def check_termination(self, best_obj_lst:List[float]=None) -> bool:
        return False
    
    def iteration_process(self,) -> Tuple[np.ndarray, np.ndarray]:
        best_obj_lst = []
        is_terminate = False
        iter_i = 1
        population = self.generate_population(self.population_size)
        feasibility_array = self.check_constraints(population)
        population = self.remove_infeasible_chromosome(population, feasibility_array)
        obj_values_array = self.cal_obj_value(population)
        best_obj_lst.append(self.optim_operator(obj_values_array))
        
        while iter_i <= self.iteration_num and not is_terminate:
                
            population = self.select_chromosomes(population, obj_values_array, self.elitism)
            
            population = self.crossover(population)
            population = self.mutate(population)
            feasibility_array = self.check_constraints(population)
            population = self.remove_infeasible_chromosome(population, feasibility_array)
            obj_values_array = self.cal_obj_value(population)
            print(self.optim_operator(obj_values_array))
            best_obj_lst.append(self.optim_operator(obj_values_array))
            
            is_terminate = self.check_termination(best_obj_lst)
            print(iter_i)
            
            iter_i += 1
        
        return population, obj_values_array

class Test1(GeneticAlgo):
    def __init__(self, 
                 population_size:int, 
                 crossover_pro:float,
                 mutate_pro:float,
                 elitism:bool,
                 obj_type:str,
                 iteration_num:int = 2,
                 ):
        super().__init__(
            population_size, 
            crossover_pro,
            mutate_pro,
            elitism,
            obj_type,
            iteration_num,
            )
        
    
    def generate_population(self, population_size:int) -> np.ndarray:
        return self.population_generator.generate_number_type(3, 0, 0.5)
    
    def crossover(self, population:np.ndarray) -> np.ndarray:
        return self.crossover_engine.single_point_crossover(population)
    
    def mutate(self, population:np.ndarray) -> np.ndarray:
        return self.mutate_engine.mutate_interval(population, low=-0.2, high=0.2)
    
    def check_constraints(self, population:np.ndarray) -> np.ndarray[bool]:
        res_1 = population @ np.array([8, 6, 2]).T
        res_2 = population @ np.array([1, 1, 2]).T
        return (res_1 <= 13) & (res_2 <= 4)
    
    def cal_obj_value(self, population:np.ndarray) -> np.ndarray[bool]:
        obj_array = population @ np.array([8, 6, 1]).T
        return obj_array
    
    def select_chromosomes(self, population:np.ndarray, obj_values_array:np.ndarray, elitism:bool) -> np.ndarray:
        return self.selector.roulette_wheel_selection(population, obj_values_array)
    
if __name__ == '__main__':
    Test1(
        population_size=10000,
        crossover_pro=0.9,
        mutate_pro=0.9,
        elitism=True,
        obj_type='Maximize',
        iteration_num = 2,
        ).iteration_process()
        
        
        
        
        
        
        
        