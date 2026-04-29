import numpy as np
from typing import List, Tuple

class GeneticAlgo():
    def __init__(self, 
                 population_size:int, 
                 crossover_type:str, 
                 crossover_pro:float,
                 mutate_type:str,
                 mutate_pro:float,
                 mutation_range:float,
                 selection:str,
                 elitism:bool,
                 obj_type:str,
                 iteration_num:int = 3,
                 ):
        
        self.population_size = population_size
        self.crossover_type = crossover_type
        self.crossover_pro = crossover_pro
        self.mutate_type = mutate_type
        self.mutate_pro = mutate_pro
        self.mutation_range = mutation_range
        self.selection = selection
        self.elitism = elitism
        self.iteration_num = iteration_num
        self.obj_type = obj_type
        self.optim_operator = np.min if self.obj_type == 'Minimize' else np.max
        
        self.iteration_process()
    
    def generate_population(self, population_size:int) -> np.ndarray:
        raise NotImplementedError
    
    def crossover(self, population:np.ndarray, crossover_pro:float) -> np.ndarray:
        raise NotImplementedError
    
    def mutate(self, population:np.ndarray, mutate_pro:float) -> np.ndarray:
        raise NotImplementedError
    
    def check_constraints(self, population:np.ndarray) -> np.ndarray[bool]:
        raise NotImplementedError
        
    def remove_infeasible_chromosome(self, population:np.ndarray, feasibility_array:np.ndarray) -> np.ndarray:
        pass
    
    def cal_obj_value(self, population:np.ndarray) -> np.ndarray[float]:
        raise NotImplementedError
    
    def select_elitism(self, population:np.ndarray, obj_values_array:np.ndarray,) -> None:
        pass
    
    def select_chromosomes(self, population:np.ndarray, obj_values_array: np.ndarray,) -> np.ndarray:
        pass
    
    def check_termination(self, best_obj_lst:List[float]=None) -> bool:
        return False
    
    def iteration_process(self,) -> Tuple[np.ndarray, List]:
        best_obj_lst = []
        is_terminate = False
        iter_i = 1
        population = self.generate_population(self.population_size)
        feasibility_array = self.check_constraints(population)
        population = self.remove_infeasible_chromosome(population, feasibility_array)
        obj_values_array = self.cal_obj_value(population)
        best_obj_lst.append(self.optim_operator(obj_values_array))
        
        if self.elitism:
            self.select_elitism(population, obj_values_array)
            
        population = self.select_chromosomes(population, obj_values_array)
        
        while iter_i <= self.iteration_num and not is_terminate:
            population = self.crossover(population, self.crossover)
            population = self.mutate(population, self.mutate_pro)
            feasibility_array = self.check_constraints(population)
            population = self.remove_infeasible_chromosome(population, feasibility_array)
            obj_values_array = self.cal_obj_value(population)
            best_obj_lst.append(self.optim_operator(obj_values_array))
            
            if self.elitism:
                self.select_elitism(population, obj_values_array)
                
            population = self.select_chromosomes(population, obj_values_array)
            is_terminate = self.check_termination(best_obj_lst)
            