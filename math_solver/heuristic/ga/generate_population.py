import numpy as np

class GeneratePopulation():
    def __init__(self, population_num:int):
        self.population_num = population_num
    
    def generate_order_type(self, subject_size:int) -> np.ndarray:
        population = np.zeros((self.population_num, subject_size))
        
        for i in range(self.population_num):
            subject = np.random.permutation(range(subject_size))
            population[i] = subject
        return population
    
    def generate_number_type(self, subject_size: int, low: float, high: float) -> np.ndarray:
        population = np.zeros((self.population_num, subject_size))
        
        for i in range(self.population_num):
            subject = np.random.uniform(low, high, subject_size)
            population[i] = subject
            
        return population

    