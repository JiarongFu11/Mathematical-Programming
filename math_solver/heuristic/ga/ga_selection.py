import numpy as np
import random

class Selection():
    def __init__(self, elitism: bool, selection_num: int, optim_type: str):
        self.elitism = elitism
        self.selection_num = selection_num
        self.optim_type = optim_type
        
    def _get_elite(self, population: np.ndarray, obj_values_array: np.ndarray) -> np.ndarray:
        """Helper method to find the single best individual based on optim_type."""
        if self.optim_type == 'Maximize':
            best_idx = np.argmax(obj_values_array)
        else:
            best_idx = np.argmin(obj_values_array)
        return population[best_idx : best_idx + 1]

    def roulette_wheel_selection(self, population: np.ndarray, obj_values_array: np.ndarray) -> np.ndarray:
        num_to_select = self.selection_num - 1 if self.elitism else self.selection_num
        
        if getattr(self, 'optim_type', 'Maximize') == 'Minimize':
            max_val = np.max(obj_values_array)
            fitness = max_val - obj_values_array + 1e-6
        else:
            min_val = np.min(obj_values_array)
            if min_val < 0:
                fitness = obj_values_array - min_val + 1e-6
            else:
                fitness = obj_values_array + 1e-6
                
        total_fitness = np.sum(fitness)
        prob = fitness / total_fitness
        cum_prob = np.cumsum(prob)

        random_spins = np.random.rand(num_to_select)
        selected_indices = np.searchsorted(cum_prob, random_spins)
        selected_population = population[selected_indices]
        
        if self.elitism:
            elite = self._get_elite(population, obj_values_array)
            selected_population = np.vstack((elite, selected_population))
            
        return selected_population
    
    def tournament(self, population: np.ndarray, obj_values_array: np.ndarray) -> np.ndarray:
        num_to_select = self.selection_num - 1 if self.elitism else self.selection_num
        selected_population = np.zeros((num_to_select, population.shape[1]))
        
        for i in range(num_to_select):
            s_p = random.sample(range(len(population)), k=2)
            idx_0, idx_1 = s_p[0], s_p[1]
            val_0, val_1 = obj_values_array[idx_0], obj_values_array[idx_1]
            
            if self.optim_type == 'Maximize':
                winner_idx = idx_0 if val_0 > val_1 else idx_1
            elif self.optim_type == 'Minimize':
                winner_idx = idx_0 if val_0 < val_1 else idx_1
                
            selected_population[i] = population[winner_idx]
            
        if self.elitism:
            elite = self._get_elite(population, obj_values_array)
            selected_population = np.vstack((elite, selected_population))
            
        return selected_population
    
    def random_selection(self, population: np.ndarray, obj_values_array: np.ndarray = None) -> np.ndarray:
        num_to_select = self.selection_num - 1 if self.elitism else self.selection_num
        
        selected_indices = np.random.choice(len(population), size=num_to_select, replace=True)
        selected_population = population[selected_indices]
        
        if self.elitism and obj_values_array is not None:
            elite = self._get_elite(population, obj_values_array)
            selected_population = np.vstack((elite, selected_population))
            
        return selected_population