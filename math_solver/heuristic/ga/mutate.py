import numpy as np
import functools
import random

def select_subject_loop(func):
    @functools.wraps(func)
    def wrapper(self, population, *args, **kwargs):
        for subject_i in range(0,population.shape[0]):
            subject = population[subject_i]
            new_subject = func(self, subject)
            population[subject_i] = new_subject
        return population
    return wrapper
    

class Mutate():
    def __init__(self, mutation_pro:float):
        self.mutation_pro = mutation_pro
    
    @select_subject_loop
    def inversion(self, subject:np.ndarray) -> np.ndarray:
        if random.random() < self.mutation_pro:
            position = sorted(random.sample(range(len(subject)), k=2))
            subject[position[0]:position[1] + 1] = subject[position[0]:position[1] + 1][::-1]
        return subject
    
    @select_subject_loop
    def insertion(self, subject:np.ndarray) -> np.ndarray:
        if random.random() < self.mutation_pro:
            position = sorted(random.sample(range(len(subject)), k=2))
            temp = subject[position[1]]
            subject[position[0] + 1:position[1] + 1] = subject[position[0]:position[1]]
            subject[position[0]] = temp
        return subject
    
    @select_subject_loop
    def reciprocal(self, subject:np.ndarray) -> np.ndarray:
        if random.random() < self.mutation_pro:
            position = sorted(random.sample(range(len(subject)), k=2))
            temp = subject[position[0]]
            subject[position[0]] = subject[position[1]]
            subject[position[1]] = temp
        return subject

    @select_subject_loop
    def pair_wise_exchange(self, subject:np.ndarray) -> np.ndarray:
        if random.random() < self.mutation_pro:
            position = random.sample(range(len(subject) - 1), k=1)[0]
            temp = subject[position]
            subject[position] = subject[position + 1]
            subject[position + 1] = temp
        return subject
    
    @select_subject_loop
    def two_opt(self, subject:np.ndarray) -> np.ndarray:
        if random.random() < self.mutation_pro:
            position = random.sample(range(len(subject) - 2), k=1)[0]
            temp = subject[position]
            subject[position] = subject[position + 2]
            subject[position + 2] = temp
        return subject
    
    @select_subject_loop
    def three_opt(self, subject:np.ndarray) -> np.ndarray:
        if random.random() < self.mutation_pro:
            position = random.sample(range(len(subject) - 3), k=1)[0]
            temp = subject[position]
            subject[position] = subject[position + 3]
            subject[position + 3] = temp
        return subject
    
    @select_subject_loop
    def mutate_interval(self, subject:np.ndarray, low:float=-0.2, high:float=0.2, ):
        if random.random() < self.mutation_pro:
            mutation_array = np.random.uniform(1 + low, 1 + high, size=len(subject))
            subject = subject @ mutation_array.T
        return subject