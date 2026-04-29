import numpy as np
import random
import functools

def select_parents_loop(func):
    @functools.wraps(func)
    def wrapper(self, population, *args, **kwargs):
        for s_i in range(0, len(population) - 1, 2):
            parent_1 = population[s_i]
            parent_2 = population[s_i + 1]
            
            assert len(parent_1) == len(parent_2), 'the length of the subjects should be the same'
            
            res = func(self, parent_1, parent_2, *args, **kwargs)
            
            if res is not None:
                population[s_i], population[s_i + 1] = res
        return population
    return wrapper

class Crossover():
    def __init__(self, crossover_pro:float, selected_num:int=None):
        self.crossover_pro = crossover_pro
        self.selected_num = selected_num
    
    @select_parents_loop
    def single_point_crossover(self, parent_1:np.ndarray, parent_2:np.ndarray):
        
        if random.random() < self.crossover_pro:
            pos_i = random.sample(range(1, len(parent_1)), k=1)[0]
            print(f'choose the position index: {pos_i}')
            temp = parent_1[pos_i:].copy()
            parent_1[pos_i:] = parent_2[pos_i:]
            parent_2[pos_i:] = temp     
        
        return parent_1, parent_2
    
    @select_parents_loop
    def Partial_map_crossover(self, parent_1:np.ndarray, parent_2:np.ndarray):
        """used in order problem"""
        if random.random() < self.crossover_pro:
            position = sorted(random.sample(range(0, len(parent_1)), k=2))
            print(f'choose the position index: {position[0]} and {position[1]}')
            sub_chromosome_1 = parent_1[position[0]:position[1]].copy()
            sub_chromosome_2 = parent_2[position[0]:position[1]].copy()
            map1 = {sub_chromosome_2[j]: sub_chromosome_1[j] for j in range(len(sub_chromosome_2))}
            map2 = {sub_chromosome_1[j]: sub_chromosome_2[j] for j in range(len(sub_chromosome_1))}
            def repair_conflict(val, mapping):
                while val in mapping:
                    val = mapping[val]
                return val
                                              
            for j in range(len(parent_1)):
                if j < position[0] or j >= position[1]:
                    parent_1[j] = repair_conflict(parent_1[j], map1)
                    parent_2[j] = repair_conflict(parent_2[j], map2)
            
            parent_1[position[0]:position[1]], parent_2[position[0]:position[1]] = sub_chromosome_2, sub_chromosome_1
        
        return parent_1, parent_2
    
    @select_parents_loop
    def order_crossover(self, parent_1:np.ndarray, parent_2:np.ndarray):
        """used in order problem"""
        if random.random() < self.crossover_pro:
            position = sorted(random.sample(range(0, len(parent_1)), k=2))
            print(f'choose the position index: {position[0]} and {position[1]}')
            set1 = set(parent_1[:position[0]]) | set(parent_1[position[1]:])
            set2 = set(parent_2[:position[0]]) | set(parent_2[position[1]:])
            
            new_p1 = parent_1.copy()
            new_p2 = parent_2.copy()
            
            def find_other_position(pt, set_):
                return [p for p in pt if p in set_ ]
            
            def generate_new_chroms(other_chrom, p):
                new_p = p.copy()
                new_p[:position[0]] = other_chrom[:position[0]]
                new_p[position[1]:] = other_chrom[position[0]:]
                return new_p
                
            other_chrom_1 = find_other_position(new_p1, set2)
            parent_1 = generate_new_chroms(other_chrom_1, new_p2)
            
            other_chrom_2 = find_other_position(new_p2, set1)
            parent_2 = generate_new_chroms(other_chrom_2, new_p1)
            
        return parent_1, parent_2

    @select_parents_loop
    def position_based_crossover(self, parent_1:np.ndarray, parent_2:np.ndarray):
        if random.random() < self.crossover_pro:
            if self.selected_num is None:
                selected_num = random.randint(1, len(parent_1) - 1)
            else:
                selected_num = self.selected_num
            print(f'choose {selected_num} genes')
            position = sorted(random.sample(range(len(parent_1)), k=selected_num))
            print(f'choose the position indexs: {position}')
            
            genes_1 = parent_1[position]
            genes_2 = parent_2[position]
            
            other_genes_1 = parent_1[~np.isin(parent_1, genes_2)]
            other_genes_2 = parent_2[~np.isin(parent_2, genes_1)]
            
            mask = np.ones(len(parent_1), dtype=bool)
            mask[position] = False
            new_parent_1 = parent_1.copy()
            new_parent_2 = parent_2.copy()
            
            new_parent_1[mask] = other_genes_2
            new_parent_2[mask] = other_genes_1
            
            return new_parent_1, new_parent_2
        return parent_1, parent_2
    
    @select_parents_loop
    def order_based_crossover(self, parent_1:np.ndarray, parent_2:np.ndarray):
        if random.random() < self.crossover_pro:
            if self.selected_num is None:
                selected_num = random.randint(1, len(parent_1) - 1)
            else:
                selected_num = self.selected_num
            print(f'choose {selected_num} genes')
            position = sorted(random.sample(range(len(parent_1)),k=selected_num))
            print(f'choose the position indexs: {position}')

            def generate_new_chroms(p, genes):
                g_i = 0
                n_p = p.copy()
                for i in range(len(n_p)):
                    if n_p[i] in genes:
                        n_p[i] = genes[g_i]
                        g_i += 1
                        if g_i >= len(genes):
                            break
                return n_p
            
            genes_1 = parent_1[position]
            genes_2 = parent_2[position]
            
            new_parent_1 = generate_new_chroms(parent_1, genes_2)
            new_parent_2 = generate_new_chroms(parent_2, genes_1)
            
            return new_parent_1, new_parent_2
        return parent_1, parent_2



import copy

def run_crossover_tests():
    base_p1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    base_p2 = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])

    methods = [
        'single_point_crossover',
        'Partial_map_crossover',
        'order_crossover',
        'position_based_crossover',
        'order_based_crossover'
    ]

    for method_name in methods:
        print(f"\n{'='*20} 测试: {method_name} {'='*20}")
        
        random.seed(42)
        np.random.seed(42)
        
        co = Crossover(crossover_pro=1.0, selected_num=4)
        
        population = [base_p1.copy(), base_p2.copy()]
        print(f"父代 1: {population[0]}")
        print(f"父代 2: {population[1]}")
        
        func = getattr(co, method_name)
        res_pop = func(population)
        
        print(f"子代 1: {res_pop[0]}")
        print(f"子代 2: {res_pop[1]}")
        
        if method_name != 'single_point_crossover': 
            is_p1_valid = len(np.unique(res_pop[0])) == len(base_p1)
            is_p2_valid = len(np.unique(res_pop[1])) == len(base_p2)
            print(f"-> 子代 1 是否仍为无重复的合法排列: {is_p1_valid}")
            print(f"-> 子代 2 是否仍为无重复的合法排列: {is_p2_valid}")

if __name__ == '__main__':
    run_crossover_tests()