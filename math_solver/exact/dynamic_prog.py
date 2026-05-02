import numpy as np

class ResourceAllocationDP():   
    def __init__(self, stages_num:int, total_resources:int, mode:str, operator:str='sum'):
        self.stages_num = stages_num
        self.total_resources = total_resources
        
        if not (mode == 'Maximize' or mode == 'Minimize'):
            raise ValueError('mode can only be either Maximize or Minimize')
        self.mode = mode
        self.mode_operator = np.max if mode == 'Maximize' else np.min
        self.mode_idx_op = np.argmax if mode == 'Maximize' else np.argmin
        
        if not (operator == 'sum' or operator == 'multiply'):
            raise ValueError('operator can only be either sum or multiply')
        self.operator = np.add if operator == 'sum' else np.multiply

        fill_val = -np.inf if mode == 'Maximize' else np.inf
        self.DP_table = np.full((total_resources + 1, self.stages_num + 1), fill_val)
        self.DP_table[0, 0] = 1.0 if operator == 'multiply' else 0.0
        
        self.policy_table = np.zeros((total_resources + 1, self.stages_num + 1))
            
    def solve(self, reward_matrix:np.ndarray):
        """solve the DP problem through forward method
           reward_matrix: shape (total_resources, stages_num)"""
        for stage in range(1, self.stages_num + 1):
            for res_idx in range(self.total_resources + 1):
                fitness_lst = []
                for cur_res_idx in range(res_idx + 1):
                    f = self.operator(self.DP_table[res_idx - cur_res_idx, stage - 1], 
                                      reward_matrix[cur_res_idx, stage - 1])
                    fitness_lst.append(f)
                
                self.DP_table[res_idx, stage] = self.mode_operator(fitness_lst)
                self.policy_table[res_idx, stage] = self.mode_idx_op(fitness_lst)

        best_fitness = self.mode_operator(self.DP_table[:, -1])
        used_resource = self.mode_idx_op(self.DP_table[:, -1])
        
        return best_fitness, self._traceback(used_resource)
    
    def _traceback(self, used_resource:int):
        """traceback the path by the policy table"""
        path = np.zeros(self.stages_num)
        n_stage = self.stages_num

        while n_stage > 0:
            path[n_stage - 1] = self.policy_table[used_resource, n_stage]
            used_resource -= int(self.policy_table[used_resource, n_stage])
            n_stage -= 1
        
        return path

if __name__ == '__main__':
    pass