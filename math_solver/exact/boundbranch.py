import numpy as np
import pulp
import os

from abc import ABC
from typing import List

import graphviz

os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'

class TreePlotter:
    """独立的绘图类，专门用于渲染 Branch & Bound 的搜索树"""
    
    def __init__(self,):
        self.tree_nodes = {}
        self.tree_edges = []
        
        self.color_map = {
            "branching": "lightblue",
            "optimal": "lightgreen",
            "pruned": "lightgray",
            "infeasible": "salmon"
        }

    def generate_plot(self):
        """generate Graphviz plot"""
        dot = graphviz.Digraph(comment='Branch and Bound Tree')
        dot.attr(dpi='150')
        dot.attr('node', shape='box', style='filled', fontname='Helvetica', fontsize='10')

        for node_id, data in self.tree_nodes.items():
            color = self.color_map.get(data["status"], "white")
            dot.node(str(node_id), data["label"], fillcolor=color)

        for parent, child, cond in self.tree_edges:
            dot.edge(str(parent), str(child), label=cond, fontname='Helvetica', fontsize='9', fontcolor='red')

        return dot

    def save_and_view(self, filename="branch_bound_tree"):
        """save plot"""
        dot = self.generate_plot()
        dot.render(filename, format="png", view=True)
        print(f"🌲 save {filename}.png successfully")
        
class BranchBound(ABC):
    def __init__(self, model, ILP_type:str):
        if not (ILP_type == 'Maximize' or ILP_type == 'Minimize'):
            raise Exception('ILP_type can only be either Maximize or Minimize')
        
        self.ILP_type = ILP_type
        self.int_var_list = self.define_int_variables()
        self.nint_var_list = self.define_nint_variables()
        if self.ILP_type == 'Maximize':
            self.incumbent = -np.inf
        elif self.ILP_type == 'Minimize':
            self.incumbent = np.inf
        
        self.node_count = 0
        
        self.plotter =  TreePlotter()
        self.define_objective(model)
        self.define_constraints(model)
        print("""
              LP relexation: drop integrality: all of the variable is continuous
              the objective value we get is the upper bound on the IP optimum 
              and we should choose the fractional variable to branch
              """)
        print(f'initial incumbent is {self.incumbent}')
        self.boundbranch(model)
        self.plotter.save_and_view("my_branch_bound_tree")
        
        print(f"the best objective value (Incumbent): {self.incumbent}")
        print(f"the optimal variable group: {self.best_solution}")


    
    def define_int_variables(self,) -> List:
        """define the integer variables"""
        pass
    
    def define_nint_variables(self,) -> List:
        """define the non-integer variables"""
        pass
    
    def define_objective(self, model):
        """define the objective"""
        pass
    
    def define_constraints(self, model):
        """define the constraints"""
        pass
    
    def add_bound_constraints(self, model, var, left:int = None, right:int = None):
        """based on the new branch condition to update constraints"""
        
        new_model = model.copy()
        
        if left is not None:
            new_model += var <= left
        elif right is not None:
            new_model += var >= right
        else:
            raise Exception('there is no bound here')
        
        return new_model
    
    def solve_lr(self, model):
        """solve Linear programming problem """
        
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        return model.status
    
    def boundbranch(self, model, bound_var=None, parent_id=None, branch_cond=""):
        self.node_count += 1
        current_id = self.node_count
        if parent_id is not None:
            self.plotter.tree_edges.append((parent_id, current_id, branch_cond))
            
        status = self.solve_lr(model)
        model_status = pulp.LpStatus[status]
        print(f'state: {model_status}')
        if model_status == 'Infeasible':
            'fathoming rule 1'
            print('fathoming by rule 1: the model is infeasibility')
            self.plotter.tree_nodes[current_id] = {"label": "Infeasible\n(Pruned)", "status": "infeasible"}
            return 
        
        obj_value = pulp.value(model.objective)
        var_strs = [f"{var.name}={var.varValue:.2f}" for var in self.int_var_list if var.varValue is not None]
        var_info = "\n".join(var_strs)
        print(f'objective value: {obj_value}')
        print(f'LP solution : {var_info}')
        if ((obj_value <= self.incumbent and self.ILP_type == 'Maximize') 
            or (obj_value >= self.incumbent and self.ILP_type == 'Minimize')):
            'fathoming rule 2'
            self.plotter.tree_nodes[current_id] = {"label": f"Pruned (Bound)\nZ = {obj_value:.2f}\n{var_info}", "status": "pruned"}
            print('fathoming by rule 2: the objective value is worser than incumbent')
            return 
        
        int_var_num = np.sum([1 if np.isclose(var.varValue, round(var.varValue)) else 0 for var in self.int_var_list])
        if int_var_num == len(self.int_var_list):
            'fathoming rule 3'
            print('fathoming by rule 3: all of the variables to the solution is integer')
            if ((obj_value >= self.incumbent and self.ILP_type == 'Maximize') 
                or (obj_value <= self.incumbent and self.ILP_type == 'Minimize')):
                print(f'update the incumbent: objective value {obj_value} is better than incumbent{self.incumbent}')
                self.incumbent = obj_value

            self.best_solution = {var.name: var.varValue for var in self.int_var_list}
            self.plotter.tree_nodes[current_id] = {"label": f"⭐ Incumbent\nZ = {obj_value:.2f}\n{var_info}", "status": "optimal"}
            return
        else:
            print('still fractional -> eligible for branching')
        
        self.plotter.tree_nodes[current_id] = {"label": f"Branching\nZ = {obj_value:.2f}\n{var_info}", "status": "branching"}
        
        if bound_var is None or np.isclose(bound_var.varValue, round(bound_var.varValue)):
            var_list = np.array([var.varValue for var in self.int_var_list])
            fractional_part = np.argsort([0 if np.isclose(var_value, round(var_value)) else(var_value - np.floor(var_value)) for var_value in var_list])
            bound_var = self.int_var_list[fractional_part[-1]]


        left = np.floor(bound_var.varValue)
        right = np.ceil(bound_var.varValue)
        
        left_model = self.add_bound_constraints(model, bound_var, left=left)
        print(f"\nadd {bound_var.name} <= {left}")
        self.boundbranch(left_model, bound_var, current_id, branch_cond=f"{bound_var.name} <= {left}")
        
        print(f"\nadd {bound_var.name} >= {right}")
        right_model = self.add_bound_constraints(model, bound_var, right=right)
        self.boundbranch(right_model, bound_var, current_id, branch_cond=f"{bound_var.name} >= {right}")
    
            
class TestIP_1(BranchBound):
    def __init__(self, model, ILP_type:str):
        super().__init__(model, ILP_type)

    def define_int_variables(self) -> List:

        self.x1 = pulp.LpVariable('x1', lowBound=0, cat='Continuous')
        self.x2 = pulp.LpVariable('x2', lowBound=0, cat='Continuous')
        return [self.x1, self.x2]


    def define_nint_variables(self) -> List:
        return [] 

    def define_objective(self, model):
        model += (3 * self.x1 + 4 * self.x2), 'Objective'

    def define_constraints(self, model):
        model += 2 * self.x1 + 3 * self.x2 <= 12, 'Constraint 1'
        model += 3 * self.x1 + 2 * self.x2 <= 12, 'Constraint 2'
        

class TestIP_2(BranchBound):
    def __init__(self, model, ILP_type:str):
        super().__init__(model, ILP_type)

    def define_int_variables(self) -> List:

        self.x1 = pulp.LpVariable('x1', lowBound=0, cat='Continuous')
        self.x2 = pulp.LpVariable('x2', lowBound=0, cat='Continuous')
        self.x3 = pulp.LpVariable('x3', lowBound=0, cat='Continuous')
        return [self.x1, self.x2, self.x3]


    def define_nint_variables(self) -> List:
        return [] 

    def define_objective(self, model):
        model += (4 * self.x1 + 5 * self.x2 + 3 * self.x3), 'Objective'

    def define_constraints(self, model):
        model += 2 * self.x1 + 1 * self.x2 + 1 * self.x3 >= 8, 'Constraint 1'
        model += 1 * self.x1 + 2 * self.x2 + 1 * self.x3 >= 6, 'Constraint 2'

class TestIP_3(BranchBound):
    def __init__(self, model, ILP_type:str):
        super().__init__(model, ILP_type)

    def define_int_variables(self) -> List:

        self.x1 = pulp.LpVariable('x1', lowBound=0, upBound=1, cat='Continuous')
        self.x2 = pulp.LpVariable('x2', lowBound=0, upBound=1, cat='Continuous')
        self.x3 = pulp.LpVariable('x3', lowBound=0, upBound=1, cat='Continuous')
        self.x4 = pulp.LpVariable('x4', lowBound=0, upBound=1, cat='Continuous')
        return [self.x1, self.x2, self.x3, self.x4]


    def define_nint_variables(self) -> List:
        return [] 

    def define_objective(self, model):
        model += (40 * self.x1 + 50 * self.x2 + 60 * self.x3 + 70 * self.x4), 'Objective'

    def define_constraints(self, model):
        model += 2 * self.x1 + 3 * self.x2 + 4 * self.x3 + 5 * self.x4 <= 7, 'Constraint 1'



if __name__ == "__main__":
    base_model = pulp.LpProblem('LP2', pulp.LpMaximize)
    solver = TestIP_3(base_model, 'Maximize')
    
    
    
        