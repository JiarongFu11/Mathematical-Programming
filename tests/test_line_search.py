import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import numpy as np
from math_solver.nonlinear.line_search import BacktrackLineSearch

import pytest
import numpy as np

class MinimizeTestFunction(BacktrackLineSearch):
    def cal_objective(self, var_values: np.ndarray) -> float:
        return np.sum(var_values**2)

class MaximizeTestFunction(BacktrackLineSearch):
    def cal_objective(self, var_values: np.ndarray) -> float:
        return -np.sum(var_values**2)

class ConstrainedTestFunction(BacktrackLineSearch):
    def check_constraints(self, vars_values: np.ndarray) -> bool:
        return np.all(vars_values >= 0.5)

    def cal_objective(self, var_values: np.ndarray) -> float:
        return np.sum(var_values**2)

def test_invalid_mode():
    """测试：传入无效的 mode 是否正确抛出异常"""
    with pytest.raises(ValueError, match="mode can only be Maximize or Minimize"):
        MinimizeTestFunction(num_vars=2, mode='FindRoot')

def test_minimize_optimization():
    """测试：Minimize 模式能否正确收敛到极小值"""
    np.random.seed(42) 
    optimizer = MinimizeTestFunction(num_vars=2, mode='Minimize', delta=0.001, num_iteration=100)
    
    final_obj = optimizer.cal_objective(optimizer.vars_lst)
    
    assert final_obj < 1e-2, f"Minimize failed, final objective value is {final_obj}"
    
    np.testing.assert_allclose(optimizer.vars_lst, np.array([0.0, 0.0]), atol=0.1)

def test_maximize_optimization():
    """测试：Maximize 模式能否正确收敛到极大值"""
    np.random.seed(42)
    optimizer = MaximizeTestFunction(num_vars=2, mode='Maximize', delta=0.001, num_iteration=100)
    
    final_obj = optimizer.cal_objective(optimizer.vars_lst)
    
    assert final_obj > -1e-2, f"Maximize failed, final objective value is {final_obj}"
    
    np.testing.assert_allclose(optimizer.vars_lst, np.array([0.0, 0.0]), atol=0.1)

def test_constraints_handling():
    """测试：优化过程是否遵守 check_constraints 中定义的约束"""
    np.random.seed(42)
    optimizer = ConstrainedTestFunction(num_vars=2, mode='Minimize', delta=0.001, num_iteration=100)
    
    assert np.all(optimizer.vars_lst >= 0.5), f"Constraints violated! Final vars: {optimizer.vars_lst}"