import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import numpy as np
from math_solver.exact.dynamic_prog import ResourceAllocationDP

import pytest



def test_scientist_allocation():
    """
    scientists allocation problem test
    """
    P = np.array([
        [0.4, 0.2, 0.15],
        [0.6, 0.4, 0.2],
        [0.8, 0.5, 0.3]
    ]).T
    
    solver = ResourceAllocationDP(stages_num=3, total_resources=2, mode='Minimize', operator='multiply')
    best_prob, allocation = solver.solve(P)
    
    assert np.isclose(best_prob, 0.06), f"Expected 0.06, but got {best_prob}"
    
    expected_allocation = np.array([1, 0, 1])
    np.testing.assert_array_equal(allocation, expected_allocation)

def test_election_allocation():
    """
    election allocation problem test
    """
    V = np.array([
        [0, 22, 39, 52, 60, 66, 71],
        [0, 17, 35, 51, 62, 70, 77],
        [0, 30, 40, 44, 47, 49, 50],
        [0, 10, 21, 29, 36, 42, 47]
    ]).T
    
    solver = ResourceAllocationDP(stages_num=4, total_resources=6, mode='Maximize', operator='sum')
    
    max_votes, allocation = solver.solve(V)
    
    assert max_votes == 120, f"Expected 120, but got {max_votes}"
    
    expected_allocation = np.array([2,3,1,0])
    np.testing.assert_array_equal(allocation, expected_allocation)

def test_invalid_parameters():

    # 测试错误的 mode
    with pytest.raises(ValueError, match="mode can only be either Maximize or Minimize"):
        ResourceAllocationDP(stages_num=3, total_resources=2, mode='WrongMode', operator='sum')

    # 测试错误的 operator
    with pytest.raises(ValueError, match="operator can only be either sum or multiply"):
        ResourceAllocationDP(stages_num=3, total_resources=2, mode='Maximize', operator='WrongOp')