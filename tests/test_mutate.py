import numpy as np
import pytest
from unittest.mock import patch

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from math_solver.heuristic.ga.mutate import Mutate


@pytest.fixture
def setup_population():
    """
    提供标准的基础父代种群 
    """
    return np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0]
    ])

def test_inversion(setup_population):
    mutator = Mutate(mutation_pro=1.0)
    
    with patch('random.sample', return_value=[2, 5]):
        res = mutator.inversion(setup_population.copy())
        
        # 索引 2:6 (包含5) [2, 3, 4, 5] 反转为 [5, 4, 3, 2]
        expected_p1 = np.array([0, 1, 5, 4, 3, 2, 6, 7])
        np.testing.assert_array_equal(res[0], expected_p1)
        assert len(np.unique(res[0])) == 8

def test_insertion(setup_population):
    mutator = Mutate(mutation_pro=1.0)
    
    with patch('random.sample', return_value=[2, 5]):
        res = mutator.insertion(setup_population.copy())
        
        # 将位置 5 的元素(值为5) 插入到位置 2，原本 2,3,4 右移到 3,4,5
        expected_p1 = np.array([0, 1, 5, 2, 3, 4, 6, 7])
        np.testing.assert_array_equal(res[0], expected_p1)
        
        # 如果这里断言失败，说明你没有按上面的提示修复那个导致基因丢失的 Bug
        assert len(np.unique(res[0])) == 8 

def test_reciprocal(setup_population):
    mutator = Mutate(mutation_pro=1.0)
    
    with patch('random.sample', return_value=[1, 5]):
        res = mutator.reciprocal(setup_population.copy())
        
        expected_p1 = np.array([0, 5, 2, 3, 4, 1, 6, 7])
        np.testing.assert_array_equal(res[0], expected_p1)
        assert len(np.unique(res[0])) == 8

def test_pair_wise_exchange(setup_population):
    mutator = Mutate(mutation_pro=1.0)
    
    with patch('random.sample', return_value=[3]):
        res = mutator.pair_wise_exchange(setup_population.copy())
        
        expected_p1 = np.array([0, 1, 2, 4, 3, 5, 6, 7])
        np.testing.assert_array_equal(res[0], expected_p1)
        assert len(np.unique(res[0])) == 8

def test_two_opt(setup_population):
    mutator = Mutate(mutation_pro=1.0)
    
    with patch('random.sample', return_value=[2]):
        res = mutator.two_opt(setup_population.copy())
        
        expected_p1 = np.array([0, 1, 4, 3, 2, 5, 6, 7])
        np.testing.assert_array_equal(res[0], expected_p1)
        assert len(np.unique(res[0])) == 8

def test_three_opt(setup_population):
    mutator = Mutate(mutation_pro=1.0)
    
    with patch('random.sample', return_value=[1]):
        res = mutator.three_opt(setup_population.copy())
        
        expected_p1 = np.array([0, 4, 2, 3, 1, 5, 6, 7])
        np.testing.assert_array_equal(res[0], expected_p1)
        assert len(np.unique(res[0])) == 8

def test_mutation_zero_probability(setup_population):
    mutator = Mutate(mutation_pro=0.0)
    res = mutator.inversion(setup_population.copy())
    np.testing.assert_array_equal(res, setup_population)