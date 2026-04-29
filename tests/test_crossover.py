import numpy as np
import random
import pytest

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from math_solver.heuristic.ga.crossover import Crossover

import pytest

@pytest.fixture
def setup_population():
    """
    Pytest fixture: 提供标准的基础父代种群，供所有测试用例复用
    """
    parent_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    parent_2 = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
    # 必须深拷贝，防止不同测试之间互相污染
    return [parent_1.copy(), parent_2.copy()]


def test_single_point_crossover(setup_population):
    """
    单点交叉测试
    """
    # 固定随机种子，确保 assert_array_equal 的预期结果完全一致
    random.seed(42)
    np.random.seed(42)
    
    co = Crossover(crossover_pro=1.0)
    res_pop = co.single_point_crossover(setup_population)
    
    expected_p1 = np.array([0, 7, 6, 5, 4, 3, 2, 1, 0])
    expected_p2 = np.array([8, 1, 2, 3, 4, 5, 6, 7, 8])
    
    np.testing.assert_array_equal(res_pop[0], expected_p1)
    np.testing.assert_array_equal(res_pop[1], expected_p2)


def test_partial_map_crossover(setup_population):
    """
    部分映射交叉 (PMX) 测试
    """
    random.seed(42)
    
    co = Crossover(crossover_pro=1.0)
    res_pop = co.Partial_map_crossover(setup_population)
    
    expected_p1 = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
    expected_p2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    np.testing.assert_array_equal(res_pop[0], expected_p1)
    np.testing.assert_array_equal(res_pop[1], expected_p2)
    
    # 针对排列问题的特性：断言子代依旧是合法的全排列 (无重复基因)
    assert len(np.unique(res_pop[0])) == 9, "PMX 产生了重复基因"
    assert len(np.unique(res_pop[1])) == 9, "PMX 产生了重复基因"


def test_order_crossover(setup_population):
    """
    顺序交叉 (OX) 测试
    """
    random.seed(42)
    
    co = Crossover(crossover_pro=1.0)
    res_pop = co.order_crossover(setup_population)
    
    expected_p1 = np.array([8, 7, 6, 5, 0, 1, 2, 3, 4])
    expected_p2 = np.array([0, 1, 2, 3, 8, 7, 6, 5, 4])
    
    np.testing.assert_array_equal(res_pop[0], expected_p1)
    np.testing.assert_array_equal(res_pop[1], expected_p2)
    
    assert len(np.unique(res_pop[0])) == 9, "OX 产生了重复基因"
    assert len(np.unique(res_pop[1])) == 9, "OX 产生了重复基因"


def test_position_based_crossover(setup_population):
    """
    基于位置的交叉 (PBX) 测试
    """
    random.seed(42)
    
    co = Crossover(crossover_pro=1.0, selected_num=4)
    res_pop = co.position_based_crossover(setup_population)
    
    expected_p1 = np.array([0, 1, 8, 7, 4, 5, 6, 3, 2])
    expected_p2 = np.array([8, 7, 0, 1, 4, 3, 2, 5, 6])
    
    np.testing.assert_array_equal(res_pop[0], expected_p1)
    np.testing.assert_array_equal(res_pop[1], expected_p2)
    
    assert len(np.unique(res_pop[0])) == 9, "PBX 产生了重复基因"
    assert len(np.unique(res_pop[1])) == 9, "PBX 产生了重复基因"


def test_order_based_crossover(setup_population):
    """
    基于顺序的交叉 (OBX) 测试
    """
    random.seed(42)
    
    co = Crossover(crossover_pro=1.0, selected_num=4)
    res_pop = co.order_based_crossover(setup_population)
    
    expected_p1 = np.array([0, 1, 8, 3, 7, 5, 6, 4, 2])
    expected_p2 = np.array([8, 7, 0, 5, 1, 3, 2, 4, 6])
    
    np.testing.assert_array_equal(res_pop[0], expected_p1)
    np.testing.assert_array_equal(res_pop[1], expected_p2)
    
    assert len(np.unique(res_pop[0])) == 9, "OBX 产生了重复基因"
    assert len(np.unique(res_pop[1])) == 9, "OBX 产生了重复基因"


def test_crossover_probability(setup_population):
    """
    测试交叉概率控制
    当 crossover_pro=0 时，种群应原样返回，不发生变化
    """
    co = Crossover(crossover_pro=0.0)
    res_pop = co.single_point_crossover(setup_population)
    
    np.testing.assert_array_equal(res_pop[0], setup_population[0])
    np.testing.assert_array_equal(res_pop[1], setup_population[1])