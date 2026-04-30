import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from math_solver.heuristic.ga.ga_selction import Selection


@pytest.fixture
def setup_data():
    population = np.array([
        [0, 1, 2], 
        [3, 4, 5],  
        [6, 7, 8],  
        [9, 0, 1]   
    ])
    
    obj_values = np.array([10, 50, 5, 20])
    return population, obj_values

def test_get_elite_maximize(setup_data):
    population, obj_values = setup_data
    selector = Selection(elitism=True, selection_num=4, optim_type='Maximize')
    
    elite = selector._get_elite(population, obj_values)
    
    expected_elite = np.array([[3, 4, 5]])
    np.testing.assert_array_equal(elite, expected_elite)

def test_get_elite_minimize(setup_data):
    population, obj_values = setup_data
    selector = Selection(elitism=True, selection_num=4, optim_type='Minimize')
    
    elite = selector._get_elite(population, obj_values)
    
    expected_elite = np.array([[6, 7, 8]])
    np.testing.assert_array_equal(elite, expected_elite)

def test_roulette_wheel_shape_and_elitism(setup_data):
    population, obj_values = setup_data
    selector = Selection(elitism=True, selection_num=5, optim_type='Maximize')
    
    res = selector.roulette_wheel_selection(population, obj_values)
    
    assert res.shape == (5, 3)
    
    np.testing.assert_array_equal(res[0], np.array([3, 4, 5]))

def test_tournament_shape_and_elitism(setup_data):
    population, obj_values = setup_data
    selector = Selection(elitism=True, selection_num=3, optim_type='Minimize')
    
    res = selector.tournament(population, obj_values)
    
    assert res.shape == (3, 3)
    
    np.testing.assert_array_equal(res[0], np.array([6, 7, 8]))

def test_tournament_logic_mocked(setup_data):
    population, obj_values = setup_data
    selector = Selection(elitism=False, selection_num=2, optim_type='Maximize')
    
    with patch('random.sample', side_effect=[[0, 1], [2, 3]]):
        res = selector.tournament(population, obj_values)
        
        expected_population = np.array([
            [3, 4, 5],
            [9, 0, 1]
        ])
        np.testing.assert_array_equal(res, expected_population)
