import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import argparse
from gridword_env.gridworld import GridWorld
import concurrent.futures
import chex
import jax.numpy as jnp
import jax
from algorithm import fPG

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
            
def save_results_to_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def save_results_to_pickle_two_data(file_path, data1,data2):
    with open(file_path, 'wb') as f:
        pickle.dump(data1, f)
        pickle.dump(data2,f)


def run_fpg(argument):
    full_path = argument[0]
    run = argument[1]
    env = argument[2]
    step_size = argument[3]
    temp = argument[4]
    true_objective_values_file =  full_path +  '/' +  ',step_'+ str(step_size) + ',temperature_'+ str(temperature)+ ',run_'+str(run) +',true_objective.pkl'
    np.random.seed(run)
    dict  = argument[5]
    fpg_method = fPG(env, step_size, temp,  **dict)
    true_objective_values, minimal_probability = fpg_method.train()
    save_results_to_pickle_two_data(true_objective_values_file, true_objective_values, minimal_probability)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='launching the experiment')
    parser.add_argument("--alpha", type=float, default=0.5, help="choose the parameter for the tsallis algorithm")
    parser.add_argument("--environment", type=int, default=0, help="0 is for the Gridword environment")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--step", type=float, nargs="+", default=[0.001], help="Step size")
    parser.add_argument("--n_iteration", type=float, default=10000, help="Number of iteration T")    
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.05], help="temperature lambda(s)")
    parser.add_argument("--runs", type=int, default=5, help="number of runs")
    parser.add_argument("--len_truncation", type=int, default=20, help="lenght  of the truncation H")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per iteration B")
    parser.add_argument("--verbose", type=bool, default=True, help="verbose")

    args = parser.parse_args()
    args_dict = vars(args)
    runs = args.runs
    alpha = args.alpha
    steps = args.step
    temeperatures = args.temperature
    steps = args.step
    if args.environment ==0:
        environment = "gridword"
        
    parent_directory = './experiments/' + str(alpha)  +'/'+ environment
    create_folder_if_not_exists(parent_directory)
    np.random.seed(0)
    for step in steps:
        for temperature in temeperatures:
            if args.environment ==0:
                env = GridWorld(3, 3, walls=((1, 1),(1,1)), success_probability=0.8)
                full_path = parent_directory
                seeds = [k for k in range(args.runs)] 
                with concurrent.futures.ProcessPoolExecutor(max_workers=runs) as executor:
                    arguments = [[full_path, seed, env, step, temperature, args_dict] for seed in seeds]
                    run_fpg(arguments[0])
                    #results = list(executor.map(run_fpg, arguments))   
            else:
                print("The environnement shoud be either 0: 0 is for Gridword")