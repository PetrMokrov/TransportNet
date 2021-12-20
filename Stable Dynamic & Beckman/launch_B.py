import argparse
import pandas as pd
import numpy as np
import data_handler as dh
import model as md
import time
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

import importlib
importlib.reload(dh)
importlib.reload(md)

import numba

import sys
from platform import python_version
import graph_tool

beckmann_save = 'beckmann_results/'

parser = argparse.ArgumentParser(description='Launches Stable Dynamics convertence')
parser.add_argument('--method', help='method solving the task', type=str, default='ustm')
parser.add_argument('--sp', help='method for solving the shortest path problems', type=str, default='dijkstra')
parser.add_argument('--eps_number', help='eps_abs parameter number', type=int)

args = parser.parse_args()
admissible_methods = ['ustm', 'ugd']
assert args.method in admissible_methods, f"method should be in [{admissible_methods}], got {args.method}"

sp_map = {
    'dijkstra': ('our_dijkstra', 'dense'),
    't_swsf': ('sparse_t_swsf', 'sparse'),
    'tradeoff': ('sparse_t_swsf_tradeoff', 'sparse')}
assert args.sp in list(sp_map.keys()), f"SP method should be in {list(sp_map.keys())}, got {args.sp}"
assert args.eps_number >= 0
assert args.eps_number < 9

solver_method = args.method
sp_algo, tswsf_type = sp_map[args.sp]
eps_number = args.eps_number

net_name = 'Anaheim_net.tntp'
trips_name = 'Anaheim_trips.tntp'

handler = dh.DataHandler()
graph_data = handler.GetGraphData(net_name, columns = ['init_node', 'term_node', 'capacity', 'free_flow_time'])
graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)

# model = md.Model(graph_data, graph_correspondences, 
#                     total_od_flow, mu = 0.25, rho = 0.15, sp_recompute='sparse_t_swsf')
epsilons = np.logspace(3,-0.5,8)

model = md.Model(graph_data, graph_correspondences, 
                    total_od_flow, mu = 0.25, rho = 0.15, sp_recompute=sp_algo)

max_iter = 10000

eps_abs = epsilons[eps_number]
print('eps_abs =', eps_abs)
solver_kwargs = {'eps_abs': eps_abs,
                 'max_iter': max_iter, 'stop_crit': 'dual_gap',
                 'verbose': True, 'verbose_step': 2000, 'save_history': True, 'tswsf_type':tswsf_type}
tic = time.perf_counter()
result = model.find_equilibrium(solver_name = solver_method, composite = True, solver_kwargs = solver_kwargs)
toc = time.perf_counter()
print('Elapsed time: {:.0f} sec'.format(toc - tic))
print('Time ratio =', np.max(result['times'] / graph_data['graph_table']['free_flow_time']))
print('Flow excess =', np.max(result['flows'] / graph_data['graph_table']['capacity']) - 1, end = '\n\n')

result['eps_abs'] = eps_abs
result['elapsed_time'] = toc - tic
with open(beckmann_save + 'anaheim_result_' + solver_method + '_sp_' + args.sp + '_eps_abs_' + str(eps_number) + '_beckmann.pickle', 'wb') as f:
    pickle.dump(result, f)
print('Well done!')
