
import argparse
import math
import os
import numpy as np
from dwave.system import DWaveSampler
import pickle
import dimod
from matplotlib import pyplot
import dwave_networkx as dnx
import matplotlib.cm as cm
from dwave.cloud.api.exceptions import RequestTimeout
from os.path import exists
import lzma
import matplotlib as mpl
import networkx as nx
from statistics import mean
from scipy.stats import sem
from tqdm import tqdm
import neal
import pandas as pd
import itertools
import seaborn as sn
from scipy.optimize import curve_fit
from netgraph import Graph
import lmfit

def pegasus_control_line_from_linear(q):
    (u, w, k, z) = dnx.pegasus_coordinates(16).linear_to_pegasus(q)
    return int(4*(u%2) + (k%2) + 2*(z%2))

def get_sol_dict(var, record):
    sol = {}
    for i in range(len(var)):
        sol[var[i]] = record['sample'][i]
    return sol

def get_average_mag_per_qubit(vars, samples):
    average_mag = {}
    for var in vars:
        curr_qubit_mag = 0
        for i in range(len(samples)):
            curr_qubit_mag += samples[i][var]
        curr_qubit_mag /= len(samples)
        average_mag[var] = curr_qubit_mag
    return average_mag
def average_mag_per_line(average_mag, qubits_in_line):
    for i in range(8):
        total_mag = 0
        for q in qubits_in_line[i]:
            total_mag += average_mag[q]
        total_mag /= len(qubits_in_line[i])
        print(total_mag)
    exit()

def get_sol_dict_from_dwave_output(solutions):
    sol = []
    for solution in solutions:
        records = solution.record
        for record in records:
            sol.append(get_sol_dict(solution.variables, record))
    return sol

def sign_of(num):
    if num > 0:
        return 1
    else:
        return -1

def get_average_coupler_correlation(couplers, solution_dicts, i, j):
    corr = 0
    for sol in solution_dicts:
        c = (i,j)
        corr += (sign_of(couplers[c]) * sol[i] * sol[j] +1) * 0.5
    corr /= len(solution_dicts)
    return corr

def get_coupler_correlation(couplers, solution_dicts):
    coupler_corr = {}
    for coupler in couplers:
        coupler_corr[coupler] = get_average_coupler_correlation(couplers, solution_dicts, coupler[0], coupler[1])
    return coupler_corr

def get_anneal_lines(qubit_in_use, couplers):
    anneal_line_qubits = []
    anneal_line_couplers = []
    for ite in range(8):
        anneal_line_qubits.append([])
        anneal_line_couplers.append([])
    for q in qubit_in_use:
        anneal_line_qubits[pegasus_control_line_from_linear(q)].append(q)


    for c in couplers:
        # Do not count FM coupler for offset shimming
        if couplers[c] < 0:
            continue
        anneal_line_left = pegasus_control_line_from_linear(c[0])
        anneal_line_right =  pegasus_control_line_from_linear(c[1])
        if (c[0], c[1]) not in anneal_line_couplers[anneal_line_left] and (c[1], c[0]) not in anneal_line_couplers[anneal_line_left]:
            anneal_line_couplers[anneal_line_left].append(c)
        if (c[0], c[1]) not in anneal_line_couplers[anneal_line_right] and (c[1], c[0]) not in anneal_line_couplers[anneal_line_right]:
            anneal_line_couplers[anneal_line_right].append(c)

    return anneal_line_qubits, anneal_line_couplers
        
def average_frust_between_anneal_lines(line1, line2, coupler_frust):
    total_frust = 0
    num_coupler = 0
    for coupler in coupler_frust:
        if pegasus_control_line_from_linear(coupler[0]) == line1 and pegasus_control_line_from_linear(coupler[1]) == line2:
            total_frust += coupler_frust[coupler]
            num_coupler += 1
        if pegasus_control_line_from_linear(coupler[0]) == line2 and pegasus_control_line_from_linear(coupler[1]) == line1:
            total_frust += coupler_frust[coupler]
            num_coupler += 1
    if total_frust == 0:
        return 0
    else:
        return total_frust/num_coupler


def get_buckets(dim_x, dim_y, num_embedding, embedding, couplers):
    buckets = []
    for i in range(dim_y):
        curr_bucket = []
        for ite in range(num_embedding):
            curr_col = embedding[ite, :, i] 
            
            for m in range(len(curr_col)-1):
                if couplers[(curr_col[m], curr_col[m+1])] > 0:
                    curr_bucket.append((curr_col[m], curr_col[m+1]))

            if couplers[(curr_col[-1], curr_col[0])] > 0:
                curr_bucket.append((curr_col[-1], curr_col[0]))

        buckets.append(curr_bucket)
    

    for i in range(dim_y-1):
        curr_bucket_odd = []
        curr_bucket_even = []

        for ite in range(num_embedding):
            left_col = embedding[ite, :, i] 
            right_col = embedding[ite, :, i + 1] 
            for m in range(len(left_col)):
                if m % 2 == 0:
                    curr_bucket_even.append((left_col[m], right_col[m]))
                else:
                    curr_bucket_odd.append((left_col[m], right_col[m]))
        buckets.append(curr_bucket_odd)
        buckets.append(curr_bucket_even)
    return buckets
       

def get_average_bucket_corr(buckets, average_corr):
    bucket_average = []
    bucket_vals = []
    for i in range(len(buckets)):
        bucket_vals.append({})
    for i in range(len(buckets)):
        b = buckets[i]
        bucket_total = 0
        for coupler in b:
            bucket_vals[i][coupler] = average_corr[coupler]
            bucket_total += average_corr[coupler]
        bucket_average.append(bucket_total / len(b))
    return bucket_average, bucket_vals

def get_average_line_frustration(anneal_line_couplers, average_coupler_corr):
    average_line_frust = []
    for i in range(8):
        line = anneal_line_couplers[i]
        sum_line_frust = 0
        for coupler in line:
            sum_line_frust += average_coupler_corr[coupler]
        if len(line) != 0:
            sum_line_frust /= len(line)
        
        average_line_frust.append(sum_line_frust)
    return average_line_frust

def get_coloring(embedding, k_away_from_boundry_omitted):
    dim_x = embedding.shape[0]
    dim_y = embedding.shape[1]
    color_0 = []
    color_1 = []
    color_2 = []

    for i in range(dim_y):
        if i < k_away_from_boundry_omitted or dim_y - i <= k_away_from_boundry_omitted:
            continue
        curr_col = embedding[:, i]
        if i % 2 == 0:
            for m in range(int(dim_x/2)):
                if m % 3 == 0:
                    color_0.append(curr_col[2 * m])
                    color_0.append(curr_col[2 * m +1])
                elif m % 3 == 1:
                    color_1.append(curr_col[2 * m])
                    color_1.append(curr_col[2 * m +1])
                else:
                    color_2.append(curr_col[2 * m])
                    color_2.append(curr_col[2 * m +1])
        else:
            for m in range(int(dim_x/2)):
                if m % 3 == 0:
                    color_1.append(curr_col[2 * m])
                    color_2.append(curr_col[2 * m + 1])
                elif m % 3 == 1:
                    color_2.append(curr_col[2 * m])
                    color_0.append(curr_col[2 * m +1])
                else:
                    color_0.append(curr_col[2 * m])
                    color_1.append(curr_col[2 * m +1])
            
    return [color_0,color_1,color_2]

def is_perfect_lattice(sample, FM_bonds):
    is_perfect = True
    for qubits in FM_bonds:
       if is_frustrated(-1, sample[qubits[0]], sample[qubits[1]]):
           is_perfect = False
           break
    if is_perfect:
        return True
    else:
        return False

def print_sample(sample, embedding, colorings):
    qubit_index = np.zeros((embedding.shape[0], embedding.shape[1]), dtype=int)
    spins = np.zeros((embedding.shape[0], embedding.shape[1]))
    for i in range(embedding.shape[0]):
        for j in range(embedding.shape[1]):
            if j % 2 == 0:
                if i % 2 == 0:
                    qubit_index[i,j] = embedding[i,j]
                    if sample[embedding[i,j]] == -1:
                        print('-1', end=" ")
                        spins[i,j] = -1
                    else:
                        print(' 1', end=" ")
                        spins[i,j] = 1
                else:
                    print("  ",end=" ")
            else:
                if i % 2 == 1:
                    qubit_index[i,j] = embedding[i,j]
                    if sample[embedding[i,j]] == -1:
                        print('-1', end=" ")
                        spins[i,j] = -1
                    else:
                        print(' 1', end=" ")
                        spins[i,j] = 1
                else:
                    print("  ",end=" ")
        print()
    
    print()

    node_colors = {}
    node_sizes = {}
    triangular_qubits = np.zeros((int(embedding.shape[0] / 2), embedding.shape[1]), dtype=int)
    triangular_spins = np.zeros((int(embedding.shape[0] / 2), embedding.shape[1]), dtype=float)
    triangular_colors = np.zeros((int(embedding.shape[0] / 2), embedding.shape[1]), dtype=int)
    for i in range(int(embedding.shape[0] / 2)):
        for j in range(embedding.shape[1]):
            if j % 2 == 0:
                triangular_qubits[i,j] = qubit_index[i*2, j]
                triangular_spins[i,j] = spins[i*2, j]
                if spins[i*2, j] == -1:
                    node_colors[triangular_qubits[i,j]] = 'tab:blue'
                    node_sizes[triangular_qubits[i,j]] = 10
                else:
                    node_colors[triangular_qubits[i,j]] = 'tab:red'
                    node_sizes[triangular_qubits[i,j]] = 10

                if triangular_qubits[i,j] in colorings[0]:
                    triangular_colors[i,j] = 1
                elif triangular_qubits[i,j] in colorings[1]:
                    triangular_colors[i,j] = 2
                elif triangular_qubits[i,j] in colorings[2]:
                    triangular_colors[i,j] = 3
                else:
                    print("wrong colors")
                    exit(1)
            else:
                triangular_qubits[i,j] = qubit_index[i*2+1, j]
                triangular_spins[i,j] = spins[i*2+1, j]
                if spins[i*2+1, j] == -1:
                    node_colors[triangular_qubits[i,j]] = 'tab:blue'
                    node_sizes[triangular_qubits[i,j]] = 10
                else:
                    node_colors[triangular_qubits[i,j]] = 'tab:red'
                    node_sizes[triangular_qubits[i,j]] = 10

                if triangular_qubits[i,j] in colorings[0]:
                    triangular_colors[i,j] = 1
                elif triangular_qubits[i,j] in colorings[1]:
                    triangular_colors[i,j] = 2
                elif triangular_qubits[i,j] in colorings[2]:
                    triangular_colors[i,j] = 3
                else:
                    print("wrong colors")
                    exit(1)
    
    
    
    triangular_components = np.zeros((int(embedding.shape[0] / 2), embedding.shape[1]), dtype=np.complex64)
    for i in range(int(embedding.shape[0] / 2)):
        for j in range(embedding.shape[1]):
            if triangular_qubits[i,j] in colorings[0]:
                triangular_components[i,j] = triangular_spins[i,j]
            elif triangular_qubits[i,j] in colorings[1]:
                triangular_components[i,j] = triangular_spins[i,j] * np.exp(2 * np.pi * 1j / 3)
            elif triangular_qubits[i,j] in colorings[2]:
                triangular_components[i,j] = triangular_spins[i,j] * np.exp(4 * np.pi * 1j / 3)
            else:
                print("wrong coloring...")
                exit(1)

    edge_list = []
    node_positions = {}
    phases = np.zeros((embedding.shape[0] - 2, embedding.shape[1] - 1), dtype=float)
    for i in range(int(embedding.shape[0] / 2) - 1):
        for j in range(embedding.shape[1] - 1):
            if j % 2 == 0:
                qubit1 = triangular_components[i, j]
                qubit2 = triangular_components[i+1, j]
                qubit3 = triangular_components[i, j+1]
                if (triangular_qubits[i, j], triangular_qubits[i+1, j]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j], triangular_qubits[i+1, j]))

                if (triangular_qubits[i+1, j], triangular_qubits[i, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i+1, j], triangular_qubits[i, j+1]))

                if (triangular_qubits[i, j], triangular_qubits[i, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j], triangular_qubits[i, j+1]))
                
                node_positions[triangular_qubits[i, j]] = (j * np.sin(np.pi/3),-i)
                node_positions[triangular_qubits[i+1, j]] = (j * np.sin(np.pi/3), -i-1)
                node_positions[triangular_qubits[i, j+1]] = ((j+1) * np.sin(np.pi/3), -i - 0.5)
                node_positions[triangular_qubits[i+1, j+1]] = ((j+1) * np.sin(np.pi/3), -i - 1.5)

                phases[2*i, j] = np.angle(qubit1 + qubit2 + qubit3, deg=True)

                qubit4 = triangular_components[i+1, j]
                qubit5 = triangular_components[i, j+1]
                qubit6 = triangular_components[i+1, j+1]

                if (triangular_qubits[i+1, j], triangular_qubits[i, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i+1, j], triangular_qubits[i, j+1]))

                if (triangular_qubits[i+1, j], triangular_qubits[i+1, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i+1, j], triangular_qubits[i+1, j+1]))

                if (triangular_qubits[i, j+1], triangular_qubits[i+1, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j+1], triangular_qubits[i+1, j+1]))

                phases[2*i+1, j] = np.angle(qubit4 + qubit5 + qubit6, deg=True)
            else:
                qubit1 = triangular_components[i, j]
                qubit2 = triangular_components[i, j+1]
                qubit3 = triangular_components[i+1, j+1]
                phases[2*i, j] = np.angle(qubit1 + qubit2 + qubit3, deg=True)

                if (triangular_qubits[i, j], triangular_qubits[i, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j], triangular_qubits[i, j+1]))

                if (triangular_qubits[i, j+1], triangular_qubits[i+1, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j+1], triangular_qubits[i+1, j+1]))

                if (triangular_qubits[i, j], triangular_qubits[i+1, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j], triangular_qubits[i+1, j+1]))

                node_positions[triangular_qubits[i, j]] = (j * np.sin(np.pi/3),-i-0.5)
                node_positions[triangular_qubits[i+1, j]] = (j * np.sin(np.pi/3), -i-1.5)
                node_positions[triangular_qubits[i, j+1]] = ((j+1) * np.sin(np.pi/3), -i )
                node_positions[triangular_qubits[i+1, j+1]] = ((j+1) * np.sin(np.pi/3), -i - 1)

                qubit4 = triangular_components[i, j]
                qubit5 = triangular_components[i+1, j]
                qubit6 = triangular_components[i+1, j+1]

                if (triangular_qubits[i, j], triangular_qubits[i+1, j]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j], triangular_qubits[i+1, j]))

                if (triangular_qubits[i+1, j], triangular_qubits[i+1, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i+1, j], triangular_qubits[i+1, j+1]))

                if (triangular_qubits[i, j], triangular_qubits[i+1, j+1]) not in edge_list:
                    edge_list.append((triangular_qubits[i, j], triangular_qubits[i+1, j+1]))
                phases[2*i+1, j] = np.angle(qubit4 + qubit5 + qubit6, deg=True)
    # np.set_printoptions(threshold=np.inf, precision=1,linewidth=9999999)
    # print(phases)
    pyplot.clf()

    angles = {0: (1,0), 60:(np.cos( np.pi / 3), np.sin( np.pi / 3)), 
              120:(-1 * np.cos( np.pi / 3), np.sin( np.pi / 3)), 180:(-1 ,0),
              -60: (np.cos( np.pi / 3), -1 * np.sin( np.pi / 3)),
              -120:(-1 * np.cos(np.pi / 3), -1 * np.sin(np.pi / 3))}

    

    
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (12,12))
    

    quiver_list = []
    triangle_colors = []

    for i in range(int(embedding.shape[0] / 2) - 1):
        for j in range(embedding.shape[1] - 1):
            if j % 2 == 0:
                qubit1 = triangular_components[i, j]
                qubit2 = triangular_components[i+1, j]
                qubit3 = triangular_components[i, j+1]

                if triangular_spins[i,j] == 1 and triangular_spins[i+1, j] == 1 and triangular_spins[i, j+1] == 1:
                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i, j+1]][0]], 
                                        [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i, j+1]][1]],
                                        1])
                elif triangular_spins[i,j] == -1 and triangular_spins[i+1, j] == -1 and triangular_spins[i, j+1] == -1:
                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i, j+1]][0]], 
                                        [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i, j+1]][1]],
                                        -1])
                else:
                    triangle_center_x = (node_positions[triangular_qubits[i, j]][0] + node_positions[triangular_qubits[i+1, j]][0] + node_positions[triangular_qubits[i, j+1]][0]) / 3
                    triangle_center_y = (node_positions[triangular_qubits[i, j]][1] + node_positions[triangular_qubits[i+1, j]][1] + node_positions[triangular_qubits[i, j+1]][1]) / 3
                    quiver_list.append([triangle_center_x, triangle_center_y, angles[int(phases[2*i, j])][0], angles[int(phases[2*i, j])][1]])
                    # ax.quiver(triangle_center_x, triangle_center_y, angles[int(phases[2*i, j])][0], angles[int(phases[2*i, j])][1], units ='xy' ,scale=2.5, pivot='mid', width = 0.02)

                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i, j+1]][0]], 
                                            [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i, j+1]][1]],
                                            int(phases[2*i, j])])
                    

                qubit4 = triangular_components[i+1, j]
                qubit5 = triangular_components[i, j+1]
                qubit6 = triangular_components[i+1, j+1]

                if triangular_spins[i+1,j] == 1 and triangular_spins[i, j+1] == 1 and triangular_spins[i+1, j+1] == 1:
                    triangle_colors.append([[node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i, j+1]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                        [node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i, j+1]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                        1])
                elif triangular_spins[i+1,j] == -1 and triangular_spins[i, j+1] == -1 and triangular_spins[i+1, j+1] == -1:
                    triangle_colors.append([[node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i, j+1]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                        [node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i, j+1]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                        -1])
                else:
                    triangle_center_x = (node_positions[triangular_qubits[i+1, j]][0] + node_positions[triangular_qubits[i, j+1]][0] + node_positions[triangular_qubits[i+1, j+1]][0]) / 3
                    triangle_center_y = (node_positions[triangular_qubits[i+1, j]][1] + node_positions[triangular_qubits[i, j+1]][1] + node_positions[triangular_qubits[i+1, j+1]][1]) / 3
                    quiver_list.append([triangle_center_x, triangle_center_y, angles[int(phases[2*i+1, j])][0], angles[int(phases[2*i+1, j])][1]])
                    # ax.quiver(triangle_center_x, triangle_center_y, angles[int(phases[2*i+1, j])][0], angles[int(phases[2*i+1, j])][1], units ='xy' ,scale=2.5, pivot='mid', width = 0.02)

                    triangle_colors.append([[node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i, j+1]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                        [node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i, j+1]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                        int(phases[2*i+1, j])])

            else:
                qubit1 = triangular_components[i, j]
                qubit2 = triangular_components[i, j+1]
                qubit3 = triangular_components[i+1, j+1]

                if triangular_spins[i,j] == 1 and triangular_spins[i, j+1] == 1 and triangular_spins[i+1, j+1] == 1:
                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i, j+1]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                        [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i, j+1]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                        1])
                elif triangular_spins[i,j] == -1 and triangular_spins[i, j+1] == -1 and triangular_spins[i+1, j+1] == -1:
                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i, j+1]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                        [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i, j+1]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                        -1])
                else:
                    triangle_center_x = (node_positions[triangular_qubits[i, j]][0] + node_positions[triangular_qubits[i, j+1]][0] + node_positions[triangular_qubits[i+1, j+1]][0]) / 3
                    triangle_center_y = (node_positions[triangular_qubits[i, j]][1] + node_positions[triangular_qubits[i, j+1]][1] + node_positions[triangular_qubits[i+1, j+1]][1]) / 3
                    quiver_list.append([triangle_center_x, triangle_center_y, angles[int(phases[2*i, j])][0], angles[int(phases[2*i, j])][1]])
                    # ax.quiver(triangle_center_x, triangle_center_y, angles[int(phases[2*i, j])][0], angles[int(phases[2*i, j])][1], units ='xy' ,scale=2.5, pivot='mid', width = 0.02)

                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i, j+1]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                            [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i, j+1]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                            int(phases[2*i, j])])

                qubit4 = triangular_components[i, j]
                qubit5 = triangular_components[i+1, j]
                qubit6 = triangular_components[i+1, j+1]

                if triangular_spins[i,j] == 1 and triangular_spins[i+1, j] == 1 and triangular_spins[i+1, j+1] == 1:
                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                        [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                        1])
                elif triangular_spins[i,j] == -1 and triangular_spins[i+1, j] == -1 and triangular_spins[i+1, j+1] == -1:
                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                        [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                        -1])
                else:

                    triangle_center_x = (node_positions[triangular_qubits[i, j]][0] + node_positions[triangular_qubits[i+1, j]][0] + node_positions[triangular_qubits[i+1, j+1]][0]) / 3
                    triangle_center_y = (node_positions[triangular_qubits[i, j]][1] + node_positions[triangular_qubits[i+1, j]][1] + node_positions[triangular_qubits[i+1, j+1]][1]) / 3
                    quiver_list.append([triangle_center_x, triangle_center_y, angles[int(phases[2*i+1, j])][0], angles[int(phases[2*i+1, j])][1]])
                    # ax.quiver(triangle_center_x, triangle_center_y, angles[int(phases[2*i+1, j])][0], angles[int(phases[2*i+1, j])][1], units ='xy' ,scale=2.5, pivot='mid', width = 0.02)

                    triangle_colors.append([[node_positions[triangular_qubits[i, j]][0], node_positions[triangular_qubits[i+1, j]][0], node_positions[triangular_qubits[i+1, j+1]][0]], 
                                            [node_positions[triangular_qubits[i, j]][1], node_positions[triangular_qubits[i+1, j]][1], node_positions[triangular_qubits[i+1, j+1]][1]],
                                            int(phases[2*i+1, j])])


    # print(edge_list)
    # print(node_positions)
    # print(node_colors)
    # print(quiver_list)
    # print(triangle_colors)
    
    angle_to_color = {0:'rosybrown', 60:'peru', 120:'wheat', 180:'lightcyan', -60: 'thistle', -120:'pink', 1: 'white', -1: 'black'}
    for tri in triangle_colors:
        ax.fill(tri[0], tri[1], angle_to_color[tri[2]])
    Graph(edge_list, node_layout = node_positions, node_color = node_colors, node_size = 6.0, ax = ax)
    for q in quiver_list:
        ax.quiver(q[0], q[1], q[2], q[3], units ='xy' ,scale=2.5, pivot='mid', width = 0.02)

    pyplot.savefig('test.pdf')
    exit()

    



def count_perfect_lattices(samples, num_embedding, embeddings, FM_bonds):
    total_count = 0
    perfect_lattice_count = 0
    for ite in range(num_embedding):
        for sample in samples:
            if is_perfect_lattice(sample, FM_bonds[ite]):
                # colorings = get_coloring(embeddings[ite], 0)
                # print_sample(sample, embeddings[ite], colorings)
                perfect_lattice_count += 1
            total_count += 1
    return total_count, perfect_lattice_count
            

                    
def get_order_parameter_triangle(samples, num_embedding, embedding, k_away_from_boundry_omitted, FM_bonds, only_count_perfect_samples = False): 
    order_parameters = []
    real_psi = []
    imag_psi = []

    order_parameter_batches = []
    current_batch = []
    
    for ite in range(num_embedding):
        # counted_samples = 0
        # psi = 0
        coloring = get_coloring(embedding[ite], k_away_from_boundry_omitted)
        for sample in samples:
            if only_count_perfect_samples:
                if not is_perfect_lattice(sample, FM_bonds[ite]):
                    continue
            m0_sum = 0
            m1_sum = 0
            m2_sum = 0
            for qubit in coloring[0]:
                m0_sum += sample[qubit]
            for qubit in coloring[1]:
                m1_sum += sample[qubit]
            for qubit in coloring[2]:
                m2_sum += sample[qubit]
            m0_avg = m0_sum / len(coloring[0])
            m1_avg = m1_sum / len(coloring[1])
            m2_avg = m2_sum / len(coloring[2])
            full_psi = (m0_avg + m1_avg * np.exp(2 * np.pi * 1j / 3) + m2_avg * np.exp(4 * np.pi * 1j / 3)) / np.sqrt(3)
            real_psi.append(np.real(full_psi))
            imag_psi.append(np.imag(full_psi))
            psi = np.abs((m0_avg + m1_avg * np.exp(2 * np.pi * 1j / 3) + m2_avg * np.exp(4 * np.pi * 1j / 3)) / np.sqrt(3))
            # counted_samples += 1
        # psi /= counted_samples
            order_parameters.append(psi)
            current_batch.append(psi)
            if len(current_batch) == 100 * embedding.shape[0]:
                order_parameter_batches.append(mean(current_batch))
                current_batch = []
    
    if current_batch:
        order_parameter_batches.append(mean(current_batch))

    return [mean(order_parameters), sem(order_parameters), sem(order_parameter_batches), np.std(order_parameters), np.std(order_parameter_batches)],  real_psi, imag_psi

def is_frustrated(coupler_strength, i, j):
    # coupler_strength = -0.5
    return (sign_of(coupler_strength) * i * j +1) * 0.5

def get_fourier_transform(samples, anneal_time_ns, working_folder, dim_x, dim_y, num_embedding, embedding, num_steps, samples_used_for_fourier = 1000, ignore_boundary = 0):
    pair_count = {}
    pairs = {}

    def get_distance(x_1,y_1,x_2,y_2):
        coord_1 = [0, y_1]
        coord_2 = [0, y_2]
        if y_1 % 2 == 0:
            if x_1 % 2 == 0:
                coord_1[0] = (x_1+1) / 2
            else:
                coord_1[0] = x_1 / 2
        else:
            if x_1 % 2 == 0:
                coord_1[0] = x_1 / 2
            else:
                if x_1 == dim_x - 1:
                    coord_1[0] = 0
                else:
                    coord_1[0] = (x_1+1)/2

        if y_2 % 2 == 0:
            if x_2 % 2 == 0:
                coord_2[0] = (x_2+1) / 2 
            else:
                coord_2[0] = x_2 / 2
        else:
            if x_2 % 2 == 0:
                coord_2[0] = x_2 / 2
            else:
                if x_2 == dim_x - 1:
                    coord_2[0] = 0
                else:
                    coord_2[0] = (x_2+1)/2
        dist_x = coord_2[0] - coord_1[0]
        if dist_x > dim_x / 4:
            dist_x -= dim_x / 2
        elif dist_x < dim_x / -4:
            dist_x += dim_x / 2

        return dist_x, coord_2[1] - coord_1[1]
    
    embedding_used = 0
    sample_array = np.empty((samples_used_for_fourier, dim_x * dim_y))
    sample_counter = 0
    for k in range(samples_used_for_fourier):
        sample = samples[sample_counter]
        spin_counter = 0
        for i in range(dim_x):
            for j in range(dim_y):
                sample_array[k,spin_counter] = sample[embedding[embedding_used, i,j]]
                spin_counter += 1
        if embedding_used == embedding.shape[0] - 1:
            embedding_used = 0
            sample_counter += 1
        else:
            embedding_used += 1

    corrmat = np.matmul(sample_array.T, sample_array) / sample_array.shape[0]

    for x_1, y_1, x_2, y_2 in itertools.product(range(dim_x), range(dim_y), range(dim_x), range(dim_y)):
        x_dist, y_dist = get_distance(x_1, y_1, x_2, y_2)
        spin1_coord_flat = x_1 * dim_x + y_1
        spin2_coord_flat = x_2 * dim_x + y_2
        if (x_dist, y_dist) not in pairs:
            pairs[(x_dist, y_dist)] = corrmat[spin1_coord_flat, spin2_coord_flat]
            pair_count[(x_dist, y_dist)] = 1
        else:
            pairs[(x_dist, y_dist)] += corrmat[spin1_coord_flat, spin2_coord_flat]
            pair_count[(x_dist, y_dist)] += 1
                            
    for key in pairs:
        pairs[key] /= pair_count[key]
    
    keylist = list(pairs.keys())
    key_array = np.empty((len(pairs.keys()), 2), dtype=np.csingle)

    for i in range(len(pairs.keys())):
        key_array[i, 0] = 1j * np.pi * keylist[i][0]
        key_array[i, 1] = 1j * np.pi * np.sin(np.pi / 3) * keylist[i][1]
    value_array = np.asarray(list(pairs.values()))
            

    fourier = np.zeros((num_steps,num_steps))
    labels = np.linspace(-2,2,num_steps)

    pre_computed_fourier_coefficients = np.zeros((len(labels), len(labels), len(keylist)), dtype=np.csingle)

    for xind, yind in itertools.product(range(len(labels)), range(len(labels))):
        kx = labels[xind]
        ky = labels[yind]
        pre_computed_fourier_coefficients[xind, yind] = np.exp(np.matmul(key_array, np.asarray([kx, ky])))



    for xind, yind in itertools.product(range(len(labels)), range(len(labels))):
        kx = labels[xind]
        ky = labels[yind]
        f = np.dot(pre_computed_fourier_coefficients[xind, yind], value_array)
        fourier[xind, yind] = np.abs(f)

    smax = np.max(fourier)
    smax_coord = np.unravel_index(fourier.argmax(), fourier.shape)
    smax_coord = [labels[smax_coord[0]], labels[smax_coord[1]]]

    # def fitting_function_x(x, gamma , global_scal):
    #     ret = global_scal / np.pi * (gamma / (( x * np.pi - smax_coord[0] * np.pi) ** 2 + gamma ** 2))
    #     return ret
    
    # def fitting_function_y(x, gamma , global_scal):
    #     ret = global_scal / np.pi * (gamma / (( x * np.pi - smax_coord[1] * np.pi) ** 2 + gamma ** 2))
    #     return ret

    def fitting_function_x(x, gamma, eta, I):
        sigma = (1/gamma) / (2 * np.sqrt(2 * np.log(2)))
        G = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x * np.pi - smax_coord[0] * np.pi)**2 / (2 * sigma ** 2))
        L = 1 / (np.pi) * ((1/gamma) / 2) / ((x * np.pi - smax_coord[0] * np.pi)**2 + ((1/gamma) / 2) ** 2)
        ret = I * (eta * G + (1-eta) * L)
        return ret
    
    def fitting_function_y(x, gamma, eta, I):
        sigma = (1/gamma) / (2 * np.sqrt(2 * np.log(2)))
        G = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x * np.pi - smax_coord[1] * np.pi)**2 / (2 * sigma ** 2))
        L = 1 / (np.pi) * ((1/gamma) / 2) / ((x * np.pi - smax_coord[1] * np.pi)**2 + ((1/gamma) / 2) ** 2)
        ret = I * (eta * G + (1-eta) * L)
        return ret
    
    if anneal_time_ns < 100:
        x_values_for_fitting = np.linspace(smax_coord[0] - 1/4, smax_coord[0] + 1/4, 10)
        x_values_for_plotting = np.linspace(smax_coord[0] - 1/4, smax_coord[0] + 1/4, 100)
    else:
        x_values_for_fitting = np.linspace(smax_coord[0] - 1/8, smax_coord[0] + 1/8, 10)
        x_values_for_plotting = np.linspace(smax_coord[0] - 1/8, smax_coord[0] + 1/8, 100)
    y_values_for_fitting = []
    y_values_for_plotting = []
    fitted_y_values_for_plotting = []

    for x_val in x_values_for_fitting:
        y_val = smax_coord[1]
        f = 0
        for key in pairs:
            f += pairs[key] * np.exp(1j*np.pi*(x_val*key[0] + y_val*key[1]*np.sin(np.pi / 3)))
        y_values_for_fitting.append(np.abs(f))

    
    for x_val in x_values_for_plotting:
        y_val = smax_coord[1]
        f = 0
        for key in pairs:
            f += pairs[key] * np.exp(1j*np.pi*(x_val*key[0] + y_val*key[1]*np.sin(np.pi / 3)))
        y_values_for_plotting.append(np.abs(f))

    
    model = lmfit.Model(fitting_function_x)
    params = model.make_params(gamma= 1, eta=dict(value=0.5, min=0, max=1), I= 22)
    result = model.fit(y_values_for_fitting, params, x=x_values_for_fitting)
    eta = result.best_values['eta']
    eta_x = eta
    gamma = result.best_values['gamma']
    I = result.best_values['I']




    ci = result.conf_interval(p_names=['gamma'], sigmas=[1,2])

    sigma_x_1_negative = ci['gamma'][0][1] - gamma
    sigma_x_2_negative = ci['gamma'][1][1] - gamma

    sigma_x_1_positive = ci['gamma'][4][1] - gamma
    sigma_x_2_positive = ci['gamma'][3][1] - gamma




    # popt, pcov = curve_fit(fitting_function_x, x_values_for_fitting, y_values_for_fitting, bounds=((-np.inf, 0, -np.inf), (np.inf, 1, np.inf)), maxfev=1600, p0=[2, 0.5, 1])
    # gamma = popt[0]
    # eta = popt[1]
    # eta_x = eta
    # I = popt[2]

    corr_length_x = gamma

    for x_val in x_values_for_plotting:
        fitted_y_values_for_plotting.append(fitting_function_x(x_val, gamma, eta, I))
    
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Triangular Lattice @ {round(anneal_time_ns,1)}ns - Periodic Dir", fontsize=20)
    ax.plot(x_values_for_plotting, y_values_for_plotting, label = 'Experiment', color='blue')
    ax.plot(x_values_for_plotting, fitted_y_values_for_plotting, label = f'Fitted ({round(gamma,4)})', color='red')
    ax.set_xlabel("X", fontsize=24)
    ax.set_ylabel("Y", fontsize=24)
    ax.tick_params(labelsize=20)
    ax.legend(loc='upper right', fontsize = 20)
    pyplot.tight_layout() 
    pyplot.savefig(working_folder +"/correlation_length_x_" + str(anneal_time_ns) + "ns_"+str(num_calib_ite) + "ite.pdf")
    pyplot.clf()

    if anneal_time_ns < 100:
        x_values_for_fitting = np.linspace(smax_coord[1] - 1/4, smax_coord[1] + 1/4, 10)
        x_values_for_plotting = np.linspace(smax_coord[1] - 1/4, smax_coord[1] + 1/4, 100)
    else:
        x_values_for_fitting = np.linspace(smax_coord[1] - 1/8, smax_coord[1] + 1/8, 10)
        x_values_for_plotting = np.linspace(smax_coord[1] - 1/8, smax_coord[1] + 1/8, 100)
    y_values_for_fitting = []
    y_values_for_plotting = []
    fitted_y_values_for_plotting = []

    for y_val in x_values_for_fitting:
        x_val = smax_coord[0]
        f = 0
        for key in pairs:
            f += pairs[key] * np.exp(1j*np.pi*(x_val*key[0] + y_val*key[1]*np.sin(np.pi / 3)))
        y_values_for_fitting.append(np.abs(f))

    
    for y_val in x_values_for_plotting:
        x_val = smax_coord[0]
        f = 0
        for key in pairs:
            f += pairs[key] * np.exp(1j*np.pi*(x_val*key[0] + y_val*key[1]*np.sin(np.pi / 3)))
        y_values_for_plotting.append(np.abs(f))


    model = lmfit.Model(fitting_function_y)
    params = model.make_params(gamma= 1, eta=dict(value=0.5, min=0, max=1), I= 22)
    result = model.fit(y_values_for_fitting, params, x=x_values_for_fitting)
    eta = result.best_values['eta']
    eta_y = eta
    gamma = result.best_values['gamma']
    I = result.best_values['I']

    ci = result.conf_interval(p_names=['gamma'], sigmas=[1,2])

    corr_length_y = gamma

    sigma_y_1_negative = ci['gamma'][0][1] - gamma
    sigma_y_2_negative = ci['gamma'][1][1] - gamma

    sigma_y_1_positive = ci['gamma'][4][1] - gamma
    sigma_y_2_positive = ci['gamma'][3][1] - gamma

    # popt, pcov = curve_fit(fitting_function_y, x_values_for_fitting, y_values_for_fitting, bounds=((-np.inf, 0, -np.inf), (np.inf, 1, np.inf)), maxfev=1600, p0=[2, 0.5, 1])
    # gamma = popt[0]
    # eta = popt[1]
    # eta_y = eta
    # I = popt[2]

    # corr_length_y = 1 / popt[0]

    # sigma_y = np.sqrt(np.diagonal(pcov))

    for x_val in x_values_for_plotting:
        fitted_y_values_for_plotting.append(fitting_function_y(x_val, gamma, eta, I))
    
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Triangular Lattice @ {round(anneal_time_ns,1)}ns - NonPeriodic Dir", fontsize=20)
    ax.plot(x_values_for_plotting, y_values_for_plotting, label = 'Experiment', color='blue')
    ax.plot(x_values_for_plotting, fitted_y_values_for_plotting, label = f'Fitted ({round(gamma,4)})', color='red')
    ax.set_xlabel("X", fontsize=24)
    ax.set_ylabel("Y", fontsize=24)
    ax.tick_params(labelsize=20)
    ax.legend(loc='upper right', fontsize = 20)
    pyplot.tight_layout() 
    pyplot.savefig(working_folder +"/correlation_length_y_" + str(anneal_time_ns) + "ns_"+str(num_calib_ite) + "ite.pdf")
    pyplot.clf()

    annealing_time_dict = {'3.6':6.32998, '4.0': 7.10992, '5.3': 7.82315, '6.5': 8.56292,
                            '8.0': 9.69824, '9.5': 10.92294, '11.0': 12.15743, '12.0': 13.0168,
                            '14.0': 14.6431, '16.0': 16.24395, '18.0': 18.10289, '20.0': 20}
    
    
    if anneal_time_ns <= 20:
        print(f'{annealing_time_dict[str(anneal_time_ns)]} ns, eta x: {eta_x}, eta y: {eta_y}')
        print(f'{annealing_time_dict[str(anneal_time_ns)]} ns, x corr: {corr_length_x}, y corr: {corr_length_y}')
    
    else:
        print(f'{anneal_time_ns} ns, eta x: {eta_x}, eta y: {eta_y}')
        print(f'{anneal_time_ns} ns, x corr: {corr_length_x}, y corr: {corr_length_y}')







    
    # sxshift = 0
    # syshift = 0

    # f = 0
    # for key in pairs:
    #     f += pairs[key] * np.exp(1j*np.pi*((smax_coord[0] + 2*np.pi / dim_x)*key[0] + smax_coord[1]*key[1]*np.sin(np.pi / 3)))
    # sxshift = np.abs(f)

    # f = 0
    # for key in pairs:
    #     f += pairs[key] * np.exp(1j*np.pi*(smax_coord[0]*key[0] + (smax_coord[1] + 2*np.pi / dim_y)*key[1]*np.sin(np.pi / 3)))
    # syshift = np.abs(f)

    # xcorr_x = dim_x / (2*np.pi) * np.sqrt((smax - sxshift) - 1)
    # xcorr_y = dim_y / (2*np.pi) * np.sqrt((smax - syshift) - 1)

    # print(f'xcorr_x: {xcorr_x}, xcorr_y: {xcorr_y}')

    return fourier, list(np.round(np.linspace(-2,2,num_steps),2)), corr_length_x, corr_length_y, sigma_x_1_negative, sigma_x_2_negative, sigma_x_1_positive, sigma_x_2_positive, sigma_y_1_negative, sigma_y_2_negative, sigma_y_1_positive, sigma_y_2_positive

def gauge_transform(sample, dim_x, dim_y, embedding):
    transformed_sample = sample.copy()
    # for i in range(dim_x):
    #     for j in range(dim_y):
    #         if i % 4 == 0 or i % 4 == 1:
    #             transformed_sample[embedding[i,j]] *= -1
    row_group1 = []
    for i in range(dim_x):
        if i % 4 == 0 or i % 4 == 1:
            row_group1.append(i)

    for i in range(dim_x):
        for j in range(dim_y):
            if i in row_group1:
                if j % 2 == 0:
                    transformed_sample[embedding[i,j]] *= -1
            else:
                if j % 2 == 1:
                    transformed_sample[embedding[i,j]] *= -1
    return transformed_sample

def get_order_parameter_villain_sandvik(samples, k_away_from_boundry_omitted, dim_x, dim_y, num_embedding, embedding):
    order_parameters = []
    real_psi = []
    imag_psi = []
    m0 = 2

    order_parameter_batches = []
    current_batch = []

    for ite in range(num_embedding):
        # psi = 0
        for sample in samples:
            m1 = 0
            m1_count = 0
            m2 = 0
            m2_count = 0
            m3 = 0
            m3_count = 0
            m4 = 0
            m4_count = 0

            sample = gauge_transform(sample, dim_x, dim_y, embedding[ite, :, :])

            for i in range(dim_x):
                curr_row = embedding[ite ,i, :]
                for m in range(len(curr_row)):
                    if m < k_away_from_boundry_omitted or dim_y-m <= k_away_from_boundry_omitted:
                        continue
                    if i % 2 == 0 and m % 2 == 0:
                        m2 += sample[curr_row[m]]
                        m2_count += 1
                    if i % 2 == 0 and m % 2 == 1:
                        m1 += sample[curr_row[m]]
                        m1_count += 1
                    if i % 2 == 1 and m % 2 == 0:
                        m3 += sample[curr_row[m]]
                        m3_count += 1
                    if i % 2 == 1 and m % 2 == 1:
                        m4 += sample[curr_row[m]]
                        m4_count += 1
            m1 /= m1_count
            m2 /= m2_count
            m3 /= m3_count
            m4 /= m4_count
            real_psi.append(((m1-m4)*np.cos(np.pi/8) + (m2-m3)*np.sin(np.pi/8))/m0)
            imag_psi.append(((m1+m4)*np.sin(np.pi/8) + (m2+m3)*np.cos(np.pi/8))/m0 )
            psi = np.abs(((m1-m4)*np.cos(np.pi/8) + (m2-m3)*np.sin(np.pi/8))/m0 + ((m1+m4)*np.sin(np.pi/8) + (m2+m3)*np.cos(np.pi/8))/m0 * 1j)
        # psi /= len(samples)
            order_parameters.append(psi)

            current_batch.append(psi)
            if len(current_batch) == 100 * embedding.shape[0]:
                order_parameter_batches.append(mean(current_batch))
                current_batch = []
    if current_batch:
        order_parameter_batches.append(mean(current_batch))

    return [mean(order_parameters), sem(order_parameters), sem(order_parameter_batches), np.std(order_parameters), np.std(order_parameter_batches)], real_psi, imag_psi

def get_order_parameter_villain(samples, k_away_from_boundry_omitted, dim_x, dim_y, num_embedding, embedding, couplers):
    order_parameters = []
    real_psi = []
    imag_psi = []
    for ite in range(num_embedding):
        psi = 0
        for sample in samples:
            even_row = 0
            even_row_num = 0
            even_col = 0
            even_col_num = 0
            odd_row = 0
            odd_row_num = 0
            odd_col = 0
            odd_col_num = 0

            for i in range(dim_x):
                curr_row = embedding[ite, i, :]
                for m in range(len(curr_row) - 1):
                    if m < k_away_from_boundry_omitted or dim_y-m <= k_away_from_boundry_omitted:
                        continue
                    if i % 2 == 0:
                        even_row += is_frustrated(couplers[(curr_row[m], curr_row[m+1])], sample[curr_row[m]], sample[curr_row[m+1]])
                        even_row_num += 1
                    else:
                        odd_row += is_frustrated(couplers[(curr_row[m], curr_row[m+1])], sample[curr_row[m]], sample[curr_row[m+1]])
                        odd_row_num += 1

            for i in range(dim_y):
                if i < k_away_from_boundry_omitted or dim_y - i <= k_away_from_boundry_omitted:
                    continue
                curr_col = list(embedding[ite, :, i])
                curr_col.append(curr_col[0])
                for m in range(len(curr_col)-1):
                    if i % 2 == 0:
                        even_col += is_frustrated(couplers[(curr_col[m], curr_col[m+1])], sample[curr_col[m]], sample[curr_col[m+1]])
                        even_col_num += 1
                    else:
                        odd_col += is_frustrated(couplers[(curr_col[m], curr_col[m+1])], sample[curr_col[m]], sample[curr_col[m+1]])
                        odd_col_num += 1
            
            
            even_row /= even_row_num
            even_col /= even_col_num
            odd_row /= odd_row_num
            odd_col /= odd_col_num
            psi += np.abs(even_row + even_col * 1j - odd_row - odd_col * 1j)
            real_psi.append(even_row - odd_row)
            imag_psi.append(even_col - odd_col)
        psi /= len(samples)
        order_parameters.append(psi)
    return order_parameters, real_psi, imag_psi

def normaliza_couplers(couplers, coupler_strength,positive_couplers):
    positive_sum = 0
    scaled_couplers = {}
    for coupler in couplers:
        if couplers[coupler] > 0:
            positive_sum += couplers[coupler]
        else:
            scaled_couplers[coupler] = couplers[coupler]
    
    positive_mean = positive_sum / len(positive_couplers)

    for coupler in positive_couplers:
        scaled_couplers[coupler] = couplers[coupler] / (positive_mean / coupler_strength)
        if scaled_couplers[coupler] > 1:
            scaled_couplers[coupler] = 1
    
    return scaled_couplers    

def count_ground_vortice(sample, couplers_per_embedding):
    total_frustrated_couplers = 0
    for couplers in couplers_per_embedding:
        for coupler in couplers:
            if is_frustrated(couplers[coupler], sample[coupler[0]], sample[coupler[1]]):
                total_frustrated_couplers += 1
    return total_frustrated_couplers / len(couplers_per_embedding)

    
def compute_average_residual_energy(samples, couplers_per_embedding, ground_energy):
    residual_energy = []
    for couplers in couplers_per_embedding:
        bqm = dimod.binary.BinaryQuadraticModel.from_ising(h={}, J=couplers)
        for sample in samples:
                curr_energy = bqm.energy(sample)
                residual_energy.append(curr_energy - ground_energy)
    return mean(residual_energy), sem(residual_energy)

def compute_averge_vortices(samples, couplers_per_embedding, ground_vortice):
    num_vortice = []

    num_vortex_batch = []
    current_batch = []

    for sample in samples:
        for couplers in couplers_per_embedding:
            num_fustrated_couplers = 0
            for coupler in couplers:
                if is_frustrated(couplers[coupler], sample[coupler[0]], sample[coupler[1]]):
                    num_fustrated_couplers += 1
            num_vortice.append(num_fustrated_couplers - ground_vortice)

            current_batch.append(num_fustrated_couplers - ground_vortice)
            if len(current_batch) == 100 * len(couplers_per_embedding):
                num_vortex_batch.append(mean(current_batch))
                current_batch = []
    
    if current_batch:
        num_vortex_batch.append(mean(current_batch))


    print(f'There are {len(num_vortex_batch)} batches')
    return mean(num_vortice), sem(num_vortice), sem(num_vortex_batch), np.std(num_vortice), np.std(num_vortex_batch)



def qpu_run(qubit_in_use, couplers, coupler_strength, num_calib_ite, dim_x, dim_y, FM_bonds, 
            couplers_per_embedding, num_embedding, embedding,  flux_step_size, corr_step_size, anneal_offset_step_size, coupler_damp, 
            anneal_offset_damp, num_reads_per_call, anneal_time_ns, corr_buckets, k_away_from_boundry_omitted,samples_used_for_analysis):
    #Collect the qpu stats without any calibration
    solver = DWaveSampler(solver='Advantage_system4.1')
    working_folder = "./triangle_" + str(dim_x) + "_" + str(dim_y) + "dim/" + str(anneal_time_ns) + "ns"
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)

    anneal_time = anneal_time_ns / 1000
    original_coupler_mag = couplers.copy()
    positive_couplers = []

    for c in original_coupler_mag:
        if original_coupler_mag[c] > 0.6:
            positive_couplers.append(c)
    

    bqm = dimod.binary.BinaryQuadraticModel.from_ising(h={}, J=couplers)
    
    # fb = [0] * solver.properties['num_qubits']
    # anneal_offset = [0] * solver.properties['num_qubits']
    # for var in range(solver.properties['num_qubits']):
    #     fb[var] = 0

    fb = [0] * 5760
    anneal_offset = [0] * 5760
    for var in range(5760):
        fb[var] = 0

    sampleset_collection = []

    # offset_file = open("./input/offsets/" + str(anneal_time_ns) + 'ns.txt')
    list_of_anneal_offset = {}
    # offset_line = offset_file.readline()
    # tokens = offset_line.split()
    # initial_offsets = np.zeros(8)
    for i in range(8):
    #     initial_offsets[i] = float(tokens[i])
        list_of_anneal_offset[i] = []


    anneal_line_qubits, anneal_line_couplers = get_anneal_lines(qubit_in_use, couplers)
    # for line in range(8):
    #     for qubit in anneal_line_qubits[line]:
    #         anneal_offset[qubit] = initial_offsets[line]

    
    if dim_x==36 and dim_y==36:
        ground_state_energy = -1827
        ground_state_vortices = 636
    elif dim_x==24 and dim_y==24:
        ground_state_energy = -800.4
        ground_state_vortices = 280
    elif dim_x==12 and dim_y==12:
        ground_state_energy = -191.4
        ground_state_vortices = 68
    elif dim_x==6 and dim_y==6:
        ground_state_energy = -43.5
        ground_state_vortices = 16
    else:
        sa_sampler = neal.SimulatedAnnealingSampler()
        sa_sampleset = sa_sampler.sample(bqm, num_reads=10000)
        ground_state_sample = sa_sampleset.first.sample
        ground_state_energy = sa_sampleset.first.energy / num_embedding
        ground_state_vortices = count_ground_vortice(ground_state_sample, couplers_per_embedding)




    if not exists(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz"):
        for ite in tqdm(range(100)):
            sampleset = solver.sample(bqm, num_reads = num_reads_per_call, answer_mode='raw', auto_scale=False, fast_anneal=True,
                        annealing_time = anneal_time, flux_drift_compensation = False, flux_biases=fb)#, anneal_offsets = anneal_offset )
            sampleset_collection.append(sampleset)
        pickle.dump(sampleset_collection, lzma.open(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz", 'wb'))
        # sampleset_collection_file_write = lzma.open(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz", 'wb')
        # sampleset_collection_file_write.write(sampleset_collection)
    else:
        sampleset_collection = pickle.load(lzma.open(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz", 'rb'))
        # sampleset_collection_file = lzma.open(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz", 'rb')
        # sampleset_collection = sampleset_collection_file.read()
    

    sample_dicts = get_sol_dict_from_dwave_output(sampleset_collection)
    average_qubit_mag = get_average_mag_per_qubit(qubit_in_use, sample_dicts)
    average_coupler_corr = get_coupler_correlation(couplers, sample_dicts)
    average_coupler_corr_per_bucket, before_bucket_vals = get_average_bucket_corr(corr_buckets, average_coupler_corr)
    average_line_frust = get_average_line_frustration(anneal_line_couplers, average_coupler_corr)

    pre_running_list = []
    for s in sample_dicts:
        pre_running_list.append(s)

    pre_shimmed_triangular_psi, pre_shimmed_real_psi_tri, pre_shimmed_imag_psi_tri = get_order_parameter_triangle(pre_running_list, num_embedding, embedding, k_away_from_boundry_omitted, FM_bonds, only_count_perfect_samples = False)

    print("Frustration before shimming:", sum(list(average_coupler_corr.values()))/len(couplers))

    # Plotting
    pyplot.figure(1)
    mag_bins = np.linspace(-1, 1, 100)
    corr_bins = np.linspace(0, 0.5, 100)
    fig, ax = pyplot.subplots(nrows=1, ncols=1,figsize=(10, 8))

    ax.hist(average_qubit_mag.values(), mag_bins, alpha=0.5, label='w/o shimming')
    ax.set_xlabel("<M>", fontsize=24)
    ax.set_ylabel("Qubit Count", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)


    # ax[1].plot(average_line_frust, color = 'blue' ,label = 'without shimming')
    initial_average_line_frust = average_line_frust.copy()
    # ax[1].set_title("Anneal Line Frustration", fontsize=40)


    list_of_flux_bias = {}
    for i in range(len(qubit_in_use)):
        list_of_flux_bias[qubit_in_use[i]] = [0]
    
    list_of_coupler_strength = {}
    for coupler in couplers:
        list_of_coupler_strength[coupler] = [couplers[coupler]]
    


    if exists(working_folder + "/anneal_offset_" + str(anneal_time_ns) + "ns.xz") and exists(working_folder +"/coupler_strength_" + str(anneal_time_ns) + "ns.xz") and exists(working_folder +"/flux_bias_" + str(anneal_time_ns) + "ns.xz"):
        print("Previous run detected, loading data...")
        list_of_flux_bias = pickle.load(lzma.open(working_folder +"/flux_bias_" + str(anneal_time_ns) + "ns.xz", 'rb'))
        list_of_coupler_strength = pickle.load(lzma.open(working_folder +"/coupler_strength_" + str(anneal_time_ns) + "ns.xz", 'rb'))
        list_of_anneal_offset = pickle.load(lzma.open(working_folder +"/anneal_offset_" + str(anneal_time_ns) + "ns.xz", 'rb'))

        # list_of_flux_bias_file = lzma.open(working_folder +"/flux_bias_" + str(anneal_time_ns) + "ns.xz", 'rb')
        # list_of_coupler_strength_file = lzma.open(working_folder +"/coupler_strength_" + str(anneal_time_ns) + "ns.xz", 'rb')
        # list_of_anneal_offset_file = lzma.open(working_folder +"/anneal_offset_" + str(anneal_time_ns) + "ns.xz", 'rb')
        # list_of_flux_bias = list_of_flux_bias_file.read()
        # list_of_coupler_strength = list_of_coupler_strength_file.read()
        # list_of_anneal_offset = list_of_anneal_offset_file.read()

        starting_ite = len(list_of_flux_bias[qubit_in_use[0]]) - 1

        # for q in qubit_in_use:
        #     fb[q] = list_of_flux_bias[q][-1]
        # for coupler in couplers:
        #     couplers[coupler] = list_of_coupler_strength[coupler][-1]
        # for line in range(8):
        #     for qubit in anneal_line_qubits[line]:
        #         anneal_offset[qubit] = list_of_anneal_offset[line][-1]
        # bqm = dimod.binary.BinaryQuadraticModel.from_ising(h={}, J=couplers)
        # success = False
        # while not success:
        #     try:
        #         sampleset = solver.sample(bqm, num_reads = num_reads_per_call, answer_mode = 'raw', auto_scale = False,
        #                         x_simple_anneal_time = anneal_time, flux_drift_compensation = False, flux_biases=fb, anneal_offsets = anneal_offset)
        #         success = True
        #     except RequestTimeout:
        #         success = False
        #         print('D-Wave Error, retrying...')
        # sample_dicts = get_sol_dict_from_dwave_output([sampleset])
        # average_qubit_mag = get_average_mag_per_qubit(qubit_in_use, sample_dicts)
        # average_coupler_corr = get_coupler_correlation(couplers, sample_dicts)
        # average_coupler_corr_per_bucket, bucket_vals = get_average_bucket_corr(corr_buckets, average_coupler_corr)
        # average_line_frust = get_average_line_frustration(anneal_line_couplers, average_coupler_corr)

    else:
        starting_ite = 0

    

    running_list = []

    if starting_ite >= num_calib_ite and exists(working_folder +"/running_list_" + str(anneal_time_ns) + "ns.xz"):
        print("No need to run...")
        running_list = pickle.load(lzma.open(working_folder +"/running_list_" + str(anneal_time_ns) + "ns.xz", 'rb'))
        print(f"Sample count in running list: {len(running_list)}")
        samples_used_for_analysis = len(running_list)
        # running_list_file = lzma.open(working_folder +"/running_list_" + str(anneal_time_ns) + "ns.xz", 'rb')
        # running_list = running_list_file.read()
    else:
        for ite in tqdm(range(starting_ite, num_calib_ite)):  
            if ite > 100:
                for q in qubit_in_use:
                    fb[q] -= flux_step_size * average_qubit_mag[q]
                    list_of_flux_bias[q].append(fb[q])
            else:
                for q in qubit_in_use:
                    fb[q] -= flux_step_size * average_qubit_mag[q] * 2
                    list_of_flux_bias[q].append(fb[q])

            need_normnalization = False
            if ite > -1:
                for i in range(len(corr_buckets)):
                    b = corr_buckets[i]
                    for coupler in b:
                        couplers[coupler] += sign_of(couplers[coupler]) * corr_step_size * (average_coupler_corr[coupler] - average_coupler_corr_per_bucket[i])
                        # couplers[coupler] = (1 - coupler_damp) * couplers[coupler] + coupler_damp * original_coupler_mag[coupler]
                        if couplers[coupler] > 1:
                            couplers[coupler] = 1
                            need_normnalization = True
                    if need_normnalization:
                        couplers = normaliza_couplers(couplers, coupler_strength,positive_couplers) 
                for coupler in couplers:        
                    list_of_coupler_strength[coupler].append(couplers[coupler])
            else:
                for coupler in couplers:        
                    list_of_coupler_strength[coupler].append(couplers[coupler])

            temp_coupler_frust_deviation = {}

            for i in range(len(corr_buckets)):
                b = corr_buckets[i]
                for coupler in b:
                    temp_coupler_frust_deviation[coupler] = (average_coupler_corr[coupler] - average_coupler_corr_per_bucket[i]) / len(b)
            
            temp_offset_per_line = np.zeros(8)

            for i in range(8):
                couplers_on_line = anneal_line_couplers[i]
                for c in couplers_on_line:
                    temp_offset_per_line[i] += temp_coupler_frust_deviation[c]
                    
            if ite > -1:
                for line in range(8):
                    temp = 0
                    for qubit in anneal_line_qubits[line]:
                        # anneal_offset[qubit] += anneal_offset_step_size * (average_line_frust[line] -  sum(average_line_frust)/8)
                        anneal_offset[qubit] += anneal_offset_step_size * temp_offset_per_line[line]
                        # anneal_offset[qubit] = (1-anneal_offset_damp) * anneal_offset[qubit]
                        temp = anneal_offset[qubit]
                    list_of_anneal_offset[line].append(temp)
            else:
                for line in range(8):
                    temp = 0
                    for qubit in anneal_line_qubits[line]:
                        anneal_offset[qubit] += anneal_offset_step_size * 5 * (average_line_frust[line] -  sum(average_line_frust)/8)
                        anneal_offset[qubit] = (1-anneal_offset_damp) * anneal_offset[qubit]
                        temp = anneal_offset[qubit]
                    list_of_anneal_offset[line].append(temp)

            bqm = dimod.binary.BinaryQuadraticModel.from_ising(h={}, J=couplers)
            success = False
            while not success:
                try:
                    sampleset = solver.sample(bqm, num_reads = num_reads_per_call, answer_mode = 'raw', auto_scale = False, fast_anneal=True,
                                    annealing_time = anneal_time, flux_biases=fb)#, anneal_offsets = anneal_offset)
                    success = True
                except RequestTimeout:
                    success = False
                    print('D-Wave Error, retrying...')

            sample_dicts = get_sol_dict_from_dwave_output([sampleset])
            average_qubit_mag = get_average_mag_per_qubit(qubit_in_use, sample_dicts)
            average_coupler_corr = get_coupler_correlation(couplers, sample_dicts)
            average_coupler_corr_per_bucket,bucket_vals = get_average_bucket_corr(corr_buckets, average_coupler_corr)
            average_line_frust = get_average_line_frustration(anneal_line_couplers, average_coupler_corr)
                
            
            # if ite >= num_calib_ite - 200:
            if ite >= num_calib_ite - int(np.ceil(samples_used_for_analysis / num_reads_per_call / embedding.shape[0])):
                for s in sample_dicts:
                    running_list.append(s)
        
        pickle.dump(running_list, lzma.open(working_folder +"/running_list_" + str(anneal_time_ns) + "ns.xz", 'wb'))
        pickle.dump(list_of_flux_bias, lzma.open(working_folder +"/flux_bias_" + str(anneal_time_ns) + "ns.xz", 'wb'))
        pickle.dump(list_of_coupler_strength, lzma.open(working_folder +"/coupler_strength_" + str(anneal_time_ns) + "ns.xz", 'wb'))
        pickle.dump(list_of_anneal_offset, lzma.open(working_folder +"/anneal_offset_" + str(anneal_time_ns) + "ns.xz", 'wb'))   

        # running_list_file_write = lzma.open(working_folder +"/running_list_" + str(anneal_time_ns) + "ns.xz", 'wb')
        # running_list_file_write.write(running_list)
        # list_of_flux_bias_file_write = lzma.open(working_folder +"/flux_bias_" + str(anneal_time_ns) + "ns.xz", 'wb')
        # list_of_flux_bias_file_write.write(list_of_flux_bias)
        # list_of_coupler_strength_file_write = lzma.open(working_folder +"/coupler_strength_" + str(anneal_time_ns) + "ns.xz", 'wb')
        # list_of_coupler_strength_file_write.write(list_of_coupler_strength)
        # list_of_anneal_offset_file_write = lzma.open(working_folder +"/anneal_offset_" + str(anneal_time_ns) + "ns.xz", 'wb')
        # list_of_anneal_offset_file_write.write(list_of_anneal_offset)
        
    # Plotting
    total_count, perfect_lattice_count = count_perfect_lattices(running_list, num_embedding, embedding, FM_bonds)
    average_qubit_mag = get_average_mag_per_qubit(qubit_in_use, running_list)
    average_coupler_corr = get_coupler_correlation(couplers, running_list)
    average_coupler_corr_per_bucket,bucket_vals = get_average_bucket_corr(corr_buckets, average_coupler_corr)
    print("Number of samples used for calculations:", len(running_list))
    print("Frustration after shimming:", sum(list(average_coupler_corr.values()))/len(couplers))
    ax.hist(average_qubit_mag.values(), mag_bins, alpha=0.5, label='with shimming')
    ax.legend(loc='upper right', fontsize = 20)

    pyplot.savefig(working_folder +"/shimming_mag_" + str(anneal_time_ns) + "ns_"+str(num_calib_ite) + "ite.pdf")

    pyplot.figure(2)
    tags = ['1', '2', '3', '4', '5', '6', '7', '8']
    offset_df = pd.DataFrame({"W/o shimming": initial_average_line_frust, "With shimming": average_line_frust}, index=tags)
    ax = offset_df.plot.bar(rot=0, figsize=(12, 8))
    ax.legend(loc='upper right', fontsize = 20)
    ax.set_xlabel("Line Number", fontsize=24)
    ax.set_ylabel("Average Line Frustration", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.axhline(y=0.333, color= 'green', linewidth=2,)
    ax.set_ylim(0.3,0.35)
    pyplot.tight_layout() 
    pyplot.savefig(working_folder +"/shimming_line_frust_" + str(anneal_time_ns) + "ns_"+str(num_calib_ite) + "ite.pdf")


    pyplot.figure(3)
    corr_bins = np.linspace(0,0.5,100)


    print(len(corr_buckets))
    pickle.dump(before_bucket_vals, open('before_shimming_bucket_frust.pickle', 'wb'))
    pickle.dump(bucket_vals, open('after_shimming_bucket_frust.pickle', 'wb'))
    plot_size = math.ceil(np.sqrt(len(corr_buckets)))
    if plot_size == 1:
        plot_size == 2
    fig, ax = pyplot.subplots(nrows=plot_size, ncols=plot_size,figsize=(8*plot_size, 6*plot_size))
    bukcet_index = 0
    inner_loop_broken = False
    for i in range(plot_size):
        for j in range(plot_size):
            ax[i,j].hist(before_bucket_vals[bukcet_index].values(), corr_bins, alpha=0.5, label='w/o shimming')
            bukcet_index += 1
            if bukcet_index == len(corr_buckets):
                inner_loop_broken = True
                break
        if inner_loop_broken:
            break
    
    inner_loop_broken = False
    bukcet_index = 0
    for i in range(plot_size):
        for j in range(plot_size):
            ax[i,j].hist(bucket_vals[bukcet_index].values(), corr_bins, alpha=0.5, label='with shimming')
            ax[i,j].legend(loc='upper right', fontsize = 24)
            ax[i,j].set_xlabel("Average Frustration", fontsize=24)
            ax[i,j].set_ylabel("Coupler Count", fontsize=24)
            ax[i,j].tick_params(axis='both', which='major', labelsize=20)
            bukcet_index += 1
            if bukcet_index == len(corr_buckets):
                inner_loop_broken = True
                break
        if inner_loop_broken:
            break
    pyplot.tight_layout()
    pyplot.savefig(working_folder + "/bucket_frustation_" + str(anneal_time_ns) + "ns_"+str(num_calib_ite) + "ite.pdf")


    
    pyplot.figure(4)
    num_plots = math.ceil(len(qubit_in_use) / 200)
    plot_size = math.ceil(np.sqrt(num_plots))
    fig, ax = pyplot.subplots(nrows=plot_size, ncols=plot_size,figsize=(6*plot_size, 6*plot_size))
    fig.suptitle("Flux Bias Evolution", fontsize=40)

    keys = list(list_of_flux_bias.keys())
    qubit_ind = 0
    fig_index = 0
    inner_loop_broken = False
    for i in range(plot_size):
        for j in range(plot_size):
            if fig_index == num_plots - 1:
                total_qubit_in_plot = len(qubit_in_use) % 200
            else:
                total_qubit_in_plot = 200
            colors = cm.gist_rainbow(np.linspace(0, 1, total_qubit_in_plot))
            for ite in range(total_qubit_in_plot):
                ax[i,j].plot(list_of_flux_bias[keys[qubit_ind]], color = colors[ite])
                qubit_ind += 1
            fig_index += 1
            if fig_index == num_plots:
                inner_loop_broken = True
                break
        if inner_loop_broken == True:
            break
    pyplot.savefig(working_folder + "/flux_bias_evolution_" + str(anneal_time_ns) + "ns_" +str(num_calib_ite) + "ite.pdf")


    pyplot.figure(5)
    num_plots_positive = math.ceil(len(positive_couplers) / 200)
    plot_size = math.ceil(np.sqrt(num_plots_positive))
    plot_size = 1

    fig, ax = pyplot.subplots(nrows=plot_size, ncols=plot_size,figsize=(12*plot_size, 8*plot_size))
    # fig.suptitle("Coupler Strength Evolution", fontsize=40)

    coupler_ind = 0
    fig_index = 0

    inner_loop_broken = False
    for i in range(plot_size):
        for j in range(plot_size):
            if fig_index == num_plots_positive - 1:
                total_coupler_in_plot = len(positive_couplers) % 25
            else:
                total_coupler_in_plot = 25
            colors = cm.gist_rainbow(np.linspace(0, 1, total_coupler_in_plot))
            ax.set_ylim([0.8, 1.0])
            ax.set_xlabel("Iteration", fontsize = 24)
            ax.set_ylabel("Coupler Strength", fontsize = 24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            for ite in range(total_coupler_in_plot):
                ax.plot(list_of_coupler_strength[positive_couplers[coupler_ind]], color = colors[ite])
                coupler_ind += 1
            fig_index += 1

            if fig_index == num_plots_positive:
                inner_loop_broken = True
                break
        if inner_loop_broken == True:
             break
    pyplot.tight_layout()
    pyplot.savefig(working_folder + "/coupler_strength_evolution_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) +"ite.pdf")


    pyplot.figure(6)         
    fig, ax = pyplot.subplots(nrows=1, ncols=1,figsize=(10, 9))
    fig.suptitle("Anneal Offsets Evolution", fontsize=40)
    colors = cm.gist_rainbow(np.linspace(0, 1, len(list_of_anneal_offset)))
    ind = 0
    for line in list_of_anneal_offset:
        ax.plot(list_of_anneal_offset[line], color = colors[ind], label = "line " + str(line+1))
        ind += 1
    ax.legend(loc='upper right')
    pyplot.savefig(working_folder +"/anneal_offset_evolution_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    
    print("Total number of lattices:",total_count)
    print("Total number of perfect triangular embeddings:", perfect_lattice_count)

    triangular_psi, real_psi_tri, imag_psi_tri = get_order_parameter_triangle(running_list, num_embedding, embedding, k_away_from_boundry_omitted, FM_bonds, only_count_perfect_samples = False)
    # villain_psi,real_psi, imag_psi = get_order_parameter_villain(running_list, k_away_from_boundry_omitted, dim_x, dim_y, num_embedding, embedding, original_coupler_mag)
    villain_psi_sandvik, real_psi_villain, imag_psi_villain = get_order_parameter_villain_sandvik(running_list, k_away_from_boundry_omitted, dim_x, dim_y, num_embedding, embedding)
    print("Average Villain Magnetization Order Parameter:", villain_psi_sandvik[0])
    print("SEM of Villain Magnetization Order Parameter", villain_psi_sandvik[1])
    print("Average Triangular Order Parameter:", triangular_psi[0])
    print("SEM of Triangular Order Parameter", triangular_psi[1])

    residual_energy_mean, residual_energy_sem = compute_average_residual_energy(running_list, couplers_per_embedding, ground_state_energy)
    vortices_mean, vortices_sem, vortices_bootstrap_sem,  vortices_std, vortices_bootstrap_std= compute_averge_vortices(running_list, couplers_per_embedding, ground_state_vortices)
    # residual_energy_mean = 0
    # residual_energy_sem = 0
    # vortices_mean = 0
    # vortices_sem = 0


    pyplot.figure(7)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (12,8))
    ax.set_facecolor(color='black')
    ax.set_aspect('equal', 'datalim')
    fig.suptitle("Villain Order Parameter 2D Histo " + str(anneal_time_ns) + "ns", fontsize=15)
    h = ax.hist2d(real_psi_villain, imag_psi_villain, bins=43, range=np.array([(-1, 1), (-1, 1)]), norm=mpl.colors.LogNorm(), cmap='magma')     
    fig.colorbar(h[3])  
    pyplot.savefig(working_folder + "/villain_order_2d_histo_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    pyplot.figure(8)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (12,8))
    ax.set_facecolor(color='black')
    ax.set_aspect('equal', 'datalim')
    fig.suptitle("Triangular Order Parameter 2D Histo " + str(anneal_time_ns) + "ns", fontsize=15)
    h = ax.hist2d(real_psi_tri, imag_psi_tri, bins=43, range=np.array([(-1, 1), (-1, 1)]), norm=mpl.colors.LogNorm(), cmap='magma')     
    fig.colorbar(h[3])  
    pyplot.savefig(working_folder + "/triangular_order_2d_histo_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    pyplot.figure(9)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Triangular Lattice @ {round(anneal_time_ns,1)}ns", fontsize=30)
    G=nx.Graph()
    G.add_nodes_from(list(range(dim_x * dim_y)))
    curr_x = 0
    curr_y = 0
    pos = {}
    for i in range(dim_x * dim_y):
        pos[i] = (curr_x, dim_y - 1 - curr_y)
        curr_y += 1
        if curr_y == dim_y:
            curr_y = 0
            curr_x += 1
    curr_sample = running_list[0]
    curr_embedding = embedding[0]
    positive_spins = []
    negative_spins = []
    colors = []
    spin_counter = 0
    for i in range(dim_y):
        for j in range(dim_x):
            if curr_sample[curr_embedding[j,i]] == 1:
                colors.append("red")
                positive_spins.append(spin_counter)
                spin_counter += 1
            else:
                colors.append("blue")
                negative_spins.append(spin_counter)
                spin_counter += 1
    nx.draw_networkx_nodes(G, pos=pos, nodelist=positive_spins,
                       node_color='red', label='+1')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=negative_spins,
                        node_color='blue', label='-1')
    nx.draw_networkx_edges(G, pos=pos)
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.axis('off')
    pyplot.savefig(working_folder + "/triangular_spin_config_1_" +  str(anneal_time_ns) +"ns" + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')

    pyplot.figure(10)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Triangular Lattice @ {round(anneal_time_ns,1)}ns", fontsize=30)
    G=nx.Graph()
    G.add_nodes_from(list(range(dim_x * dim_y)))
    curr_x = 0
    curr_y = 0
    pos = {}
    for i in range(dim_x * dim_y):
        pos[i] = (curr_x, dim_y - 1 - curr_y)
        curr_y += 1
        if curr_y == dim_y:
            curr_y = 0
            curr_x += 1
    curr_sample = running_list[10]
    curr_embedding = embedding[0]
    positive_spins = []
    negative_spins = []
    colors = []
    spin_counter = 0
    for i in range(dim_y):
        for j in range(dim_x):
            if curr_sample[curr_embedding[j,i]] == 1:
                colors.append("red")
                positive_spins.append(spin_counter)
                spin_counter += 1
            else:
                colors.append("blue")
                negative_spins.append(spin_counter)
                spin_counter += 1
    nx.draw_networkx_nodes(G, pos=pos, nodelist=positive_spins,
                       node_color='red', label='+1')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=negative_spins,
                        node_color='blue', label='-1')
    nx.draw_networkx_edges(G, pos=pos)
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.axis('off')
    pyplot.savefig(working_folder + "/triangular_spin_config_2_" +  str(anneal_time_ns) +"ns" + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')

    pyplot.figure(11)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Triangular Lattice @ {round(anneal_time_ns,1)}ns", fontsize=30)
    G=nx.Graph()
    G.add_nodes_from(list(range(dim_x * dim_y)))
    curr_x = 0
    curr_y = 0
    pos = {}
    for i in range(dim_x * dim_y):
        pos[i] = (curr_x, dim_y - 1 - curr_y)
        curr_y += 1
        if curr_y == dim_y:
            curr_y = 0
            curr_x += 1
    curr_sample = running_list[10]
    curr_embedding = embedding[0]
    positive_spins = []
    negative_spins = []
    colors = []
    spin_counter = 0
    for i in range(dim_y):
        for j in range(dim_x):
            if curr_sample[curr_embedding[j,i]] == 1:
                colors.append("red")
                positive_spins.append(spin_counter)
                spin_counter += 1
            else:
                colors.append("blue")
                negative_spins.append(spin_counter)
                spin_counter += 1
    nx.draw_networkx_nodes(G, pos=pos, nodelist=positive_spins,
                       node_color='red', label='+1')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=negative_spins,
                        node_color='blue', label='-1')
    nx.draw_networkx_edges(G, pos=pos)
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.axis('off')
    pyplot.savefig(working_folder + "/triangular_spin_config_3_" +  str(anneal_time_ns) +"ns" + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')

    pyplot.figure(12)
    fourier, labels, corr_length_x, corr_length_y, sigma_x_1_negative, sigma_x_2_negative, sigma_x_1_positive, sigma_x_2_positive, sigma_y_1_negative, sigma_y_2_negative, sigma_y_1_positive, sigma_y_2_positive = get_fourier_transform(running_list, anneal_time_ns, working_folder, dim_x, dim_y, num_embedding, embedding, 513, samples_used_for_fourier= samples_used_for_analysis, ignore_boundary=k_away_from_boundry_omitted)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (12,8))
    df_cm = pd.DataFrame(fourier, index = labels, columns = labels)
    sn.heatmap(df_cm, annot=False, ax=ax, square=True, cmap="magma", cbar=True)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # im = ax.imshow(fourier, cmap="magma", interpolation='gaussian')
    # fig.colorbar(im, cax=cax, orientation='vertical')
    fig.suptitle(f"{dim_x} X {dim_y} Tri Lattice Fourier Transform {anneal_time_ns} ns", fontsize=15)
    ax.set_xlabel("ky")
    ax.set_ylabel("kx")
    pyplot.tight_layout()
    pyplot.savefig(working_folder + f"/fourier_heatmap_noboundary_{k_away_from_boundry_omitted}_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    pickle.dump(fourier, open(f'36_36_{anneal_time_ns}ns.pickle', 'wb'))
    
    # villain_psi_sandvik, residual_energy_mean, residual_energy_sem, vortices_mean, vortices_sem = 0
    return pre_shimmed_triangular_psi, triangular_psi, villain_psi_sandvik, [residual_energy_mean, residual_energy_sem], [vortices_mean, vortices_sem, vortices_bootstrap_sem,  vortices_std, vortices_bootstrap_std], corr_length_x, corr_length_y, sigma_x_1_negative, sigma_x_2_negative, sigma_x_1_positive, sigma_x_2_positive, sigma_y_1_negative, sigma_y_2_negative, sigma_y_1_positive, sigma_y_2_positive

def create_qpu_input(dim_x, dim_y, input_filename, coupler_strength):
    input_file = open(input_filename, 'r')
    lines = input_file.readlines()
    num_embedding = len(lines)
    input_file.close()
    embedding = np.zeros((num_embedding, dim_x, dim_y), dtype=int)
    qubit_in_use = []

    for ite in range(num_embedding):
        curr_embedding = lines[ite]
        tokens = curr_embedding.split()
        counter = 0
        for col in range(dim_y):
            for row in range(dim_x):
                embedding[ite, row, col] = int(tokens[counter])
                qubit_in_use.append(int(tokens[counter]))
                counter += 1

    ####
    # solver = DWaveSampler(solver='Advantage_system4.1')
    # fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (36,36))
    # H = dnx.pegasus_graph(16,node_list=solver.properties['qubits'],edge_list=solver.properties['couplers'],check_node_list=True,check_edge_list=True)
    # nodelist = H.nodes()
    # node_colors = []
    # for n in nodelist:
    #     if n in qubit_in_use:
    #         node_colors.append('red')
    #     else:
    #         node_colors.append('blue')
    # dnx.draw_pegasus(H,node_size=20,ax=ax,node_color=node_colors)
    # pyplot.savefig('pegasus.png')
    # pyplot.tight_layout()
    # exit()
    ####
    couplers = {}
    couplers_per_embedding = []
    FM_bonds = []
    for ite in range(num_embedding):
        couplers_in_curr_embedding = {}
        FM_bonds.append([])
        for i in range(dim_x):
            curr_row = embedding[ite, i, :] 
            for m in range(len(curr_row) - 1):
                couplers[(curr_row[m], curr_row[m+1])] = coupler_strength
                couplers_in_curr_embedding[(curr_row[m], curr_row[m+1])] = coupler_strength

        for i in range(dim_y):
            curr_col = embedding[ite, :, i] 
            if i % 2 == 0:
                for m in range(len(curr_col) - 1):
                    if m % 2 == 0:
                        if i == 0:
                            couplers[(curr_col[m], curr_col[m+1])] = -1
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = -1
                            FM_bonds[-1].append((curr_col[m], curr_col[m+1]))
                            
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = -2
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = -2
                            FM_bonds[-1].append((curr_col[m], curr_col[m+1]))
                            
                    else:
                        if i == 0:
                            couplers[(curr_col[m], curr_col[m+1])] = coupler_strength * 0.5
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = coupler_strength * 0.5
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = coupler_strength 
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = coupler_strength 
                if i == 0:
                    couplers[(curr_col[-1], curr_col[0])] = coupler_strength * 0.5
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = coupler_strength * 0.5
                else:
                    couplers[(curr_col[-1], curr_col[0])] = coupler_strength 
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = coupler_strength 
            else:
                for m in range(len(curr_col) - 1):
                    if m % 2 != 0:
                        if i == dim_y - 1:
                            couplers[(curr_col[m], curr_col[m+1])] = -1
                            FM_bonds[-1].append((curr_col[m], curr_col[m+1]))
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = -1
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = -2
                            FM_bonds[-1].append((curr_col[m], curr_col[m+1]))
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = -2
                    else:
                        if i == dim_y - 1:
                            couplers[(curr_col[m], curr_col[m+1])] = coupler_strength * 0.5
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = coupler_strength * 0.5
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = coupler_strength 
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = coupler_strength 
                if i == dim_y-1:
                    couplers[(curr_col[-1], curr_col[0])] = -1
                    FM_bonds[-1].append((curr_col[-1], curr_col[0]))
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = -1
                else:
                    couplers[(curr_col[-1], curr_col[0])] = -2
                    FM_bonds[-1].append((curr_col[-1], curr_col[0]))
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = -2
        couplers_per_embedding.append(couplers_in_curr_embedding)

    return num_embedding, embedding, qubit_in_use, couplers, FM_bonds, couplers_per_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triangular Model')
    parser.add_argument('-x', metavar='dim_x', type=int,
                        help='x dimension of the square lattice')

    parser.add_argument('-y', metavar='dim_y', type=int,
                        help='y dimension of the square lattice')

    parser.add_argument("-ns", metavar='annealing_time', type=float,
                        help='annealing time in the unit of ns')

    parser.add_argument("-ite", metavar='num_calib_ite', type=int,
                        help='number of iterations for shimming')
    
    # parser.add_argument("-out", metavar='output_file_name', type=str,
    #                     help='Output file name for order parameters' )

    args = parser.parse_args()
    dim_x = args.x
    dim_y = args.y
    if dim_x % 3 != 0 and dim_y % 2 != 0 :
        print("Dimension not supported")
        exit(1)
    annealing_time_ns = args.ns
    num_calib_ite = args.ite
    output_filename = f'{dim_x}_{dim_y}'


    input_filename = './input/bay8_square_cylinder_multiple_txt_' + str(dim_x).zfill(2) + 'x' + str(dim_y).zfill(2) + '.txt'
    coupler_strength = 0.9

    num_embedding, embedding, qubit_in_use, couplers, FM_bonds, couplers_per_embedding = create_qpu_input(dim_x, dim_y, input_filename, coupler_strength)
    
    individual_buckets = get_buckets(dim_x,dim_y, num_embedding, embedding, couplers)
    # total_buckets = []
    # for b in individual_buckets:
    #     for c in b:
    #         total_buckets.append(c)
    k_away_from_boundry_omitted = 0
    samples_used_for_analysis = 100000

    #corr_step_size = 0.01

    pre_shimmed_triangular_psi, triangular_psi, villain_psi, residual_energy, vortices, corr_length_x, corr_length_y, sigma_x_1_negative, sigma_x_2_negative, sigma_x_1_positive, sigma_x_2_positive, sigma_y_1_negative, sigma_y_2_negative, sigma_y_1_positive, sigma_y_2_positive= qpu_run(qubit_in_use = qubit_in_use, 
            couplers = couplers, coupler_strength = coupler_strength, num_calib_ite = num_calib_ite, 
            dim_x = dim_x, dim_y = dim_y,  FM_bonds = FM_bonds, couplers_per_embedding = couplers_per_embedding,
            num_embedding = num_embedding, embedding = embedding, 
            flux_step_size = 2e-6, corr_step_size = 0.0025, anneal_offset_step_size = 0.001, 
            coupler_damp = 0, anneal_offset_damp = 0, num_reads_per_call = 100, anneal_time_ns = annealing_time_ns, 
            corr_buckets = individual_buckets, k_away_from_boundry_omitted = k_away_from_boundry_omitted, samples_used_for_analysis=samples_used_for_analysis)
    

    


    if output_filename == 'null':
        exit(0)
    # input("Press Enter to continue...")

    working_folder = f".{os.sep}triangle_" + str(dim_x) + "_" + str(dim_y) + f"dim{os.sep}"

    out_file = open(working_folder +'triangular_' + output_filename + '_boundary_' + str(k_away_from_boundry_omitted) + '.csv', 'a')
    if os.stat(working_folder +'triangular_' + output_filename + '_boundary_' + str(k_away_from_boundry_omitted) + '.csv').st_size == 0:
        out_file.write(f'annealing_time,unshimmed_average_triangular_op,unshimmed_triangular_op_sem,unshimmed_triangular_op_bootstrap_sem,unshimmed_triangular_op_std,unshimmed_triangular_op_bootstrap_std,average_triangular_op,triangular_op_sem,triangular_op_bootstrap_sem,triangular_op_std,triangular_op_bootstrap_std,average_villain_op,villain_op_sem,villain_op_bootstrap_sem,villain_op_std,villain_op_bootstrap_std,average_vortices,vortices_sem,vortices_bootstrap_sem,vortices_std,vortices_bootstrap_std,corr_length_periodic,corr_length_periodic_sigma_1_negative,corr_length_periodic_sigma_1_positive,corr_length_periodic_sigma_2_negative,corr_length_periodic_sigma_2_positive,corr_length_non_periodic,corr_length_non_periodic_sigma_1_negative,corr_length_non_periodic_sigma_1_positive,corr_length_non_periodic_sigma_2_negative,corr_length_non_periodic_sigma_2_positive\n')
    out_file.write(str(annealing_time_ns))

    for s in pre_shimmed_triangular_psi:
        out_file.write(',' + str(s))

    for s in triangular_psi:
        out_file.write(',' + str(s))
    
    for s in villain_psi:
        out_file.write(',' + str(s))

    for s in vortices:
        out_file.write(',' + str(s))

    out_file.write(f',{corr_length_x},{sigma_x_1_negative},{sigma_x_1_positive},{sigma_x_2_negative},{sigma_x_2_positive},{corr_length_y},{sigma_y_1_negative},{sigma_y_1_positive},{sigma_y_2_negative},{sigma_y_2_positive}')
    out_file.write('\n')
    out_file.close()






    # out_file = open(working_folder +'residual_energy_' + output_filename +  '.txt', 'a')
    # out_file.write(str(annealing_time_ns) + '\n')
    # for s in residual_energy:
    #     out_file.write(str(s) + ' ')
    # out_file.write('\n')
    # out_file.close()


    











