
import argparse
import math
import os
import numpy as np
from dwave.system import DWaveSampler
import pickle
import dimod
import matplotlib.pyplot as plt
from matplotlib import pyplot
import dwave_networkx as dnx
import matplotlib.cm as cm
from os.path import exists
import lzma
import matplotlib as mpl
from tqdm import tqdm
from statistics import mean
from scipy.stats import sem
import colorama
from colorama import Fore, Style
import time
import random
import seaborn as sn
import pandas as pd
import itertools
import neal
import networkx as nx


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
        if c not in couplers:
            c = (j,i)
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
        anneal_line_left = pegasus_control_line_from_linear(c[0])
        anneal_line_right =  pegasus_control_line_from_linear(c[1])
        if (c[0], c[1]) not in anneal_line_couplers[anneal_line_left] and (c[1], c[0]) not in anneal_line_couplers[anneal_line_left]:
            anneal_line_couplers[anneal_line_left].append(c)
        if (c[0], c[1]) not in anneal_line_couplers[anneal_line_right] and (c[1], c[0]) not in anneal_line_couplers[anneal_line_right]:
            anneal_line_couplers[anneal_line_right].append(c)

    return anneal_line_qubits, anneal_line_couplers


def get_buckets(dim_x, dim_y, num_embedding, embedding):
    buckets = []
    for i in range(dim_y):
        curr_bucket_odd = []
        curr_bucket_even = []
        for ite in range(num_embedding):
            curr_col = embedding[ite, :, i] 
            
            for m in range(len(curr_col)-1):
                if m % 2 == 0:
                    curr_bucket_even.append((curr_col[m], curr_col[m+1]))
                else:
                    curr_bucket_odd.append((curr_col[m], curr_col[m+1]))
            curr_bucket_odd.append((curr_col[-1], curr_col[0]))


        buckets.append(curr_bucket_even)
        buckets.append(curr_bucket_odd)
    
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

def is_frustrated(coupler_strength, i, j):
    # coupler_strength = -0.5
    return (sign_of(coupler_strength) * i * j +1) * 0.5

def get_order_parameter_villain(samples, k_away_from_boundry_omitted, dim_x, dim_y, num_embedding, embedding, couplers):
    order_parameters = []
    real_psi = []
    imag_psi = []
    for ite in range(num_embedding):
        # psi = 0
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
            psi = np.abs(even_row + even_col * 1j - odd_row - odd_col * 1j)
            real_psi.append(even_row - odd_row)
            imag_psi.append(even_col - odd_col)
        # psi /= len(samples)
            order_parameters.append(psi)
    return [mean(order_parameters), sem(order_parameters)], real_psi, imag_psi

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

def get_order_parameter_afm_mag(samples, k_away_from_boundry_omitted,dim_x, dim_y, num_embedding, embedding):
    order_parameters = []

    order_parameter_batches = []
    current_batch = []

    for ite in range(num_embedding):
        # psi = 0
        for sample in samples:
            qubit_used = 0
            mag = 0
            for i in range(dim_x):
                curr_row = embedding[ite ,i, :]
                for m in range(len(curr_row)):
                    if m < k_away_from_boundry_omitted or dim_y-m <= k_away_from_boundry_omitted:
                        continue
                    
                    mag += sample[curr_row[m]] * ((-1) ** (i+m))
                    qubit_used += 1
            mag /= qubit_used
            psi = np.abs(mag)
        # psi /= len(samples)
            order_parameters.append(psi)
            current_batch.append(psi)
            if len(current_batch) == 100 * embedding.shape[0]:
                order_parameter_batches.append(mean(current_batch))
                current_batch = []
    if current_batch:
        order_parameter_batches.append(mean(current_batch))

    return [mean(order_parameters), sem(order_parameters), sem(order_parameter_batches), np.std(order_parameters), np.std(order_parameter_batches)]

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


def get_fourier_transform(samples, dim_x, dim_y, num_embedding, embedding, num_steps):
    fourier = np.zeros((num_steps,num_steps))
    labels = np.linspace(-2,2,num_steps)


    for ite in range(num_embedding):
        for sample in np.random.choice(samples, 6, replace=False):
            sample_matrix = np.zeros((dim_x, dim_y))
            for i in range(dim_x):
                for j in range(dim_y):
                    sample_matrix[i,j] = embedding[ite, i,j]
            fft_result = np.fft.fft2(sample_matrix)
            print(fft_result)
            exit()


    for xind, yind in tqdm(itertools.product(range(len(labels)), range(len(labels)))):
            kx = labels[xind]
            ky = labels[yind]
            f = 0
            total_count = 0
            for ite in range(num_embedding):
                list_of_spins = []
                for x in range(dim_x):
                    for y in range(dim_y):
                        list_of_spins.append((embedding[ite, x,y], x, y))
                
                for sample in np.random.choice(samples, 6, replace=False):
                    for i in range(len(list_of_spins)-1):
                        for j in range(i+1, len(list_of_spins)):
                            f += sample[list_of_spins[i][0]] * sample[list_of_spins[j][0]] * np.exp(1j*np.pi*kx*(list_of_spins[i][1]-list_of_spins[j][1]) + 1j*np.pi*ky*(list_of_spins[i][2]-list_of_spins[j][2]))
                    total_count += 1

            
            fourier[xind, yind] = np.abs(f / (dim_x * dim_y) / total_count)

    return fourier, list(np.round(np.linspace(-2,2,num_steps),2))


def get_fourier_transform_redacted(samples, dim_x, dim_y, num_embedding, embedding, num_steps):
    fourier = np.zeros((num_steps,num_steps))
    labels = np.linspace(-2,2,num_steps)


    for xind, yind in tqdm(itertools.product(range(len(labels)), range(len(labels)))):
            kx = labels[xind]
            ky = labels[yind]
            f = 0
            total_count = 0
            for ite in range(num_embedding):
                list_of_spins = []
                for x in range(dim_x):
                    for y in range(dim_y):
                        list_of_spins.append((embedding[ite, x,y], x, y))
                
                for sample in np.random.choice(samples, 6, replace=False):
                    for i in range(len(list_of_spins)-1):
                        for j in range(i+1, len(list_of_spins)):
                            f += sample[list_of_spins[i][0]] * sample[list_of_spins[j][0]] * np.exp(1j*np.pi*kx*(list_of_spins[i][1]-list_of_spins[j][1]) + 1j*np.pi*ky*(list_of_spins[i][2]-list_of_spins[j][2]))
                    total_count += 1

            
            fourier[xind, yind] = np.abs(f / (dim_x * dim_y) / total_count)

    return fourier, list(np.round(np.linspace(-2,2,num_steps),2))
                    



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

def get_order_parameter_triangle(samples, num_embedding, embedding, k_away_from_boundry_omitted): 
    order_parameters = []
    real_psi = []
    imag_psi = []

    order_parameter_batches = []
    current_batch = []

    for ite in range(num_embedding):
        # psi = 0
        coloring = get_coloring(embedding[ite], k_away_from_boundry_omitted)
        for sample in samples:
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
        # psi /= len(samples)
            order_parameters.append(psi)
            current_batch.append(psi)
            if len(current_batch) == 100 * embedding.shape[0]:
                order_parameter_batches.append(mean(current_batch))
                current_batch = []
    if current_batch:
        order_parameter_batches.append(mean(current_batch))

    return [mean(order_parameters), sem(order_parameters), sem(order_parameter_batches), np.std(order_parameters), np.std(order_parameter_batches)], real_psi, imag_psi

def normaliza_couplers(couplers, J_AFM, J_FM, positive_couplers, negative_couplers):
    positive_sum = 0
    negative_sum = 0
    scaled_couplers = {}
    for coupler in couplers:
        if couplers[coupler] > 0:
            positive_sum += couplers[coupler]
        else:
            negative_sum += couplers[coupler]
    
    positive_mean = positive_sum / len(positive_couplers)
    negative_mean = negative_sum / len(negative_couplers)
    for coupler in positive_couplers:
        scaled_couplers[coupler] = couplers[coupler] / (positive_mean / J_AFM)
        if scaled_couplers[coupler] > 1:
            scaled_couplers[coupler] = 1
    
    for coupler in negative_couplers:
        scaled_couplers[coupler] = couplers[coupler] / (-1 * negative_mean / J_FM)
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
    


    
def qpu_run(qubit_in_use, couplers, J_AFM, J_FM, num_calib_ite, dim_x, dim_y, couplers_per_embedding, num_embedding, embedding,  flux_step_size, corr_step_size, anneal_offset_step_size, coupler_damp, anneal_offset_damp, num_reads_per_call, anneal_time_ns, corr_buckets, k_away_from_boundry_omitted,samples_used_for_analysis ):
    #Collect the qpu stats without any calibration
    solver = DWaveSampler(solver='Advantage_system4.1')
    working_folder = "./villain_" + str(dim_x) + "_" + str(dim_y) + "dim_" +str(round(J_FM/J_AFM, 2))+ "/" + str(anneal_time_ns) + "ns"
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)


    anneal_time = anneal_time_ns / 1000
    original_coupler_mag = couplers.copy()
    positive_couplers = []
    negative_couplers = []

    for c in original_coupler_mag:
        if original_coupler_mag[c] < 0:
            negative_couplers.append(c)
        else:
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
        # initial_offsets[i] = float(tokens[i])
        list_of_anneal_offset[i] = []


    anneal_line_qubits, anneal_line_couplers = get_anneal_lines(qubit_in_use, couplers)
    # for line in range(8):
    #     for qubit in anneal_line_qubits[line]:
    #         anneal_offset[qubit] = initial_offsets[line]

    sa_sampler = neal.SimulatedAnnealingSampler()
    sa_sampleset = sa_sampler.sample(bqm, num_reads=1000)

    ground_state_sample = sa_sampleset.first.sample
    ground_state_energy = sa_sampleset.first.energy / num_embedding
    ground_state_vortices = count_ground_vortice(ground_state_sample, couplers_per_embedding)

    if not exists(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz"):
        for ite in tqdm(range(100)):
            sampleset = solver.sample(bqm, num_reads = num_reads_per_call, answer_mode='raw', auto_scale=False,
                        fast_anneal=True,
                        annealing_time = anneal_time, flux_drift_compensation = False, flux_biases=fb )
            sampleset_collection.append(sampleset)
        pickle.dump(sampleset_collection, lzma.open(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz", 'wb'))
    else:
        sampleset_collection = pickle.load(lzma.open(working_folder + "/baseline_samples_"+ str(anneal_time_ns) + "ns" ".xz", 'rb'))

    
    sample_dicts = get_sol_dict_from_dwave_output(sampleset_collection)
    average_qubit_mag = get_average_mag_per_qubit(qubit_in_use, sample_dicts)
    average_coupler_corr = get_coupler_correlation(couplers, sample_dicts)
    average_coupler_corr_per_bucket, before_bucket_vals = get_average_bucket_corr(corr_buckets, average_coupler_corr)
    average_line_frust = get_average_line_frustration(anneal_line_couplers, average_coupler_corr)
    print("Frustration before shimming:", sum(list(average_coupler_corr.values()))/len(couplers))

    # Plotting
    pyplot.figure(1)
    mag_bins = np.linspace(-1, 1, 100)
    corr_bins = np.linspace(0, 0.5, 100)
    fig, ax = pyplot.subplots(nrows=2, ncols=1,figsize=(10, 12))
    ax[0].hist(average_qubit_mag.values(), mag_bins, alpha=0.5, label='without shimming')
    ax[0].set_title("Qubit Magnetization", fontsize=40)
    # ax[1].hist(average_coupler_corr.values(), corr_bins, alpha=0.5, label='without shimming')
    # ax[1].set_title("Coupler Frustration", fontsize=40)
    ax[1].plot(average_line_frust, color = 'blue' ,label = 'without shimming')
    ax[1].set_title("Anneal Line Frustration", fontsize=40)


    list_of_flux_bias = {}
    for i in range(len(qubit_in_use)):
        list_of_flux_bias[qubit_in_use[i]] = [0]
    
    list_of_coupler_strength = {}
    for coupler in couplers:
        list_of_coupler_strength[coupler] = [couplers[coupler]]
    


    if exists(working_folder + "/anneal_offset_" + str(anneal_time_ns) + "ns.xz") and exists(working_folder + "/coupler_strength_" + str(anneal_time_ns) + "ns.xz") and exists( working_folder + "/flux_bias_" + str(anneal_time_ns) + "ns.xz"):
        print("Previous run detected, loading data...")
        list_of_flux_bias = pickle.load(lzma.open(working_folder + "/flux_bias_" + str(anneal_time_ns) + "ns.xz", 'rb'))
        list_of_coupler_strength = pickle.load(lzma.open(working_folder + "/coupler_strength_" + str(anneal_time_ns) + "ns.xz", 'rb'))
        list_of_anneal_offset = pickle.load(lzma.open(working_folder + "/anneal_offset_" + str(anneal_time_ns) + "ns.xz", 'rb'))

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
        #     except:
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


    if starting_ite >= num_calib_ite and exists(working_folder + "/running_list_" + str(anneal_time_ns) + "ns.xz"):
        print("No need to run...")
        
        running_list = pickle.load(lzma.open(working_folder + "/running_list_" + str(anneal_time_ns) + "ns.xz", 'rb'))
        print(f"Samples count in running list: {len(running_list)}")
    else:
        for ite in tqdm(range(starting_ite, num_calib_ite)):  

            if ite > 50:
                for q in qubit_in_use:
                    fb[q] -= flux_step_size * average_qubit_mag[q]
                    list_of_flux_bias[q].append(fb[q])
            else:
                for q in qubit_in_use:
                    fb[q] -= flux_step_size * average_qubit_mag[q] * 5
                    list_of_flux_bias[q].append(fb[q])

            need_normnalization = False
            if ite > -1:
                for i in range(len(corr_buckets)):
                    b = corr_buckets[i]
                    for coupler in b:
                        if couplers[coupler] == 0:
                            continue
                        couplers[coupler] += sign_of(couplers[coupler]) * corr_step_size * (average_coupler_corr[coupler] - average_coupler_corr_per_bucket[i])
                        # couplers[coupler] = (1 - coupler_damp) * couplers[coupler] + coupler_damp * original_coupler_mag[coupler]
                        if couplers[coupler] > 1:
                            couplers[coupler] = 1
                            need_normnalization = True
                    if need_normnalization:
                        couplers = normaliza_couplers(couplers, J_AFM, J_FM,positive_couplers, negative_couplers) 
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
                    sampleset = solver.sample(bqm, num_reads = num_reads_per_call, answer_mode = 'raw', auto_scale = False,
                                    fast_anneal=True,
                                    annealing_time = anneal_time, flux_biases=fb)
                    success = True
                except:
                    success = False
                    print('D-Wave Error, retrying...')

            sample_dicts = get_sol_dict_from_dwave_output([sampleset])
            average_qubit_mag = get_average_mag_per_qubit(qubit_in_use, sample_dicts)
            average_coupler_corr = get_coupler_correlation(couplers, sample_dicts)
            average_coupler_corr_per_bucket,bucket_vals = get_average_bucket_corr(corr_buckets, average_coupler_corr)
            average_line_frust = get_average_line_frustration(anneal_line_couplers, average_coupler_corr)
                

            if ite >= num_calib_ite - int(np.ceil(samples_used_for_analysis / num_reads_per_call / embedding.shape[0])):
                for s in sample_dicts:
                    running_list.append(s)

        pickle.dump(running_list, lzma.open(working_folder + "/running_list_" + str(anneal_time_ns) + "ns.xz", 'wb'))
        pickle.dump(list_of_flux_bias, lzma.open(working_folder + "/flux_bias_" + str(anneal_time_ns) + "ns.xz", 'wb'))
        pickle.dump(list_of_coupler_strength, lzma.open(working_folder + "/coupler_strength_" + str(anneal_time_ns) + "ns.xz", 'wb'))
        pickle.dump(list_of_anneal_offset, lzma.open(working_folder + "/anneal_offset_" + str(anneal_time_ns) + "ns.xz", 'wb')) 
        
    # Plotting
    

    average_qubit_mag = get_average_mag_per_qubit(qubit_in_use, running_list)
    average_coupler_corr = get_coupler_correlation(couplers, running_list)
    average_coupler_corr_per_bucket,bucket_vals = get_average_bucket_corr(corr_buckets, average_coupler_corr)

    print("Number of samples used for calculations:", len(running_list))
    print("Frustration after shimming:", sum(list(average_coupler_corr.values()))/len(couplers))
    ax[0].hist(average_qubit_mag.values(), mag_bins, alpha=0.5, label='with shimming')
    # ax[1].hist(average_coupler_corr.values(), corr_bins, alpha=0.5, label='with shimming')
    ax[1].plot(average_line_frust, color = 'orange' , label = 'with shimming')
    ax[0].legend(loc='upper right')
    # ax[1].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    pyplot.savefig(working_folder + "/shimming_" + str(anneal_time_ns) + "ns_"+str(num_calib_ite) + "ite.png")

    pyplot.figure(2)
    corr_bins = np.linspace(0,1,100)



    plot_size = math.ceil(np.sqrt(len(corr_buckets)))
    fig, ax = pyplot.subplots(nrows=plot_size, ncols=plot_size,figsize=(6*plot_size, 6*plot_size))
    fig.suptitle("Symmetric Class Frustration", fontsize=40)
    bukcet_index = 0
    inner_loop_broken = False
    for i in range(plot_size):
        for j in range(plot_size):
            ax[i,j].hist(before_bucket_vals[bukcet_index].values(), corr_bins, alpha=0.5, label='without shimming')
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
            ax[i,j].legend(loc='upper right')
            bukcet_index += 1
            if bukcet_index == len(corr_buckets):
                inner_loop_broken = True
                break
        if inner_loop_broken:
            break
    pyplot.savefig(working_folder + "/bucket_frustation_" + str(anneal_time_ns) + "ns_"+str(num_calib_ite) + "ite.png")


    
    pyplot.figure(3)
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
    pyplot.savefig(working_folder + "/flux_bias_evolution_" + str(anneal_time_ns) + "ns_" +str(num_calib_ite) + "ite.png")


    pyplot.figure(4)
    num_plots_positive = math.ceil(len(positive_couplers) / 200)
    num_plots_negative = math.ceil(len(negative_couplers) / 200)
    plot_size = math.ceil(np.sqrt(num_plots_positive + num_plots_negative))

    fig, ax = pyplot.subplots(nrows=plot_size, ncols=plot_size,figsize=(6*plot_size, 6*plot_size))
    fig.suptitle("Coupler Strength Evolution", fontsize=40)

    coupler_ind = 0
    fig_index = 0

    positive_negative_exit = 0
    for i in range(plot_size):
        for j in range(plot_size):

            if positive_negative_exit == 0:
                if fig_index == num_plots_positive - 1:
                    total_coupler_in_plot = len(positive_couplers) % 200
                else:
                    total_coupler_in_plot = 200
                colors = cm.gist_rainbow(np.linspace(0, 1, total_coupler_in_plot))

                for ite in range(total_coupler_in_plot):
                    ax[i,j].plot(list_of_coupler_strength[positive_couplers[coupler_ind]], color = colors[ite])
                    coupler_ind += 1
                fig_index += 1
            elif positive_negative_exit == 1:
                if fig_index == num_plots_positive +  num_plots_negative - 1:
                    total_coupler_in_plot = len(negative_couplers) % 200
                else:
                    total_coupler_in_plot = 200
                colors = cm.gist_rainbow(np.linspace(0, 1, total_coupler_in_plot))

                for ite in range(total_coupler_in_plot):
                    ax[i,j].plot(list_of_coupler_strength[negative_couplers[coupler_ind]], color = colors[ite])
                    coupler_ind += 1
                fig_index += 1
            else:
                continue

            if fig_index == num_plots_positive:
                positive_negative_exit = 1
                coupler_ind = 0
            if fig_index >= num_plots_positive + num_plots_negative:
                positive_negative_exit = 2

    pyplot.savefig(working_folder + "/coupler_strength_evolution_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) +"ite.png")


    pyplot.figure(5)         
    fig, ax = pyplot.subplots(nrows=1, ncols=1,figsize=(10, 9))
    fig.suptitle("Anneal Offsets Evolution", fontsize=40)
    colors = cm.gist_rainbow(np.linspace(0, 1, len(list_of_anneal_offset)))
    ind = 0
    for line in list_of_anneal_offset:
        ax.plot(list_of_anneal_offset[line], color = colors[ind], label = "line " + str(line+1))
        ind += 1
    ax.legend(loc='upper right')
    pyplot.savefig(working_folder + "/anneal_offset_evolution_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    triangular_psi, real_psi_tri, imag_psi_tri = get_order_parameter_triangle(running_list, num_embedding, embedding, k_away_from_boundry_omitted)
    # villain_psi,real_psi, imag_psi = get_order_parameter_villain(running_list, k_away_from_boundry_omitted,dim_x, dim_y, num_embedding, embedding, original_coupler_mag)
    villain_psi_sandvik, real_psi_villain, imag_psi_villain = get_order_parameter_villain_sandvik(running_list, k_away_from_boundry_omitted, dim_x, dim_y, num_embedding, embedding)
    square_psi_afm_mag = get_order_parameter_afm_mag(running_list, k_away_from_boundry_omitted, dim_x, dim_y, num_embedding, embedding)

    # fourier, labels = get_fourier_transform(running_list, dim_x, dim_y, num_embedding, embedding, 64)


    print("Average Villain Magnetization Order Parameter:", villain_psi_sandvik[0])
    print("SEM of Villain Magnetization Order Parameter", villain_psi_sandvik[1])
    print("Average Triangular Order Parameter:", triangular_psi[0])
    print("SEM of Triangular Order Parameter", triangular_psi[1])
    print("Average AFM Magnetization Order Parameter:", square_psi_afm_mag[0])
    print("SEM of AFM Magnetization Order Parameter", square_psi_afm_mag[1])

    residual_energy_mean, residual_energy_sem = compute_average_residual_energy(running_list, couplers_per_embedding, ground_state_energy)
    vortices_mean, vortices_sem, vortices_bootstrap_sem,  vortices_std, vortices_bootstrap_std = compute_averge_vortices(running_list, couplers_per_embedding, ground_state_vortices)


    pyplot.figure(6)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (12,8))
    ax.set_facecolor(color='black')
    ax.set_aspect('equal', 'datalim')
    fig.suptitle("Villain Order Parameter 2D Histo " + str(anneal_time_ns) + "ns", fontsize=15)
    h = ax.hist2d(real_psi_villain, imag_psi_villain, bins=43, range=np.array([(-1, 1), (-1, 1)]), norm=mpl.colors.LogNorm(), cmap='magma')     
    fig.colorbar(h[3])  
    pyplot.savefig(working_folder + "/villain_order_2d_histo_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    pyplot.figure(7)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (12,8))
    ax.set_facecolor(color='black')
    ax.set_aspect('equal', 'datalim')
    fig.suptitle("Triangular Order Parameter 2D Histo " + str(anneal_time_ns) + "ns", fontsize=15)
    h = ax.hist2d(real_psi_tri, imag_psi_tri, bins=43, range=np.array([(-1, 1), (-1, 1)]), norm=mpl.colors.LogNorm(), cmap='magma')     
    fig.colorbar(h[3])  
    pyplot.savefig(working_folder + "/triangular_order_2d_histo_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    # pyplot.figure(8)
    # fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (12,8))
    # df_cm = pd.DataFrame(fourier, index = labels, columns = labels)
    # sn.heatmap(df_cm, annot=False, ax=ax, square=True, cmap="magma", cbar=True)
    # fig.suptitle("Fourier Transform " + str(anneal_time_ns) + "ns", fontsize=15)
    # ax.set_xlabel("kx")
    # ax.set_ylabel("ky")

    # pyplot.savefig(working_folder + "/fourier_heatmap_" +  str(anneal_time_ns) +"ns_" +str(num_calib_ite) + "ite.png")

    pyplot.figure(9)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Villain Lattice @ {round(anneal_time_ns,1)}ns", fontsize=30)
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
    pyplot.savefig(working_folder + "/villain_spin_config_1_" +  str(anneal_time_ns) +"ns" + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')

    pyplot.figure(10)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Villain Lattice @ {round(anneal_time_ns,1)}ns", fontsize=30)
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
    pyplot.savefig(working_folder + "/villain_spin_config_2_" +  str(anneal_time_ns) +"ns" + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')

    pyplot.figure(11)
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize = (10,8))
    ax.set_title(f"{dim_x} X {dim_y} Villain Lattice @ {round(anneal_time_ns,1)}ns", fontsize=30)
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
    pyplot.savefig(working_folder + "/villain_spin_config_3_" +  str(anneal_time_ns) +"ns" + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    
    return triangular_psi, villain_psi_sandvik, square_psi_afm_mag, [residual_energy_mean, residual_energy_sem], [vortices_mean, vortices_sem, vortices_bootstrap_sem,  vortices_std, vortices_bootstrap_std]

def create_qpu_input(dim_x, dim_y, input_filename,J_AFM, J_FM):
    input_file = open(input_filename, 'r')
    lines = input_file.readlines()
    num_embedding = len(lines)
    input_file.close()
    embedding = np.zeros((num_embedding, dim_x, dim_y))
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

    couplers = {}
    couplers_per_embedding = []
    for ite in range(num_embedding):
        couplers_in_curr_embedding = {}
        for i in range(dim_x):
            curr_row = embedding[ite, i, :] 
            for m in range(len(curr_row) - 1):
                couplers[(curr_row[m], curr_row[m+1])] = J_AFM
                couplers_in_curr_embedding[(curr_row[m], curr_row[m+1])] = J_AFM
        for i in range(dim_y):
            curr_col = embedding[ite, :, i] 
            if i % 2 == 0:
                for m in range(len(curr_col) - 1):
                    if m % 2 == 0:
                        if i  == 0:
                            couplers[(curr_col[m], curr_col[m+1])] = 0.5 * J_FM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = 0.5 * J_FM
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = 1 * J_FM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = 1 * J_FM
                    else:
                        if i == 0:
                            couplers[(curr_col[m], curr_col[m+1])] = 0.5 *  J_AFM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = 0.5 *  J_AFM
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = J_AFM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = J_AFM
                if i == 0:
                    couplers[(curr_col[-1], curr_col[0])] = 0.5 * J_AFM
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = 0.5 * J_AFM
                else:
                    couplers[(curr_col[-1], curr_col[0])] = J_AFM
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = J_AFM
            else:
                for m in range(len(curr_col) - 1):
                    if m % 2 != 0:
                        if i == dim_y - 1:
                            couplers[(curr_col[m], curr_col[m+1])] = 0.5 * J_FM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = 0.5 * J_FM
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = 1 * J_FM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = 1 * J_FM
                    else:
                        if i == dim_y - 1:
                            couplers[(curr_col[m], curr_col[m+1])] = 0.5 * J_AFM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = 0.5 * J_AFM
                        else:
                            couplers[(curr_col[m], curr_col[m+1])] = J_AFM
                            couplers_in_curr_embedding[(curr_col[m], curr_col[m+1])] = J_AFM
                if i == dim_y - 1:
                    couplers[(curr_col[-1], curr_col[0])] = 0.5 * J_FM
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = 0.5 * J_FM
                else:
                    couplers[(curr_col[-1], curr_col[0])] = 1 * J_FM
                    couplers_in_curr_embedding[(curr_col[-1], curr_col[0])] = 1 * J_FM
        couplers_per_embedding.append(couplers_in_curr_embedding)
    return num_embedding, embedding, qubit_in_use, couplers, couplers_per_embedding

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Villain Model')
    parser.add_argument('-x', metavar='dim_x', type=int,
                        help='x dimension of the square lattice')

    parser.add_argument('-y', metavar='dim_y', type=int,
                        help='y dimension of the square lattice')

    parser.add_argument("-ns", metavar='annealing_time', type=float,
                        help='annealing time in the unit of ns')

    parser.add_argument("-ite", metavar='num_calib_ite', type=int,
                        help='number of iterations for shimming')
    
    parser.add_argument("-FM_scaling", metavar='FM_scaling', type=float)

    args = parser.parse_args()
    dim_x = args.x
    dim_y = args.y
    if dim_x % 3 != 0 and dim_y % 2 != 0 :
        print("Dimension not supported")
        exit(1)
    annealing_time_ns = args.ns
    num_calib_ite = args.ite
    FM_scaling = args.FM_scaling
    output_filename = f'{dim_x}_{dim_y}'

    if FM_scaling > 0:
        print("")
        print(Fore.RED + "Your model is an AFM Unfrustrated Square Lattice, please cancel if it is not desired.")
        print(Style.RESET_ALL)


    input_filename = './input/bay8_square_cylinder_multiple_txt_' + str(dim_x).zfill(2) + 'x' + str(dim_y).zfill(2) + '.txt'

    J_AFM = -1.9
    J_FM = FM_scaling * J_AFM

    
    
    num_embedding, embedding, qubit_in_use, couplers, couplers_per_embedding = create_qpu_input(dim_x, dim_y, input_filename,J_AFM, J_FM)
    k_away_from_boundry_omitted = 0

    samples_used_for_analysis = 50000

    triangular_psi, villain_psi_sandvik, square_psi_afm_mag, residual_energy, vortices = qpu_run(qubit_in_use = qubit_in_use, couplers = couplers, J_AFM=J_AFM, J_FM = J_FM,num_calib_ite = num_calib_ite, 
            dim_x = dim_x, dim_y = dim_y,  couplers_per_embedding=couplers_per_embedding,
            num_embedding = num_embedding, embedding = embedding, 
            flux_step_size = 2e-6, corr_step_size = 0.005, anneal_offset_step_size = 0.001, 
            coupler_damp = 0, anneal_offset_damp = 0, num_reads_per_call = 100, anneal_time_ns = annealing_time_ns, 
            corr_buckets = get_buckets(dim_x,dim_y, num_embedding, embedding), k_away_from_boundry_omitted = k_away_from_boundry_omitted,
            samples_used_for_analysis = samples_used_for_analysis)
    # input("Press Enter to continue...")

    working_folder = "./villain_" + str(dim_x) + "_" + str(dim_y) + "dim_" +str(round(J_FM/J_AFM, 2))

    if output_filename == 'null':
        exit(0)
    input("Press Enter to continue...")

    out_file = open(working_folder + '/villain_' + output_filename + '_boundary_' + str(k_away_from_boundry_omitted) + '.csv', 'a')

    if os.stat(working_folder + '/villain_' + output_filename + '_boundary_' + str(k_away_from_boundry_omitted) + '.csv').st_size == 0:
        out_file.write(f'annealing_time,average_triangular_op,triangular_op_sem,triangular_op_bootstrap_sem,triangular_op_std,triangular_op_bootstrap_std,average_villain_op,villain_op_sem,villain_op_bootstrap_sem,villain_op_std,villain_op_bootstrap_std,average_afm_op,afm_op_sem,afm_op_bootstrap_sem,afm_op_std,afm_op_bootstrap_std,average_vortices,vortices_sem,vortices_bootstrap_sem,vortices_std,vortices_bootstrap_std\n')

    out_file.write(str(annealing_time_ns))

    for s in triangular_psi:
        out_file.write(',' + str(s))
    
    for s in villain_psi_sandvik:
        out_file.write(',' + str(s))

    for s in square_psi_afm_mag:
        out_file.write(',' + str(s))

    for s in vortices:
        out_file.write(',' + str(s))

    out_file.write('\n')
    out_file.close()
    
    