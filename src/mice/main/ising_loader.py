import os
import re
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import h5py
from tqdm import tqdm
import numba

@numba.jit(nopython=True)
def bits_array_to_bytes_array(bits_array):
    bytes_array_shape = (bits_array.shape[0], bits_array.shape[1], bits_array.shape[2]*8)
    bytes_array = np.zeros(bytes_array_shape, dtype=np.uint8)
    for i in range(bits_array.shape[0]):
        for j in range(bits_array.shape[1]):
            for k in range(bits_array.shape[2]):
                for bit_idx in range(8):
                    byte = bits_array[i,j,k]
                    bytes_array[i,j,k*8 + bit_idx] = np.uint8(byte & (np.uint8(1) << bit_idx) != 0)
    return bytes_array

def file_to_ising_array(filename):
    size = 64
    frames = 10000
    all_spins = np.fromfile(filename, dtype=np.uint8)
    return bits_array_to_bytes_array(np.reshape(all_spins, (frames, size, int(size/8))))

def get_ising_matrices(T, directory):
    for filename in sorted(os.listdir(directory)):
        sim_T = float(re.findall("\d+\.\d+", filename)[0])
        if sim_T == T:
            ising_matrices = file_to_ising_array(os.path.join(directory,filename))
            return ising_matrices

# This is not an efficient implementation, but it is good enough
def get_all_sub_matrices(matrices, sub_shape, step=1):
    if len(matrices.shape) != 3:
        raise Exception('Works only for 3d right now')
    if(len(sub_shape) != 2):
        raise Exception('Works only for 3d right now')
        
    all_subs = []
    max_i = matrices.shape[1] - sub_shape[0] + 1
    max_j = matrices.shape[2] - sub_shape[1] + 1
    for i in range(0, max_i, step):
        for j in range(0, max_j, step):
            sub_matrix = matrices[:, i:i+sub_shape[0], j:j+sub_shape[1]]
            all_subs.append(sub_matrix)
    return np.concatenate(all_subs, axis=0)

def runner_loader(T):
    directory = os.path.join('./', 'ising_data')
    ising_matrices = get_ising_matrices(T, directory)
    ising_matrices = np.array(ising_matrices, dtype=np.float32)
    ising_matrices -= 0.5
    ising_matrices *= 2
    ising_matrices = np.concatenate([ising_matrices, np.flip(ising_matrices, axis=1), np.flip(ising_matrices, axis=2)])
    print(ising_matrices.shape)
    subsystem_shape = (16,16)
    ising_subsystems = get_all_sub_matrices(ising_matrices, subsystem_shape, step=10)
    print(ising_subsystems.shape)
    with h5py.File(os.path.join('./', 'ising_h5py', f'ising_data_64_{T}.h5'), 'w') as f:
        f.create_dataset('dataset_1', data=ising_subsystems)
    from sklearn.model_selection import train_test_split 
    train_data, test_data = train_test_split(ising_subsystems)
    print(train_data.shape)

if __name__ == '__main__':
    directory = os.path.join('./', 'ising_data')
    Ts_x = [round(T,2) for T in np.linspace(0.1, 4, 40)]
    Ts_y = [0.1, 1, 2, 2.1, 2.2, 2.3, 2.4, 2.7, 2.9, 3, 3.2, 4]
    Ts = [i for i in Ts_x if i not in Ts_y]
    for T in tqdm(Ts):
        runner_loader(T)
    
    
    
