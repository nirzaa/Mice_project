import torch
import numpy as np
import gin
import mice
from tqdm import tqdm

def ising_creator(dims: np.array, R) -> np.array:
    ''''
    create a tensor of shape (num_tensors, J, dims)
    '''
    ising_tensor = (R.choice((-1, 1), size=np.prod(dims))).reshape(dims)
    return ising_tensor

@gin.configurable
def ising_runner(num_isings, dims, J, kT, R):
    list_ising = []
    ising_tensor = ising_creator(dims , R)
    for index, i in enumerate(tqdm(range(int(num_isings)))):
        i = R.choice(range(ising_tensor.shape[0]), size=1, replace=True)
        j = R.choice(range(ising_tensor.shape[1]), size=1, replace=True)
        k = R.choice(range(ising_tensor.shape[2]), size=1, replace=True)
        temp_ising_tensor = ising_tensor.copy()
        if ising_tensor.shape[0] == 1:
            energy = (ising_tensor * J * (
                                np.roll(ising_tensor, shift=1, axis=1) +
                                np.roll(ising_tensor, shift=-1, axis=1) + 
                                np.roll(ising_tensor, shift=1, axis=2) + 
                                np.roll(ising_tensor, shift=-1, axis=2)
                                ))[i,j,k]
            if ising_tensor[i,j,k] == 1: temp_ising_tensor[i,j,k] = -1
            elif ising_tensor[i,j,k] == -1: temp_ising_tensor[i,j,k] = 1
            temp_energy = (temp_ising_tensor * J * ( +
                                np.roll(temp_ising_tensor, shift=1, axis=1) +
                                np.roll(temp_ising_tensor, shift=-1, axis=1) +
                                np.roll(temp_ising_tensor, shift=1, axis=2) + 
                                np.roll(temp_ising_tensor, shift=-1, axis=2)
                                ))[i,j,k]
        elif ising_tensor.shape[0] != 1:
            energy = (ising_tensor * J * (
                                np.roll(ising_tensor, shift=1, axis=0) +
                                np.roll(ising_tensor, shift=-1, axis=0) +
                                np.roll(ising_tensor, shift=1, axis=1) +
                                np.roll(ising_tensor, shift=-1, axis=1) + 
                                np.roll(ising_tensor, shift=1, axis=2) + 
                                np.roll(ising_tensor, shift=-1, axis=2)
                                ))[i,j,k]
            i = R.choice(range(ising_tensor.shape[0]), size=1, replace=True)
            j = R.choice(range(ising_tensor.shape[1]), size=1, replace=True)
            k = R.choice(range(ising_tensor.shape[2]), size=1, replace=True)
            if ising_tensor[i,j,k] == 1: temp_ising_tensor[i,j,k] = -1
            elif ising_tensor[i,j,k] == -1: temp_ising_tensor[i,j,k] = 1
            temp_energy = (temp_ising_tensor * J * (
                                np.roll(temp_ising_tensor, shift=1, axis=0) +
                                np.roll(temp_ising_tensor, shift=-1, axis=0) +
                                np.roll(temp_ising_tensor, shift=1, axis=1) +
                                np.roll(temp_ising_tensor, shift=-1, axis=1) +
                                np.roll(temp_ising_tensor, shift=1, axis=2) + 
                                np.roll(temp_ising_tensor, shift=-1, axis=2)
                                ))[i,j,k]
        
        if energy > temp_energy:
            ising_tensor[i, j, k] = temp_ising_tensor[i,j,k]
        elif energy < temp_energy:
            if R.rand() < np.exp(-(temp_energy - energy) / kT):
                ising_tensor[i, j, k] = temp_ising_tensor[i,j,k]
        if index % 700 == 0:
            ising_tensor_appender = ising_tensor.copy()
            # ising_tensor_appender[ising_tensor_appender == -1] = 0
            list_ising.append(ising_tensor_appender)
    return list_ising

def part_lattices(lattices, x_size, y_size, z_size, R):
    part_lattices = []
    xleny = lattices[0].shape[0]
    yleny = lattices[0].shape[1]
    zleny = lattices[0].shape[2]
    x_steps = xleny - x_size
    y_steps = yleny - y_size
    z_steps = zleny - z_size
    for lat in lattices:
        if x_steps == 0:
            i = 0
        else:
            i = R.randint(0, x_steps+1)
        if y_steps == 0:
            j = 0
        else:
            j = R.randint(0, y_steps+1)
        if z_steps == 0:
            k = 0
        else:
            k = R.randint(0, z_steps+1)

        part_lattices.append(lat[i:i+x_size, j:j+y_size, k:k + z_size])
    
    return part_lattices
    


if __name__ == '__main__':
    x = mice.ising_runner(kT = 2)
    print('Done!')
    
    
    
    
