import numpy as np
import h5py
import os
import gin
import sys
from tqdm import tqdm
import mice
import argparse

@gin.configurable
def file_load(file_name, number_lines):
    '''
    This function will take the dump.0 file that we will get by running the simulation,
    and will generate an .h5 file for future processing of the data.

    number_lines: the number of lines to read from the `data.h5` file

    return:
    None
    '''
    path_data = os.path.join('./', 'data')

    with open(os.path.join(path_data, file_name, 'dump.0')) as f:
        print('Reading the lines from the file...')
        lines = [f.readline() for x in tqdm(range(number_lines))]
    f.close()

    mod265 = np.arange(len(lines)) % 265
    blocks = np.array(lines)[(mod265 <= 264) & (mod265 >= 9)]
    blocks = blocks[:-1]
    print('Constructing the coordinates of the particles...')
    blocks = np.array([np.array(x.split()[2:], dtype=float) for x in tqdm(blocks)])
    
    mod265 = np.arange(len(lines)) % 265
    names = np.array(lines)[(mod265 <= 264) & (mod265 >= 9)]
    names = names[:-1]
    print('Constructing the names of the particles...')
    names = np.array([np.array(x.split()[:1], dtype=float) for x in tqdm(names)])
    
    leny = len(blocks)
    leny_mod = leny // 256
    leny_mod = leny_mod * 256
    blocks = blocks[:leny_mod].reshape((-1, 256, 3))
    blocks = blocks.astype(float)

    leny = len(names)
    leny_mod = leny // 256
    leny_mod = leny_mod * 256
    names = names[:leny_mod].reshape((-1, 256))
    names = names.astype(int)

    with h5py.File(os.path.join(path_data, file_name, 'names.h5'), 'w') as hf:
        hf.create_dataset('dataset_1', data=names)
    hf = h5py.File(os.path.join(path_data, file_name, 'data.h5'), 'w')
    hf.create_dataset('dataset_1', data=blocks)
    hf.close()
     
    return None

if __name__ == '__main__':
    '''
    This module will take the dump.0 file that we will get by running the simulation,
    and will generate an .h5 file for future processing of the data.
    '''
    parser = argparse.ArgumentParser(description='input_data')
    parser.add_argument('--T', type=int, default=400, metavar='N', help='The temperature')
    args = parser.parse_args()
    T = args.T
    print(f'Working on T = {T}')
    file_name = os.path.join('dump_files', 'Aluminum', 'pure_liquid', str(T))
    mice.file_load(file_name=file_name)
    