import mice
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np
import os
import h5py

def dist_me():
    dists_list = []
    sample = 0
    blocks = mice.read_data()
    df_particles = pd.DataFrame(
                blocks[sample],
                columns=['X', 'Y', 'Z']
            )
    m = df_particles.shape[0]
    Y = pdist(df_particles, 'euclidean')
    for i in range(256):
        dists = np.array([Y[m * i + j - ((i + 2) * (i + 1)) // 2] for j in range(256) if j != i])
        dists_list.append(dists.min())
        
    dists_list = np.array(dists_list)
    Y_mean = dists_list.mean()
    Y_std = dists_list.std()
    return Y_mean, Y_std

def fluctuation_me():
    sample_1 = 0
    sample_2 = 1
    path_data = os.path.join('./', 'data')
    with h5py.File(os.path.join(path_data, 'names.h5'), 'r') as hf:
        names = np.array(hf.get('dataset_1'))
    blocks = mice.read_data()
    df_particles_1 = pd.DataFrame(
                blocks[sample_1],
                columns=['X', 'Y', 'Z']
            )
    df_particles_2 = pd.DataFrame(
                blocks[sample_2],
                columns=['X', 'Y', 'Z']
            )
    names_1 = names[sample_1]
    names_2 = names[sample_2]
    places = np.where(names_1.reshape(names_1.size, 1) == names_2)[1]
    new_list = [x for x in range(1, 10) if x % 2 == 0]
    dists = np.array([np.linalg.norm(df_particles_1.iloc[i] - df_particles_2.iloc[places[i]]) for i in range(256) if np.linalg.norm(df_particles_1.iloc[i] - df_particles_2.iloc[places[i]]) < 0.8])
    
    Y_mean = dists.mean()
    Y_std = dists.std()
    return Y_mean, Y_std


if __name__ == '__main__':
    dist_mean, dist_std = dist_me()
    print(f'The mean and std for distance to other particles:\nmean: {dist_mean}, std: {dist_std}')

    dist_mean, dist_std = fluctuation_me()
    print(f'The mean and std for distance to fluctuation of same particle:\nmean: {dist_mean}, std: {dist_std}')
