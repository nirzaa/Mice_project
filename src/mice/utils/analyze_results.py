import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exp_ave(data, window_frac=0.1):
    data = np.array(data)
    window = np.floor(data.shape[0] * window_frac).astype(int)
    ave_arr = np.zeros((data.shape[0]))
    mi = data[0]
    alpha = 2 / float(window + 1)
    for i in range(data.shape[0]):
        mi =  ((data[i] - mi) * alpha) + mi
        ave_arr[i] = mi
    return ave_arr

df_slow = pd.read_csv(os.path.join('csv_files', 'al_liquid_slow.csv'))
df_fast = pd.read_csv(os.path.join('csv_files', 'al_liquid_fast.csv'))
data_list = list()
col_list = list()
for col in df_fast.columns:
    if col.isnumeric():
        col_list.append(col)
        data = np.array(df_fast[col])
        data = exp_ave(data, window_frac=0.1)
        data_list.append(data)
data_tensor = np.stack(data_list, axis=0)
df_fast_smooth = pd.DataFrame(data_tensor.T, columns=col_list)
df_fast_smooth.reset_index(inplace=True)
df_fast_smooth = df_fast_smooth.rename(columns = {'index':'Step'})
plt.figure(num=0, figsize=(12, 6))
for num, col in enumerate(df_fast.columns):
    if col.isnumeric():
        plt.figure(num=num, figsize=(12, 6))
        sns.set_style("darkgrid")
        plt.clf()
        plt.plot(df_fast_smooth['Step']*100, df_fast_smooth[col], label=f'fast smooth, T={col}', linewidth=3, linestyle='--')
        plt.plot(df_slow['Step']*100, df_slow[col], label=f'slow, T={col}', linewidth=2, linestyle='-.')
        plt.xlabel('Epoch')
        plt.ylabel('mi')
        plt.title('mutual information as a function of epoch')
        plt.legend()
        plt.savefig(os.path.join('figures', col))
plt.figure(num=0, figsize=(12, 6))
plt.clf()
for num, col in enumerate(df_fast.columns):
    if col.isnumeric():
        sns.set_style("darkgrid")
        plt.plot(df_fast['Step']*100, df_fast[col], label=f'fast, T={col}', linewidth=4)
        plt.plot(df_fast_smooth['Step']*100, df_fast_smooth[col], label=f'fast smooth, T={col}', linewidth=3, linestyle='--')
        plt.plot(df_slow['Step']*100, df_slow[col], label=f'slow, T={col}', linewidth=2, linestyle='-.')
        plt.xlabel('Step')
        plt.ylabel('mi')
        plt.title('mutual information as a function of epoch')
        plt.legend()
        plt.savefig(os.path.join('figures', 'together'))
print()