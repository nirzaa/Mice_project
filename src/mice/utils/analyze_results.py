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

# def exp_ave(data, loss=None):
#     temp_data = data.copy()
#     if loss is not None:
#         temp_data.append(np.array(loss.detach().cpu()))
#     temp_data = np.array(temp_data, dtype='float64')
#     if len(temp_data) > 100:
#         temp_data = temp_data[-20:]
#         weights = torch.tensor([np.power(i, 2) for i in range(len(temp_data))], dtype=torch.float64)
#         temp_data = torch.tensor(temp_data, dtype=torch.float64, requires_grad=True)
#         ave_arr = torch.matmul(temp_data, weights) / weights.sum()
#         return ave_arr
#     elif len(temp_data) <= 100:
#         return loss

def mutual_as_epoch():
    df_slow = pd.read_csv(os.path.join('csv_files', '12_6.csv'))
    df_fast = pd.read_csv(os.path.join('csv_files', '16_8.csv'))
    data_list = list()
    col_list = list()
    for col in df_fast.columns:
        if col.isnumeric():
            col_list.append(col)
            data = np.array(df_fast[col])
            data = exp_ave(data)
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
    for cntr, col in enumerate(df_fast.columns):
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

def mi_as_temp():
    dirs = ['mi_vs_discrete', 'mi_vs_epoch', 'mi_vs_T']
    for dir in dirs:
        path = os.path.join('figures', dir)
        if not os.path.exists(path):
            os.makedirs(path)
    nums = [6,8,10,12,14,16]
    temps = [900,850,800,750,700]
    all_mi_list = list()
    for num in nums:
        df = pd.read_csv(os.path.join('csv_files', f'{num*2}_{num}.csv'))
        data_list = list()
        col_list = list()
        for col in df.columns:
            if col.isnumeric():
                col_list.append(col)
                data = np.array(df[col])
                data = exp_ave(data)
                data_list.append(data)
        data_tensor = np.stack(data_list, axis=0)
        df = pd.DataFrame(data_tensor.T, columns=col_list)
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'Step'})
        plt.figure(num=num, figsize=(12, 6))
        plt.clf()
        sns.set_style("darkgrid")
        for cntr, col in enumerate(df.columns):
            if col.isnumeric():
                plt.plot(df['Step']*100, df[col], label=f'{num*2}_{num}_{col}', linewidth=3, linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('mi')
        plt.title('mutual information as a function of epoch')
        plt.legend()
        plt.savefig(os.path.join('figures', 'mi_vs_epoch', f'{num*2}_{num}'))
        mi_list = list()
        temp_list = list()
        for cntr, col in enumerate(df.columns):
            if col.isnumeric():
                mi_list.append(df[col].iloc[len(df[col]) - 1:])
                temp_list.append(col)
        all_mi_list.append(mi_list)
        plt.figure(num=0, figsize=(12, 6))
        sns.set_style("darkgrid")
        plt.clf()
        plt.plot(temp_list, mi_list, label=f'{num*2}_{num}', linewidth=3, linestyle='--')
        plt.xlabel('Temperature')
        plt.ylabel('mi')
        plt.title('mutual information as a function of temperature')
        plt.legend()
        plt.savefig(os.path.join('figures', 'mi_vs_T', f'{num*2}_{num}'))
    all_mi = np.stack(all_mi_list, axis=0)
    plt.figure(num=0, figsize=(12, 6))
    sns.set_style("darkgrid")
    plt.clf()
    for cntr, num in enumerate(nums):
        plt.plot(temp_list, all_mi[cntr], label=f'{num*2}_{num}', linewidth=3, linestyle='--')
    plt.xlabel('Temperature')
    plt.ylabel('mi')
    plt.title('mutual information as a function of temperature')
    plt.legend()
    plt.savefig(os.path.join('figures', 'mi_vs_T', f'together'))

    all_mi = all_mi.reshape(all_mi.shape[0], all_mi.shape[1])
    all_mi = all_mi.T

    plt.figure(num=0, figsize=(12, 6))
    sns.set_style("darkgrid")
    for cntr, num in enumerate(range(all_mi.shape[0])):
        plt.clf()
        plt.plot(nums, all_mi[cntr], label=f'{temps[cntr]}', linewidth=3, linestyle='--')
        plt.xlabel('discrete')
        plt.ylabel('mi')
        plt.title('mutual information as a function of discrete')
        plt.legend()
        plt.savefig(os.path.join('figures', 'mi_vs_discrete', f'{temps[cntr]}'))

    plt.figure(num=0, figsize=(12, 6))
    sns.set_style("darkgrid")
    plt.clf()
    for cntr, num in enumerate(range(all_mi.shape[0])):
        plt.plot(nums, all_mi[cntr], label=f'{temps[cntr]}', linewidth=3, linestyle='--')
    plt.xlabel('discrete')
    plt.ylabel('mi')
    plt.title('mutual information as a function of discrete')
    plt.legend()
    plt.savefig(os.path.join('figures', 'mi_vs_discrete', f'together'))

def experiment_exp_ave():
    df = pd.read_csv(os.path.join('csv_files', f'12_6.csv'))
    epochs = df['Step']
    df = df['900']
    df_5 = pd.read_csv(os.path.join('csv_files', f'12_6_5expave.csv'))
    epochs_5 = df_5['Step']
    df_5 = df_5['900']
    # df = exp_ave(df)
    # df_5 = exp_ave(df_5)
    plt.figure(num=0, figsize=(12, 6))
    sns.set_style("darkgrid")
    plt.clf()
    plt.plot(epochs, df, label=f'looking 20 back exp ave', linewidth=3, linestyle='--')
    plt.plot(epochs_5, df_5, label=f'looking 5 back exp ave', linewidth=3, linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('mi')
    plt.title('mutual information as a function of epoch')
    plt.legend()
    plt.show()
    return None

if __name__ == '__main__':
    print('Starting...')
    # mi_as_temp()
    experiment_exp_ave()
