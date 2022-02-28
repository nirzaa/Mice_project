import os
import matplotlib.pyplot as plt
import numpy as np

def all_together():
    meessages_path = os.path.join('./', 'figures', 'losses', 'ising')
    models_folder = ['mice_conv', 'sandnet']
    plt.figure(num=0, figsize=(12, 6))
    plt.clf()
    lines_num = np.linspace(3, 81, 40, dtype='int')
    for model in models_folder:
        temp = []
        mi = []
        read_path = os.path.join(meessages_path, model, 'message_isingcalc.log')
        with open(read_path) as f:
            content = f.readlines()
            for i in lines_num:
                temp.append(float(content[i].split()[-3]))
                mi.append(float(content[i].split()[-1]))
        plt.scatter(temp, mi, label=model)

    temp = np.linspace(0.1, 4, 40)
    mi = [0.75, 0.64, 0.78, 0.68, 0.72, 0.67, 0.78, 0.53, 0.704, 0.83, 0.63, 0.73, 0.8, 0.75, 0.76, 0.68, 0.607, 0.75, 0.701, 0.722, 1.13, 1.13, 1.194, 0.89, 1.29, 1.26, 1.512, 1.05, 1.01, 1.04, 0.78, 0.72, 0.616, 0.66, 0.77, 0.665, 0.443, 0.586, 0.441, 0.367]
    plt.scatter(temp, mi, label='amit results')
    plt.title('MI as a function of T for 64x64 ising model with 16x16 subsystems')
    plt.xlabel('Temperature')
    plt.ylabel('mutual information')
    plt.legend()
    plt.savefig(os.path.join(meessages_path, 'together'))

def sub_figures():
    meessages_path = os.path.join('./', 'figures', 'losses', 'ising')
    models_folder = ['mice_conv', 'sandnet']
    # plt.figure(num=0, figsize=(12, 6))
    # plt.clf()
    f, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    f.suptitle('MI as a function of T for 64x64 ising model with 16x16 subsystems', fontsize=20)
    lines_num = np.linspace(3, 81, 40, dtype='int')
    for cntr, model in enumerate(models_folder):
        temp = []
        mi = []
        read_path = os.path.join(meessages_path, model, 'message_isingcalc.log')
        with open(read_path) as f:
            content = f.readlines()
            for i in lines_num:
                temp.append(float(content[i].split()[-3]))
                mi.append(float(content[i].split()[-1]))
        axes[0,cntr].scatter(temp, mi, label=model)
        axes[1,1].scatter(temp, mi, label=model)

    temp = np.linspace(0.1, 4, 40)
    mi = [0.75, 0.64, 0.78, 0.68, 0.72, 0.67, 0.78, 0.53, 0.704, 0.83, 0.63, 0.73, 0.8, 0.75, 0.76, 0.68, 0.607, 0.75, 0.701, 0.722, 1.13, 1.13, 1.194, 0.89, 1.29, 1.26, 1.512, 1.05, 1.01, 1.04, 0.78, 0.72, 0.616, 0.66, 0.77, 0.665, 0.443, 0.586, 0.441, 0.367]
    axes[1,0].scatter(temp, mi, label='amit results')
    # axes[1,1].scatter(temp, mi, label='amit results')
    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlabel('Temperature')
            axes[i,j].set_ylabel('Mutual information')
            axes[i,j].legend()
    plt.savefig(os.path.join(meessages_path, 'together_sub'))


if __name__ == '__main__':
    all_together()
    sub_figures()
