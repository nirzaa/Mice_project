import os
import numpy as np
import gin
import mice
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numba
from itertools import combinations_with_replacement
import math
import wandb
import argparse
import json

def sweep_wandb_ising(num_boxes, idx, comb, number_combinations):
    lr = 1e-3
    batch_size = 32
    T = np.linspace(0.1, 4, 40)
    T_list = list()
    for t in T:
        if t.is_integer():
            t = int(t)
        else:
            t = round(t, 2)
            t = float(t)
        T_list.append(t)
    print('The temperatures are:')
    print(T_list)
    wandb.login()

    sweep_config = {
    'method': 'grid'
    }
    metric = {
    'name': 'valid_losses_exp',
    'goal': 'maximize'   
    }
    sweep_config['metric'] = metric
    parameters_dict = {
    'lr': {
          'values': [lr/4]
        },
    'batch_size': {
        'values': [batch_size]
        },
    'Temperature': {
        'values': T_list
        },
    }
    
    sweep_config['parameters'] = parameters_dict
    parameters_dict.update({
    'num_boxes': {
        'value': num_boxes},
    'idx': {
        'value': idx},
    'comb': {
        'value': comb},
    'number_combinations': {
        'value': number_combinations},
    })

    sweep_id = wandb.sweep(sweep_config, project="mice project ising sand")
    wandb.agent(sweep_id, sweep_ising_run)

@gin.configurable
def sweep_ising_run(max_epochs, freq_print, genom, weight_decay, num_samples, transfer_epochs, window_size=3, config=None):
    '''
    Running the neural network in order to calculate the right number of boxes to split our space into

    box_sizes: the number of boxes in each axis to split our space into
    max_epochs: the maximum number of epochs to use in the beginning
    batch_size: the size of the batch
    freq_print: the number of epochs between printing to the user the mutual information
    axis: the axis we will split our boxes into, in order to calculate the mutual information
    genom: the type of architecture we are going to use in the neural net
    lr: the learning rate
    weight_decay: regularization technique by adding a small penalty
    box_frac: what is the value of the box from the total space we are calculating the mutual information to

    return:
    None
    '''
    with wandb.init(project="mice project ising sand", config=config):
        config = wandb.config
        lr, batch_size = config.lr, config.batch_size
        T, num_boxes, idx, comb, number_combinations = config.Temperature, config.num_boxes, config.idx, config.comb, config.number_combinations

        weights_path = os.path.join('./', 'src', 'model_weights')
        PATH = os.path.join(weights_path, genom+'_model_weights.pth')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        R = np.random.RandomState(seed=0)
        
        cntr = 0
        mi_entropy_dependant = []
        mi_entropy_dependant_valid = []
        
        x_labels = []
        saved_directory = os.path.join('ising_h5py')

        if idx == 0: n_epochs = max_epochs
        elif idx != 0: n_epochs = transfer_epochs
        
        i, j, k = comb
        sizes = (i, j, k)
        with h5py.File(os.path.join(saved_directory, f'ising_data_64_{T}.h5'), "r") as hf:
            lattices = np.array(hf.get('dataset_1'))
            lattices = np.expand_dims(lattices, axis=1)
        
        list_ising = lattices.copy()
        x_size = list_ising[0].shape[0]
        y_size = list_ising[0].shape[1]
        z_size = list_ising[0].shape[2]
        input_size = int(8 * 1 * ((y_size-2)/1+1) * ((z_size-2)/1+1))
        model = mice.mi_model(genom=genom, n_epochs=n_epochs, max_epochs=max_epochs, input_size=input_size)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = mice.LRScheduler(optimizer)
        early_stopping = mice.EarlyStopping()
        train_losses = []
        valid_losses = []
        axis = int(np.argmax((i, j, k)))
        print('='*50)
        print(f'The size of the small boxes is: {i}x{j}x{k}\n'
                f'Therefore we cut on the {axis} axis\n'
                f'Building the boxes... we are going to start training...')
        epochs = (n_epochs // (cntr+1))
        epochs = int(np.ceil((n_epochs * 2) // ((i*j*k)**(1/3))))
        epochs = max(epochs, 1)
        model = mice.mi_model(genom=genom, n_epochs=n_epochs, max_epochs=max_epochs, input_size=input_size)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses = []
        valid_losses = []
        wandb.watch(model, mice.my_criterion, log="all", log_freq=1000)
        for epoch in tqdm(range(int(n_epochs))):

            place = random.sample(range(len(list_ising)), k=int(num_samples))
            lattices = np.array(list_ising)[place]
            left_lattices, right_lattices = mice.lattice_splitter(lattices=lattices, axis=axis)
            joint_lattices = np.concatenate((left_lattices, right_lattices), axis=axis + 1)
            right_lattices_random = right_lattices.copy()
            R.shuffle(right_lattices_random)
            product_lattices = np.concatenate((left_lattices, right_lattices_random), axis=axis + 1)
            joint_lattices, joint_valid, product_lattices, product_valid = train_test_split(joint_lattices, product_lattices,
                                                                                            test_size=0.2, random_state=42)
            AB_joint, AB_product = torch.tensor(joint_lattices), torch.tensor(product_lattices)
            AB_joint_train, AB_product_train = AB_joint.to(device), AB_product.to(device)
            dataset_train = mice.MiceDataset(x_joint=AB_joint_train, x_product=AB_product_train)

            AB_joint, AB_product = torch.tensor(joint_valid), torch.tensor(product_valid)
            AB_joint_valid, AB_product_valid = AB_joint.to(device), AB_product.to(device)
            dataset_valid = mice.MiceDataset(x_joint=AB_joint_valid, x_product=AB_product_valid)

            loader = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=0, shuffle=False)
            loss_train, mutual_train = mice.train_one_epoch(window_size=window_size, epoch=epoch, train_losses=train_losses, model=model, data_loader=loader, optimizer=optimizer)
            train_losses.append(mutual_train.cpu().detach().numpy())

            loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, num_workers=0, shuffle=False)
            valid_loss, valid_mutual = mice.valid_one_epoch(window_size=window_size, epoch=epoch, valid_losses=valid_losses, model=model, data_loader=loader)
            valid_losses.append(valid_mutual.cpu().detach().numpy())

            train_losses_exp = list(mice.exp_ave(data=train_losses))
            valid_losses_exp = list(mice.exp_ave(data=valid_losses))
            train_losses_exp = list(mice.exp_ave(data=train_losses_exp))
            valid_losses_exp = list(mice.exp_ave(data=valid_losses_exp))

            if epoch > 1e3:
                early_stopping(train_losses_exp[-1])
                if early_stopping.early_stop:
                    break
                elif epoch > 2000 and train_losses_exp[-1] < 1e-6:
                    break
            if epoch > 1e5:
                lr = lr_scheduler(train_losses_exp[-1]).param_groups[0]["lr"]
                early_stopping(train_losses_exp[-1])
                if early_stopping.early_stop:
                    break
                elif epoch > 2000 and train_losses_exp[-1] < 1e-6:
                    break
            
            if epoch % freq_print == 0:
                print(f'\nMI for train {train_losses_exp[-1]}, val {valid_losses_exp[-1]} at step {epoch}')
                wandb.log({"epoch": epoch, "train loss": train_losses_exp[-1], "valid loss": valid_losses_exp[-1],
                "learning rate":lr, "batch size": batch_size, "Temperature": T})
        
        cntr += 1
        x_labels.append(str((i, j, k)))
        torch.save(model.state_dict(), PATH)

        wandb.save("model.onnx")

        return None

        

def run_sweep_ising():
    num_boxes = 16
    limit = 5096
    
    my_root = int(np.floor(np.log2(num_boxes)))
    temporal_combinations = list(combinations_with_replacement([2 << expo for expo in range(0, my_root)], 3))
    temporal_combinations.sort(key=lambda x: math.prod(x))
    print('Our combinations are:')
    my_combinations = list()
    my_combinations.append((1,10,10))
    mice.print_combinations(my_combinations)
    number_combinations = len(my_combinations)
    x_labels = []
    mi_entropy_dependant = []
    mi_entropy_dependant_valid = []
    for idx, (i, j, k) in enumerate(my_combinations):
        x_labels.append(str((i, j, k)))
        comb = (i, j, k)
        mice.sweep_wandb_ising(num_boxes=num_boxes, idx=idx, comb=comb, number_combinations=number_combinations)

def input_func_ising():
    parser = argparse.ArgumentParser(description='input_data')
    parser.add_argument('--T', type=int, default=400, metavar='N', help='The temperature')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = mice.input_func()
    T = args.T
    # print(f'Working on T = {T}')
    # mice.sand()
    mice.run_sweep_ising()
    
    # mice.run_sweep(T=T)
