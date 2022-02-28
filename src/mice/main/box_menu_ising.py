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

@gin.configurable
def ising_box_runner(idx, T, max_epochs, batch_size, freq_print, genom, lr, weight_decay, num_samples, transfer_epochs, window_size=3):
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
    if idx == 0: n_epochs = max_epochs
    elif idx != 0: n_epochs = transfer_epochs

    weights_path = os.path.join('./', 'src', 'model_weights')
    PATH = os.path.join(weights_path, genom+'_model_weights.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R = np.random.RandomState(seed=0)
    mi_num_box_dependant = []
    mi_num_box_dependant_valid = []
        
    # list_ising = mice.ising_runner(kT=kT, R=R)
    with h5py.File(os.path.join('./', 'ising_h5py', f'ising_data_64_{T}.h5'), 'r') as f:
        n1 = np.array(f.get('dataset_1'))
        n1 = np.expand_dims(n1, axis=1)
        list_ising = list(n1)
    x_size = list_ising[0].shape[0]
    y_size = list_ising[0].shape[1]
    z_size = list_ising[0].shape[2]
    input_size = x_size * y_size * z_size
    model = mice.mi_model(genom=genom, n_epochs=n_epochs, max_epochs=max_epochs, input_size=batch_size*18)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = mice.LRScheduler(optimizer)
    early_stopping = mice.EarlyStopping()
    train_losses = []
    valid_losses = []
    axis = int(np.argmax((list_ising[0].shape[0], list_ising[0].shape[1], list_ising[0].shape[2])))
    axis = int(np.argmax((list_ising[0].shape[0], list_ising[0].shape[1])))
    saver = 0
    cntr = 0
    for epoch in tqdm(range(int(n_epochs))):
        place = random.sample(range(len(list_ising)), k=int(num_samples))
        lattices = np.array(list_ising)[place]
        # lattices = mice.part_lattices(lattices, x_size, y_size, z_size, R)
        left_lattices, right_lattices = mice.lattice_splitter(lattices=lattices, axis=axis)
        joint_lattices = np.concatenate((left_lattices, right_lattices), axis=axis + 1)
        right_lattices_random = right_lattices.copy()
        R.shuffle(right_lattices_random)
        product_lattices = np.concatenate((left_lattices, right_lattices_random), axis=axis + 1)
        # joint_lattices, joint_valid, product_lattices, product_valid = train_test_split(joint_lattices, product_lattices,
        #                                                                                 test_size=0.2, random_state=42)

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
        
        # lr_scheduler(mice.lin_adve_running(epoch=epoch, data=valid_losses, window_size=window_size))
        # early_stopping(mice.lin_ave_running(epoch=epoch, data=valid_losses, window_size=window_size))
        lr_scheduler(valid_losses[-1])
        if epoch > 300:
            early_stopping(valid_losses[-1])
            if early_stopping.early_stop:
                break
        cntr += 1
        if epoch % freq_print == 0:
            print(f'\nMI for train {train_losses[-1]}, val {valid_losses[-1]} at step {epoch}')
        
    torch.save(model.state_dict(), PATH)
    # train_losses = mice.exp_ave(data=train_losses)
    # valid_losses = mice.exp_ave(data=valid_losses)
    mice.logger(f'MI train for ising, T = {T} is: {train_losses[-1]:.2f}', flag_message=2)
    mice.ising_fig(num=0, genom=genom, T=T, train_losses=train_losses, valid_losses=valid_losses)
    mi_num_box_dependant.append(train_losses[-1])
    mi_num_box_dependant_valid.append(valid_losses[-1])
    mi_num_box_dependant = np.array(mi_num_box_dependant)
    return mi_num_box_dependant[-1]

def ising_temp():
    my_path = os.path.join('./', 'ising_h5py')
    Ts = [round(T,2) for T in np.linspace(0.1, 4, 40)]

    mi = np.zeros(len(Ts))
    for idx, T in enumerate(Ts):
        if T.is_integer():
            T = int(T)
        print('loading into .h5py format...')
        mice.runner_loader(T)
        print(f'Running on T = {T}')
        print('='*50)
        current_mi = mice.ising_box_runner(idx=idx, T=T)
        mi[idx] = current_mi
        df = pd.DataFrame({'T': Ts, 'MI': mi})
        mice.ising_temp_fig_running(df=df)
        file_path = os.path.join(my_path, f'ising_data_64_{T}.h5')
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'{file_path} got deleted')
    df = pd.DataFrame({'T': Ts, 'MI': mi})
    mice.ising_temp_fig(df=df)

if __name__ == '__main__':
    mice.ising_temp()    
