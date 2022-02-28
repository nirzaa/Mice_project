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
def box_runner(num_boxes, box_frac, idx, max_epochs, batch_size, freq_print, genom, lr, weight_decay, num_samples, transfer_epochs, window_size=3):
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
    sizes = mice.sizer(num_boxes=num_boxes, box_frac=box_frac)
    num_frames = mice.frames()
    lattices = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
    list_ising = lattices.copy()
    x_size = list_ising[0].shape[1]
    y_size = list_ising[0].shape[2]
    z_size = list_ising[0].shape[3]
    input_size = int(8 * ((x_size-2)/1+1) * ((y_size-2)/1+1) * ((z_size-2)/1+1))
    model = mice.mi_model(genom=genom, n_epochs=n_epochs, max_epochs=max_epochs, input_size=input_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = mice.LRScheduler(optimizer)
    early_stopping = mice.EarlyStopping()
    train_losses = []
    valid_losses = []
    axis = int(np.argmax((list_ising[0].shape[1], list_ising[0].shape[2], list_ising[0].shape[3]))) + 1
    cntr = 0
    for epoch in tqdm(range(int(n_epochs))):
        lattices = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
        list_ising = lattices.copy()
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

        if epoch > 1000:
            lr_scheduler(valid_losses_exp[-1])
            early_stopping(valid_losses_exp[-1])
            if early_stopping.early_stop:
                break
        cntr += 1
        if epoch % freq_print == 0:
            print(f'\nMI for train {train_losses_exp[-1]}, val {valid_losses_exp[-1]} at step {epoch}')
        
    train_losses = mice.exp_ave(data=train_losses)
    valid_losses = mice.exp_ave(data=valid_losses)
    torch.save(model.state_dict(), PATH)
    mice.logger(f'MI train for num boxes, num_boxes = {num_boxes} is: {train_losses[-1]:.2f}', flag_message=0)
    mice.box_fig(num=0, genom=genom, num_boxes=num_boxes, train_losses=train_losses, valid_losses=valid_losses)
    mi_num_box_dependant.append(train_losses[-1])
    mi_num_box_dependant_valid.append(valid_losses[-1])
    mi_num_box_dependant = np.array(mi_num_box_dependant)
    return mi_num_box_dependant[-1]

def box_caller():
    box_sizes = list(np.arange(4, 80, 2, dtype='int'))

    mi = np.zeros(len(box_sizes))
    for idx, num_boxes in enumerate(box_sizes):
        print(f'Running on number of boxes = {num_boxes}')
        print('='*50)
        current_mi = mice.box_runner(num_boxes=num_boxes, idx=idx)
        mi[idx] = current_mi
        df = pd.DataFrame({'number of splits of space': box_sizes, 'MI': mi})
        mice.box_temp_fig_running(df=df)
    df = pd.DataFrame({'number of splits of space': box_sizes, 'MI': mi})
    mice.box_temp_fig(df=df)

if __name__ == '__main__':
    mice.box_caller()
