import h5py
import numpy as np
import os
import gin
import torch
import mice
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from tqdm import tqdm
import wandb
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class MiceDataset(Dataset):
    '''
    Defining the Dataset to be used in the DataLoader

    x_joint: the lattices we got for the joint
    x_product: the lattices we got for the product
    n_samples: the number of lattices we got
    '''

    def __init__(self, x_joint, x_product):
        self.x_joint = x_joint
        self.x_product = x_product
        self.n_samples = x_joint.shape[0]

    def __getitem__(self, item):
        return self.x_joint[item], self.x_product[item]

    def __len__(self):
        return self.n_samples

def read_data(T):
    '''
    Reading the data to train our neural net on

    return:
    blocks: the coordinates of our particles in different frames in time
    '''
    my_hf = h5py.File(os.path.join(f'./data/dump_files/Aluminum/pure_liquid/{T}', 'data.h5'), 'r')
    n1 = my_hf.get('dataset_1')
    blocks = np.array(n1)
    my_hf.close()
    return blocks

def frames(T):
    '''
    Calculating the number of frames in the input

    return:
    the number of frames
    '''
    blocks = read_data(T)
    num_frames = blocks.shape[0]  # number of frames in the input
    print(f'The number of frames in the input is: {num_frames}',
          f'\n',
          '='*50)
    return num_frames

def sizer(num_boxes, box_frac):
    '''
    Calculate the size for our boxes to split our space to
    
    num_boxes: the number of boxes we split our space to
    box_frac: what is the value of the box from the total space we are calculating the mutual information to
    
    return:
    the size of our box we are calculating the mutual information to
    '''
    x_size, y_size, z_size = int(np.floor(num_boxes*box_frac)), int(np.floor(num_boxes*box_frac)), int(np.floor(num_boxes*box_frac))
    x_size, y_size, z_size =  x_size - x_size%2, y_size - y_size%2, z_size - z_size%2
    print(f'\nWe split the space into {num_boxes}x{num_boxes}x{num_boxes} boxes\n',
          f'The size of the small box is: ({x_size}, {y_size}, {z_size})\n',
          f'='*50)
    return (x_size, y_size, z_size)

def mi_model(genom, n_epochs, max_epochs, input_size=100, sizes=(10,10,10)):
    '''
    Declare the model and loading the weights if necessary
    
    genom: the type of architecture we will use for the neural net
    n_epochs: number of epochs in the current run
    max_epochs: the maximum number of epochs we are using at the start, before we are using transfer learning

    return:
    the relevant model loaded with its weights
    '''
    # early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"
    weights_path = os.path.join('./', 'src', 'model_weights')

    if genom == 'linear':
        model = mice.Net()
        model.to(device)
    elif genom == 'fcn':
        model = mice.Model()
        model.to(device)
    elif genom == 'new_fcn':
        model = mice.Modely(input_size=input_size)
        model.to(device)
    elif genom == 'mice_conv':
        model = mice.MiceConv()
        model.to(device)
    elif genom == 'sandnet':
        model = mice.Sandnet(input_size=input_size)
        model.to(device)
    elif genom == 'sandnet3d':
        model = mice.Sandnet3d(input_size=input_size)
        model.to(device)
    elif genom == 'sandnet2d':
        model = mice.Sandnet2d(input_size=input_size)
        model.to(device)
    elif genom == 'sandnet3d_emb':
        model = mice.Sandnet3d_emb(input_size=input_size)
        model.to(device)

    if n_epochs != max_epochs and genom == 'linear':
        print(f'==== linear ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'lin_model_weights.pth')
        model = mice.Net()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'linear':
        PATH = os.path.join(weights_path, 'lin_model_weights.pth')
        print(f'==== linear ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'fcn':
        print(f'==== fcn ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'fcn_model_weights.pth')
        model = mice.Model()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'fcn':
        PATH = os.path.join(weights_path, 'fcn_model_weights.pth')
        print(f'==== fcn ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'new_fcn':
        print(f'==== new fcn ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'new_fcn_model_weights.pth')
        model = mice.Modely(input_size=input_size)
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'new_fcn':
        PATH = os.path.join(weights_path, 'new_fcn_model_weights.pth')
        print(f'==== new fcn ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'mice_conv':
        print(f'==== mice_conv ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'mice_conv_model_weights.pth')
        model = mice.MiceConv()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'mice_conv':
        PATH = os.path.join(weights_path, 'mice_conv_model_weights.pth')
        print(f'==== mice_conv ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'sandnet':
        print(f'==== sandnet ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'sandnet_model_weights.pth')
        model = mice.Sandnet()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'sandnet':
        PATH = os.path.join(weights_path, 'sandnet_model_weights.pth')
        print(f'==== sandnet ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'sandnet3d':
        print(f'==== sandnet3d ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'sandnet3d_model_weights.pth')
        model = mice.Sandnet3d()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'sandnet3d':
        PATH = os.path.join(weights_path, 'sandnet3d_model_weights.pth')
        print(f'==== sandnet3d ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'sandnet2d':
        print(f'==== sandnet2d ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'sandnet2d_model_weights.pth')
        model = mice.Sandnet()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'sandnet2d':
        PATH = os.path.join(weights_path, 'sandnet2d_model_weights.pth')
        print(f'==== sandnet2d ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    if n_epochs != max_epochs and genom == 'sandnet3d_emb':
        print(f'==== sandnet3d_emb ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'sandnet3d_emb_model_weights.pth')
        model = mice.Sandnet3d_emb()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'sandnet3d_emb':
        PATH = os.path.join(weights_path, 'sandnet3d_emb_model_weights.pth')
        print(f'==== sandnet3d_emb ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')


    return model

@gin.configurable
def boxes_maker(T, num_boxes, sample, flag):
    '''
    Generating the sliced box of the sample mentioned in {sample}
    
    num_boxes: the number of boxes we split our space to
    sample: number of sample we chose randomly to take our data from
    flag: whether to use the data from data.h5 (flag=0), random data (flag=1), or the data that reproduces log2 mutual information (flag=2)

    return:
    our splitted space into the number of boxes we defined
    '''
    if flag == 0:
        borders = np.linspace(0, 1, num_boxes+1, endpoint=True)
        blocks = read_data(T)
        boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))

        df_particles = pd.DataFrame(
            blocks[sample],
            columns=['X', 'Y', 'Z']
        )

        x_bin = borders.searchsorted(df_particles['X'].to_numpy())
        y_bin = borders.searchsorted(df_particles['Y'].to_numpy())
        z_bin = borders.searchsorted(df_particles['Z'].to_numpy())

        g = dict((*df_particles.groupby([x_bin, y_bin, z_bin]),))

        g_keys = list(g.keys())

        for cntr, cor in enumerate(g_keys):
            boxes_tensor[cor[0]-1, cor[1]-1, cor[2]-1] = 1

    elif flag == 1:
        boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
        flag_torch = torch.randint_like(torch.tensor(boxes_tensor), low=0, high=2)
        boxes_tensor[flag_torch == 1] = 1

    elif flag == 2:
        boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
        i = np.random.randint(low=0, high=boxes_tensor.shape[0])
        j = np.random.randint(low=0, high=boxes_tensor.shape[1])
        k = np.random.randint(low=0, high=boxes_tensor.shape[2])
        boxes_tensor[i, j, k] = 1
    
    return boxes_tensor

@gin.configurable
def lattices_generator(num_samples, samples_per_snapshot, R, num_frames, num_boxes, sizes, cntr=0, lattices=None):
    '''
    Generate the lattices that will be used in our neural net
    
    num_samples: number of samples we will have in each epoch
    samples_per_snapshot: the number of samples to take from each snapshot
    R: np.random.RandomState
    num_frames: number of frames we have in the data, from it we will pick 1 frame randomly to take our data from
    num_boxes: the number of boxes we split our space to
    sizes: the sizes of the box we are calculating the mutual information to
    cntr: just a counter
    lattice: a list we will put the lattices we will construct into

    return:
    list of lattices we've constracted
    '''
    if lattices is None:
        lattices = []
    x_size, y_size, z_size = sizes
    num_sample = R.randint(num_frames)
    my_tensor = mice.boxes_maker(num_boxes=num_boxes, sample=num_sample)  # returns a tensor
    leny_x = my_tensor.shape[0]
    leny_y = my_tensor.shape[1]
    leny_z = my_tensor.shape[2]
    x_steps = leny_x - x_size
    y_steps = leny_y - y_size
    z_steps = leny_z - z_size
    while True:
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

        lattices.append(np.expand_dims(my_tensor[i:i+x_size, j:j+y_size, k:k + z_size], axis=0))
        cntr += 1
        if cntr == num_samples:
            return lattices
        elif cntr % (samples_per_snapshot) == 0:
            return lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes, cntr=cntr, lattices=lattices)

@gin.configurable
def lattices_generator_h5py(num_samples, samples_per_snapshot, R, num_frames, num_boxes, sizes, cntr=0, lattices=None, flag=0):
    '''
    Generate the lattices that will be used in our neural net
    
    num_samples: number of samples we will have in each epoch
    samples_per_snapshot: the number of samples to take from each snapshot
    R: np.random.RandomState
    num_frames: number of frames we have in the data, from it we will pick 1 frame randomly to take our data from
    num_boxes: the number of boxes we split our space to
    sizes: the sizes of the box we are calculating the mutual information to
    cntr: just a counter
    lattice: a list we will put the lattices we will construct into

    return:
    list of lattices we've constracted
    '''
  
    lattices = []
    x_size, y_size, z_size = sizes
    num_sample = R.randint(num_frames)
    my_tensor = mice.boxes_maker(num_boxes=num_boxes, sample=num_sample)  # returns a tensor
    leny_x = my_tensor.shape[0]
    leny_y = my_tensor.shape[1]
    leny_z = my_tensor.shape[2]
    x_steps = leny_x - x_size
    y_steps = leny_y - y_size
    z_steps = leny_z - z_size

    for _ in tqdm(range(int(num_samples))):
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

        lattices.append(np.expand_dims(my_tensor[i:i+x_size, j:j+y_size, k:k + z_size], axis=0))
        cntr += 1
        if cntr % (samples_per_snapshot) == 0:
            num_sample = R.randint(num_frames)
            my_tensor = mice.boxes_maker(num_boxes=num_boxes, sample=num_sample)  # returns a tensor
    return lattices

def lattice_splitter(lattices, axis):
    '''
    Here we are splitting the lattices given in {lattices} on the axis given in the {axis}
    
    lattices: list of the lattices we've constructed
    axis: the axis we will split our lattices on

    return:
    left lattices and right lattices
    '''
    
    left_lattices, right_lattices = [], []
    for lattice in lattices:
        left_lattice, right_lattice = np.split(lattice, 2, axis=axis)
        left_lattices.append(left_lattice)
        right_lattices.append(right_lattice)
    return np.array(left_lattices), np.array(right_lattices)

def loss_function(joint_output, product_output):
    """
    calculating the loss function
    
    joint_output: the joint lattices we've constructed
    product_output: the product lattices we've constructed

    return:
    mutual: the mutual information
    joint_output: the joint lattices we've constructed
    exp_product: the exponent of the product_output
    """
    exp_product = torch.exp(product_output)
    mutual = torch.mean(joint_output) - torch.log(torch.mean(exp_product))
    return mutual, joint_output, exp_product

def train_one_epoch(window_size, epoch, train_losses, model, data_loader, optimizer, ma_rate=0.01):
    '''
    train one epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    optimizer: optimizer
    ma_rate: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    model.train()
    total_loss = 0
    total_mutual = 0
    for batch_idx, data in enumerate(data_loader):
        loss, mutual = train_one_step(window_size, epoch, train_losses, model, data, optimizer, ma_rate)
        total_loss += loss
        total_mutual += mutual
    total_loss = total_loss / len(data_loader)
    total_mutual = total_mutual / len(data_loader)
    if epoch > window_size*2:
        total_mutual = float(mice.loss_lin_ave(current_loss=total_mutual.cpu().detach().numpy(), data=train_losses, window_size=window_size))
        total_mutual = torch.tensor(total_mutual, requires_grad=True)
    return total_loss, total_mutual

def train_one_step(window_size, epoch, train_losses, model, data, optimizer, ma_rate, ma_et=1.0):
    '''
    train one batch in the epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    optimizer: optimizer
    ma_rate: used in order to calculate the loss function
    ma_et: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    x_joint, x_product = data
    optimizer.zero_grad()
    joint_output = model(x_joint.float())
    product_output = model(x_product.float())
    try:
        mutual, joint_output, exp_product = loss_function(joint_output, product_output)
    except Exception as e:
        print(f'The error is:\n{e}')
        return 'problem'
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(exp_product)
    loss_train = my_criterion(joint_output, ma_et, exp_product)
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    return loss_train, mutual

def my_criterion(joint_output, ma_et, exp_product):
    return -(torch.mean(joint_output) - (1 / ma_et.mean()).detach() * torch.mean(exp_product))

def valid_one_epoch(window_size, epoch, valid_losses, model, data_loader, ma_rate=0.01):
    '''
    validation of one epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    ma_rate: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    model.eval()
    total_loss = 0
    total_mutual = 0
    for batch_idx, data in enumerate(data_loader):
        with torch.no_grad():
            loss, mutual = valid_one_step(window_size, epoch, valid_losses, model, data, ma_rate)
            total_loss += loss
            total_mutual += mutual
    total_loss = total_loss / len(data_loader)
    total_mutual = total_mutual / len(data_loader)
    if epoch > window_size*2:
        total_mutual = float(mice.loss_lin_ave(current_loss=total_mutual.cpu().detach().numpy(), data=valid_losses, window_size=window_size))
        total_mutual = torch.tensor(total_mutual, requires_grad=True)
    return total_loss, total_mutual
def valid_one_step(window_size, epoch, valid_losses, model, data, ma_rate, ma_et=1.0):
    '''
    validation of one batch in the epoch
    
    model: the model we will train
    data_loader: the data_loader that keeps our data
    ma_rate: used in order to calculate the loss function
    ma_et: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    x_joint, x_product = data
    joint_output = model(x_joint.float())
    product_output = model(x_product.float())
    try:
        mutual, joint_output, exp_product = loss_function(joint_output, product_output)
    except Exception as e:
        print(f'The error is:\n{e}')
        return 'problem'
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(exp_product)
    loss_train = -(torch.mean(joint_output) - (1 / ma_et.mean()).detach() * torch.mean(exp_product))
    if epoch > window_size*2:
        loss_train = float(mice.loss_lin_ave(current_loss=loss_train.cpu().detach().numpy(), data=valid_losses, window_size=window_size))
        loss_train = torch.tensor(loss_train, requires_grad=True)
    return loss_train, mutual

@gin.configurable
def box_fig(num, genom, num_boxes, train_losses, valid_losses, figsize):
    '''
    Plot the results
    
    num: number of the figure
    genom: the type of architecture we've trained our neural net with
    num_boxes: the number of boxes we split our space to
    train_losses: the losses we got from training
    valid_losses: the losses we got from validating
    figsize: the size of the figure

    return:
    mi_train: mutual information gained from training
    mi_valid: mutual information gained from validating
    '''
    plt.figure(num=num, figsize=figsize)
    plt.title(f'Searching for perfect box number, try: {num_boxes} boxes')
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    saved_path = os.path.join('./', 'figures', "losses", "box_size_search", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, str(num_boxes)+"_boxes"))
    plt.figure(num=num).clear()
    plt.close(num)
    mi_train, mi_valid = train_losses[-1], valid_losses[-1]
    
    return mi_train, mi_valid

@gin.configurable
def box_fig_together(box_sizes, mi_num_box_dependant, mi_num_box_dependant_valid, genom, figsize):
    '''
    Plot the results together
    
    boxes_sizes: the sizes of the boxes we calculated for
    mi_num_box_dependant: mutual informations were calculated for training
    mi_num_box_dependant_valid: mutual informations were calculated for validation
    genom: the type of architecture we've trained our neural net with
    figsize: the size of the figure

    return:
    None
    '''
    plt.figure(num=len(box_sizes)+1, figsize=figsize)
    plt.clf()
    plt.xlabel('number of box splits of the biggest box')
    plt.ylabel('Mutual Information')
    plt.title('Box size searching: All the MI together')
    plt.tight_layout()
    plt.plot(box_sizes, mi_num_box_dependant, label=(genom + ' - train'))
    plt.plot(box_sizes, mi_num_box_dependant_valid, label=(genom + ' - valid'))
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "box_size_search")
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'all_mi_together'))

    return None

@gin.configurable
def box_fig_running(box_sizes, mi_num_box_dependant, mi_num_box_dependant_valid, genom, figsize):
    '''
    Plot the results together while running the simulation
    
    boxes_sizes: the sizes of the boxes we calculated for
    mi_num_box_dependant: mutual informations were calculated for training
    mi_num_box_dependant_valid: mutual informations were calculated for validation
    genom: the type of architecture we've trained our neural net with
    figsize: the size of the figure

    return:
    None
    '''
    plt.figure(num=len(box_sizes)+2, figsize=figsize)
    plt.clf()
    plt.xlabel('number of box splits of the biggest box')
    plt.ylabel('Mutual Information')
    plt.title('Box size searching...')
    plt.tight_layout()
    plt.plot(box_sizes, mi_num_box_dependant, label=(genom + ' - train'))
    plt.plot(box_sizes, mi_num_box_dependant_valid, label=(genom + ' - valid'))
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "box_size_search")
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'simulation_running'))

    return None

@gin.configurable
def entropy_fig(num, genom, sizes, train_losses, valid_losses, figsize):
    '''
    Plot the results
    
    num: number of the figure
    genom: the type of architecture we've trained our neural net with
    sizes: the size of the small box
    train_losses: the losses we got from training
    valid_losses: the losses we got from validating
    figsize: the size of the figure

    return:
    mi_train: mutual information gained from training
    mi_valid: mutual information gained from validating
    '''
    x_size, y_size, z_size = sizes
    plt.figure(num=num, figsize=figsize)
    plt.clf()
    plt.title(f'Calculating MI for: ({x_size}, {y_size}, {z_size}) shape box')
    plt.plot(train_losses, label=(genom + ' - train'))
    plt.plot(valid_losses, label=(genom + ' - valid'))
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    saved_path = os.path.join('./', 'figures', "losses", "entropy_calculation", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path,str(x_size)+"_"+str(y_size)+"_"+str(z_size)))
    plt.figure(num=num).clear()
    plt.close(num)
    mi_train, mi_valid = train_losses[-1], valid_losses[-1]
    return mi_train, mi_valid

@gin.configurable
def entropy_fig_together(x_labels, mi_entropy_dependant, mi_entropy_dependant_valid, genom, figsize):
    '''
    Plot the results together
    
    x_labels: the sizes of the boxes of the calculation
    mi_entropy_dependant: the mutual informations together of the training
    mi_entropy_dependant_valid: the mutual informations together of the validation
    genom: the type of architecture we've trained our neural net with
    figsize: the size of the figure

    return:
    None
    '''
    plt.figure(num=len(x_labels)+1, figsize=figsize)
    plt.clf()
    plt.xlabel('size of the small box')
    plt.ylabel('Mutual Information')
    plt.title('entropy searching: All the MI together')
    plt.tight_layout()
    plt.plot(x_labels, mi_entropy_dependant, label=(genom + ' - train'))
    plt.plot(x_labels, mi_entropy_dependant_valid, label=(genom + ' - valid'))
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "entropy_calculation")
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'all_mi_together'))

    return None

@gin.configurable
def entropy_fig_running(x_labels, mi_entropy_dependant, mi_entropy_dependant_valid, genom, figsize):
    '''
    Plot the results together while the simulation is running
    
    x_labels: the sizes of the boxes of the calculation
    mi_entropy_dependant: the mutual informations together of the training
    mi_entropy_dependant_valid: the mutual informations together of the validation
    genom: the type of architecture we've trained our neural net with
    figsize: the size of the figure

    return:
    None
    '''
    plt.figure(num=len(x_labels)+2, figsize=figsize)
    plt.clf()
    plt.xlabel('size of the small box')
    plt.ylabel('Mutual Information')
    plt.title('Entropy searching...')
    plt.tight_layout()
    plt.plot(x_labels, mi_entropy_dependant, label=(genom + ' - train'))
    plt.plot(x_labels, mi_entropy_dependant_valid, label=(genom + ' - valid'))
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "entropy_calculation")
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'simulation_running'))

@gin.configurable
def ising_fig(num, genom, T, train_losses, valid_losses, figsize):
    '''
    Plot the results
    
    num: number of the figure
    genom: the type of architecture we've trained our neural net with
    num_boxes: the number of boxes we split our space to
    train_losses: the losses we got from training
    valid_losses: the losses we got from validating
    figsize: the size of the figure

    return:
    mi_train: mutual information gained from training
    mi_valid: mutual information gained from validating
    '''
    plt.figure(num=num, figsize=figsize)
    plt.title(f'Calculating the ising mi for T = {T}')
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    saved_path = os.path.join('./', 'figures', "losses", "ising", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'ising_T='+str(T).replace('.','_')))
    plt.figure(num=num).clear()
    plt.close(num)
    mi_train, mi_valid = train_losses[-1], valid_losses[-1]
    
    return mi_train, mi_valid

@gin.configurable
def logger(my_str, mod, flag=[], number_combinations=0, flag_message=0, num_boxes=0):
    '''
    prints the results

    my_str: the string to be plotted
    mod: mod 0 prints both | mod 1 : prints only output | mod 2 : prints only to file
    flag: if it is first time printing
    number_combinations: the lengh of our printing
    flag_message: if we are printing the box size searching or the entropy calculation
    '''
    message_path = os.path.join('./', 'src', 'mice')
    mice.folder_checker(message_path)
    if flag_message == 0:
        message_path = os.path.join(message_path, 'message_boxcalc.log')
    elif flag_message == 1:
        message_path = os.path.join(message_path, 'message_entropycalc.log')
    elif flag_message == 2:
        message_path = os.path.join(message_path, 'message_isingcalc.log')

    try:
        logger.counter += 1
    except Exception:
        logger.counter = 0

    if flag == []:
        flag.append('stop')
        log_file = open(message_path, "w")
        sys.stdout = log_file
        if flag_message == 0:
            print(f'==== log file for the Mutual Information for different number of boxes ====\n\n'
                  f'We have {number_combinations} runs in total\n\n')
        elif flag_message == 1:
            print(f'==== log file for the Mutual Information for different box shapes ====\n\n'
                  f'We split our space into: {num_boxes} boxes.\nWe have {number_combinations} runs in total\n\n')
        elif flag_message == 2:
            print(f'==== log file for the Mutual Information for ising ====\n\n')
        sys.stdout = sys.__stdout__
        log_file.close()

    if mod == 0:
        log_file = open(message_path, "a+")
        sys.stdout = log_file
        print(logger.counter,". ",my_str,"\n")
        sys.stdout = sys.__stdout__
        log_file.close()
        print(my_str)
    elif mod == 1:
        print(my_str)
    elif mod == 2:
        log_file = open(message_path, "a+")
        sys.stdout = log_file
        print(logger.counter,". ",my_str,"\n")
        sys.stdout = sys.__stdout__
        log_file.close()

def folder_checker(path):
    '''
    if a folder in {path} does not exist, this function will create it

    return:
    None
    '''
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    return None

def sort_func(args):
    '''
    calculating the i*j*k of the input

    return:
    i*j*k
    '''
    i,j,k = args
    return i*j*k
        
def print_combinations(my_combinations):
    '''
    printing all of the combinations

    my_combinations: or combinations

    return:
    None
    '''
    print(f'All of my combinations are:')
    for cntr, i in enumerate(my_combinations):
        print(f'{cntr}. {i}')
    return None

@gin.configurable
def exp_ave(data, window_frac):
    data = np.array(data)
    window = np.floor(data.shape[0] * window_frac).astype(int)
    ave_arr = np.zeros((data.shape[0]))
    mi = data[0]
    alpha = 2 / float(window + 1)
    for i in range(data.shape[0]):
        mi =  ((data[i] - mi) * alpha) + mi
        ave_arr[i] = mi
    return ave_arr

@gin.configurable
def lin_ave(data, window_frac):
    data = np.array(data)
    window = np.floor(data.shape[0] * window_frac).astype(int)
    return [np.mean(data[i:i+window]) for i in range(0,len(data)-window)]

def lin_ave_running(epoch, data, window_size):
    data = np.array(data)
    if epoch == 0:
        return [0]
    elif epoch < window_size:
        return [np.mean(data[i:i+epoch]) for i in range(0,len(data)-epoch)]
    return [np.mean(data[i:i+window_size]) for i in range(0,len(data)-window_size)]

def loss_lin_ave(current_loss, data, window_size):
    current_data = data.copy()
    current_data.append(current_loss)
    current_data = np.array(current_data)
    return current_data[-window_size:].mean()

def loss_exp_ave(current_loss, data, window_size):
    weights = np.linspace(1, 10, 10, dtype='int')
    weights = np.exp(weights)
    current_data = data.copy()
    current_data.append(current_loss)
    current_data = np.array(current_data)
    return np.ma.average(current_data, weights=weights)

    
@gin.configurable
def ising_temp_fig(df, figsize, genom):
    '''
    Plot the mutual information of ising as a function of the temperature

    figsize: the size of the figure
    df: dataframe with our data

    return:
    None
    '''
    plt.figure(num=0, figsize=figsize)
    plt.clf()
    plt.xlabel('Temperature')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information as a function of the Temperature')
    plt.tight_layout()
    sns.relplot(x='T', y='MI', data=df)
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "ising", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'all_mi_together'))

    return None

@gin.configurable
def ising_temp_fig_running(df, figsize, genom):
    '''
    Plot the mutual information of ising as a function of the temperature

    figsize: the size of the figure
    df: dataframe with our data

    return:
    None
    '''
    plt.figure(num=0, figsize=figsize)
    plt.clf()
    plt.xlabel('Temperature')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information as a function of the Temperature')
    plt.tight_layout()
    sns.relplot(x='T', y='MI', data=df)
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "ising", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'simulation_running'))

    return None

@gin.configurable
def box_temp_fig_running(df, figsize, genom):
    '''
    Plot the mutual information of ising as a function of the temperature

    figsize: the size of the figure
    df: dataframe with our data

    return:
    None
    '''
    plt.figure(num=0, figsize=figsize)
    plt.clf()
    plt.xlabel('Box sizes')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information as a function of the Box size')
    plt.tight_layout()
    sns.relplot(x='number of splits of space', y='MI', data=df)
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "box_size_search", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'simulation_running'))

    return None

@gin.configurable
def box_temp_fig(df, figsize, genom):
    '''
    Plot the mutual information of ising as a function of the temperature

    figsize: the size of the figure
    df: dataframe with our data

    return:
    None
    '''
    plt.figure(num=0, figsize=figsize)
    plt.clf()
    plt.xlabel('Box sizes')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information as a function of the Box size')
    plt.tight_layout()
    sns.relplot(x='number of splits of space', y='MI', data=df)
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "box_size_search", genom)
    mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'all_mi_together'))

    return None