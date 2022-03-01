import gin
import numpy as np

from mice.neural_net.architectures import (
    MiceConv,
    Net,
    Model,
    Modely,
    Sandnet,
    Sandnet3d,
    Sandnet2d
)

from mice.utils.my_utils import (
    MiceDataset,
    read_data,
    frames,
    sizer,
    mi_model,
    boxes_maker,
    lattices_generator,
    lattice_splitter,
    loss_function,
    train_one_epoch,
    train_one_step,
    valid_one_epoch,
    valid_one_step,
    box_fig,
    box_fig_together,
    box_fig_running,
    entropy_fig,
    entropy_fig_together,
    entropy_fig_running,
    ising_fig,
    folder_checker,
    sort_func,
    logger,
    print_combinations,
    exp_ave,
    lin_ave,
    ising_temp_fig,
    ising_temp_fig_running,
    lin_ave_running,
    loss_lin_ave,
    box_temp_fig_running,
    box_temp_fig,
    lattices_generator_h5py,
    my_criterion
)

from mice.main.box_menu import (
    box_runner,
    box_caller
)

from mice.main.box_menu_ising import (
    ising_box_runner,
    ising_temp
)

from mice.main.entropy_menu import (
    entropy_runner
)

from mice.main.entropy_menu_sweep import (
    wandb_sweep,
    entropy_run,
)

from mice.main.sweep_func import (
    sweep_wandb_sweep,
    sweep_entropy_run,
    run_sweep,
    input_func
)

from mice.main.sweep_func_ising import (
    sweep_wandb_ising,
    sweep_ising_run,
    run_sweep_ising,
    input_func_ising,
    sand
)

from bin.load_data import (
    file_load
)

from mice.ising_const.ising_script import (
    ising_runner,
    part_lattices
)

from mice.utils.pytorch_modules import (
    LRScheduler,
    EarlyStopping
)

from mice.main.ising_loader import (
    runner_loader
)

# gin configurations
gin.external_configurable(np.array)
gin.parse_config_file('./src/mice/config.gin')
