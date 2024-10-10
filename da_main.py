import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
from da_test import test
from model import WSAD
# from utils import Visualizer
import os
from dataset_loader import *
from tqdm import tqdm
import wandb


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None
    gpus = [0]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    # config.len_feature = 1024

    wandb.init(
            project="UR-DMU", 
            config=args, 
            job_type="train")
    wandb.config.update({
            "batch_size": args.batch_size,
            "lr": config.lr[0],
            "max_epoch": config.num_iters
        }, allow_val_change=True)
    

    net = WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60)
    net = net.cuda()

    train_nset = Drone_Anomaly(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = args.num_segments, len_feature = config.len_feature, is_normal = True)
    train_aset = Drone_Anomaly(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = args.num_segments, len_feature = config.len_feature, is_normal = False)
    test_set = Drone_Anomaly(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature)
    print(f'train_Nset: {len(train_nset)}')
    print(f'train_Aset: {len(train_aset)}')
    print(f'test_set: {len(test_set)}')

    train_nloader = data.DataLoader(train_nset,
            batch_size = args.batch_size,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    train_aloader = data.DataLoader(train_aset,
            batch_size = args.batch_size,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(test_set,
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)
    print(f'train_Nloader: {len(train_nloader)}')
    print(f'train_Aloader: {len(train_aloader)}')
    print(f'test_loader: {len(test_loader)}')

    # breakpoint()
    test_info = {"step": [], "AUC": [],"AP":[]}
    best_auc = 0
    criterion = AD_Loss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
        betas = (0.9, 0.999), weight_decay = 0.00005)


    test(net, config, wandb, test_loader, test_info, 0)
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(train_nloader) == 0:
            normal_loader_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            abnormal_loader_iter = iter(train_aloader)

        train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion, wandb, step)

        if step % 10 == 0 and step > 10:
            test(net, config, wandb, test_loader, test_info, step)
            if test_info["AUC"][-1] > best_auc:
                best_auc = test_info["AUC"][-1]
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "da_best_record_{}.txt".format(config.seed)))

                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "da_trans_{}.pkl".format(config.seed)))
            if step == config.num_iters:
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "da_trans_{}.pkl".format(step)))

