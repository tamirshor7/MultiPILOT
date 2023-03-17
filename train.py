import itertools
import logging
import pathlib
import random
import shutil
import time
import pandas
import os
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio
from sewar.full_ref import vifp
from image_similarity_measures.quality_metrics import fsim
from torch.utils.data import DataLoader
import argparse
from data import transforms as transforms
from data.mri_mf_data import SliceData
import h5py
import matplotlib.pyplot as plt
from models.subsampling_model import Subsampling_Model
from common.utils import get_vel_acc

#Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#This class performs data transformations on the data coming from the dataset
class DataTransform:
    def __init__(self, resolution=[384, 144]):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution[0], self.resolution[1]))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)

        target = transforms.to_tensor(target)
        target = transforms.center_crop(target.unsqueeze(0), (self.resolution[0], self.resolution[1])).squeeze()
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        mean = std = 0
        return image, target, mean, std

def get_metrics(gt, pred,compute_vif=False,compute_fsim=False):
    """ Compute Peak Signal to Noise Ratio metric (PSNR).
     By default we do not compute FSIM and VIF for every iteration because their long
     compute times. We advise training the model and then computing these values only for
     post-training evaluations.
     """
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    stacked_gt, stacked_pred = gt.reshape(-1, gt.shape[2], gt.shape[3]), pred.reshape(-1, gt.shape[2], gt.shape[3])

    vif = np.mean([vifp(stacked_gt[i], stacked_pred[i]) for i in range(stacked_gt.shape[0])]) if compute_vif else None
    fsim_val = fsim(stacked_gt,stacked_pred) if compute_fsim else None

    return peak_signal_noise_ratio(gt, pred, data_range=gt.max()), vif, fsim_val


def boost_examples(files, num_frames_per_example):
    '''Boost existing examples. This option is provided, however it is not used in our work
        (since we use augmentations that hold boosting within them)'''
    examples_per_clip = {}
    for fname in sorted(files):
        curr_num_examples = 0
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace']  # [slice, frames, coils, h,w]
            for start_frame_index in range(kspace.shape[1] - num_frames_per_example):
                num_slices = kspace.shape[0]
                curr_examples = [(fname, slice, start_frame_index, start_frame_index + num_frames_per_example) for slice
                                 in range(num_slices)]
                curr_num_examples += len(curr_examples)
        examples_per_clip[fname] = curr_num_examples

    max_examples = np.max([k for k in examples_per_clip.values()])
    factors = {k: np.int(np.floor(max_examples / v)) for k, v in examples_per_clip.items()}
    return factors


def get_rel_files(files, resolution, num_frames_per_example):
    '''Filter data to only use files with desired resolution and frame length'''
    rel_files = []
    for fname in sorted(files):
        with h5py.File(fname, 'r') as data:
            if not 'aug.h5' in fname:
                kspace = data['kspace']  # [slice, frames, coils, h,w]
            else:
                kspace = data['images']
            if kspace.shape[3] < resolution[0] or kspace.shape[4] < resolution[1]:
                continue
            if kspace.shape[1] < num_frames_per_example:
                continue
        rel_files.append(fname)
    return rel_files


def create_datasets(args):
    if args.augment:  # all pre testing already done in this case
        rel_files = [str(args.data_path) + '/' + str(fname) for fname in os.listdir(args.data_path) if
                     os.path.isfile(os.path.join(args.data_path, fname))]
    else:
        ocmr_data_attributes_location = args.ocmr_path
        df = pandas.read_csv(ocmr_data_attributes_location)
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        rel_files = [args.data_path._str + '/' + k for k in df[df['smp'] == 'fs']['file name'].values]
        rel_files = get_rel_files(rel_files, DataTransform().resolution, args.num_frames_per_example)
    clips_factors = None
    if args.boost:
        clips_factors = boost_examples(rel_files, args.num_frames_per_example)
    np.random.shuffle(rel_files)
    train_ratio = 0.8
    num_train = int(np.ceil(len(rel_files) * train_ratio))
    train_files = rel_files[:num_train]
    val_files = rel_files[num_train:]

    train_data = SliceData(
        files=train_files,
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        num_frames_per_example=args.num_frames_per_example,
        clips_factors=clips_factors
    )
    dev_data = SliceData(
        files=val_files,
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        num_frames_per_example=args.num_frames_per_example,
        clips_factors=clips_factors
    )
    return dev_data, train_data

def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 8)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=args.batch_size,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer, loader_len):
    model.train()
    avg_loss = 0.

    #The interpolation gap is only meaningful if we do not use projection
    if epoch < 20:
        model.subsampling.interp_gap = 32
    elif epoch == 20:
        model.subsampling.interp_gap = 16
    elif epoch == 30:
        model.subsampling.interp_gap = 8
    elif epoch == 40:
        model.subsampling.interp_gap = 4
    elif epoch == 46:
        model.subsampling.interp_gap = 2
    elif epoch == 50:
        model.subsampling.interp_gap = 2

    start_epoch = time.perf_counter()
    print(f'Imposing Machine Constraints: a_max={args.a_max}, v_max={args.v_max}')
    for iter, data in data_loader:

        optimizer.zero_grad()

        input, target, mean, std = data
        input = input.to(args.device)

        target = transforms.complex_abs(input)

        if args.noise: #add noise to trajectory initialialization
            noise_factor = 10e-6
            noise_precentage = 0.99

            for j in range(8):
                theta = np.pi / args.n_shots
                for i in range(args.n_shots):
                    Lx = torch.arange(-144 / 2, 144 / 2, 144 / 513).float()
                    Ly = torch.arange(-384 / 2, 384 / 2, 384 / 513).float()
                    model.subsampling.x.data[j, i, :, 0] = Lx * np.cos(theta * i)
                    model.subsampling.x.data[j, i, :, 1] = Ly * np.sin(theta * i)
                    for k in range(513):
                        num = np.random.rand()
                        if num > noise_precentage:
                            sign = np.random.randint(0, 2)
                            if sign:
                                model.subsampling.x.data[j, i, k, 1] -= np.random.rand() * noise_factor
                                model.subsampling.x.data[j, i, k, 0] -= np.random.rand() * noise_factor
                            else:
                                model.subsampling.x.data[j, i, k, 1] += np.random.rand() * noise_factor
                                model.subsampling.x.data[j, i, k, 0] += np.random.rand() * noise_factor

        output = model(input)

        # Loss on trajectory vel and acc
        x = model.get_trajectory()
        v, a = get_vel_acc(x)
        acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max).abs() + 1e-8, 2)))
        vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max).abs() + 1e-8, 2)))

        # target loss

        rec_loss = F.mse_loss(output.to(torch.float64), target.to(
            torch.float64))

        #weigh kinematic los in overall loss
        loss = args.rec_weight * rec_loss + args.vel_weight * vel_loss + args.acc_weight * acc_loss

        if args.traj_dropout: #apply traj dropout - only learn a portion of points for each trajectory (randomly)
            fixed_percentage = 0.1
            orig_x = torch.clone(x)
            mask = torch.rand_like(orig_x) < fixed_percentage

        loss.backward()

        #Apply trajectory freezing - zero grad on all trajectories corresponding to any frame besides the one currently optimized
        if args.traj_freeze and args.trajectory_learning and not model.subsampling.curr_frame == args.num_frames_per_example:
            if model.subsampling.curr_frame == args.num_frames_per_example-1:
                model.subsampling.x.grad[0:args.num_frames_per_example-1] *= 0
            elif model.subsampling.curr_frame == 0:
                model.subsampling.x.grad[1:] *= 0

            else:
                model.subsampling.x.grad[0:model.subsampling.curr_frame] *= 0
                model.subsampling.x.grad[model.subsampling.curr_frame + 1:] *= 0

        optimizer.step()

        if args.traj_dropout:
            model.subsampling.x.data = orig_x * mask + model.get_trajectory() * (~mask)

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        psnr_train, vif_train, fsim_train = get_metrics(target, output,args.vif,args.fsim)


        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{loader_len:4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'rec_loss: {rec_loss:.4g}, vel_loss: {vel_loss:.4g}, acc_loss: {acc_loss:.4g}'
                f' PSNR: {psnr_train}'
            )
        if iter == loader_len - 1:
            break

    return avg_loss, time.perf_counter() - start_epoch, rec_loss, vel_loss, acc_loss, psnr_train


def evaluate(args, epoch, model, data_loader, writer, dl_len, train_loss=None, train_rec_loss=None, train_vel_loss=None,
             train_acc_loss=None, psnr_train=None):
    model.eval()
    losses = []
    psnrs = []
    vifs = []
    fsims = []
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in data_loader:

                input, target, mean, std = data
                input = input.to(args.device)
                target = transforms.complex_abs(input)
                output = model(input)

                loss = F.mse_loss(output, target)
                psnr_dev, vif_dev, fsim_dev = get_metrics(target, output,args.vif,args.fsim)

                psnrs.append(psnr_dev)

                if vif_dev is not None:
                    vifs.append(vif_dev)
                if fsim_dev is not None:
                    fsims.append(fsim_dev)

                losses.append(loss.item())

                if iter == dl_len - 1:
                    break

            x = model.get_trajectory()
            v, a = get_vel_acc(x)
            acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
            vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
            rec_loss = np.mean(losses)

            psnr = np.mean(psnrs)
            vif = np.mean(vifs) if len(vifs) else None
            fsim = np.mean(vifs) if len(fsims) else None

            if train_rec_loss is None:
                writer.add_scalars('Rec_Loss', {'val': rec_loss}, epoch)
            else:
                writer.add_scalars('Rec_Loss', {'val': rec_loss, 'train': train_rec_loss}, epoch)
            if train_acc_loss is None:
                writer.add_scalars('Acc_Loss', {'val': acc_loss.detach().cpu().numpy()}, epoch)
            else:
                writer.add_scalars('Acc_Loss', {'val': acc_loss.detach().cpu().numpy(), 'train': train_acc_loss}, epoch)
            if train_vel_loss is None:
                writer.add_scalars('Vel_Loss', {'val': vel_loss.detach().cpu().numpy()}, epoch)
            else:
                writer.add_scalars('Vel_Loss', {'val': vel_loss.detach().cpu().numpy(), 'train': train_vel_loss}, epoch)
            if train_loss is None:
                writer.add_scalars('Total_Loss', {
                    'val': rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy()}, epoch)
            else:
                writer.add_scalars('Total_Loss',
                                   {'val': rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy(),
                                    'train': train_loss}, epoch)
            if psnr_train is None:
                writer.add_scalars('PSNR', {'val': psnr_dev}, epoch)
            else:
                writer.add_scalars('PSNR', {'val': psnr_dev, 'train': psnr_train}, epoch)

            if len(vifs):
                writer.add_scalars('VIF', {'val': vif}, epoch)

            if len(fsims):
                writer.add_scalars('FSIM', {'val': fsim}, epoch)

        x = model.get_trajectory()
        v, a = get_vel_acc(x)

        writer.add_figure('Trajectory', plot_trajectory(x.detach().cpu().numpy()), epoch)
        writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)
        writer.add_figure('Accelerations_plot', plot_acc(a.cpu().numpy(), args.a_max), epoch)
        writer.add_figure('Velocity_plot', plot_acc(v.cpu().numpy(), args.v_max), epoch)
        writer.add_text('Coordinates', str(x.detach().cpu().numpy()).replace(' ', ','), epoch)
    if epoch == 0:
        return None, time.perf_counter() - start, None
    else:
        return np.mean(losses), time.perf_counter() - start, psnr

def plot_scatter(x):
    if len(x.shape) == 4:
        return plot_scatters(x)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1], '.')
    return fig


def plot_scatters(x):
    fig = plt.figure(figsize=[10, 10])
    for frame in range(x.shape[0]):
        ax = fig.add_subplot(2, x.shape[0]//2, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1], '.')
            ax.axis([-165, 165, -165, 165])
    return fig


def plot_trajectory(x):
    if len(x.shape) == 4:
        return plot_trajectories(x)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1])
    return fig


def plot_trajectories(x):
    fig = plt.figure(figsize=[10, 10])
    for frame in range(x.shape[0]):
        ax = fig.add_subplot(2, x.shape[0]//2, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1])
            ax.axis([-165, 165, -165, 165])
    return fig


def plot_acc(a, a_max=None):
    fig, ax = plt.subplots(2, sharex=True)
    for i in range(a.shape[0]):
        ax[0].plot(a[i, :, 0])
        ax[1].plot(a[i, :, 1])
    if a_max is not None:
        limit = np.ones(a.shape[1]) * a_max
        ax[1].plot(limit, color='red')
        ax[1].plot(-limit, color='red')
        ax[0].plot(limit, color='red')
        ax[0].plot(-limit, color='red')
    return fig


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        videos_display = []
        for example in image.transpose(1, 0):
            grid = torchvision.utils.make_grid(example, nrow=3, pad_value=1)
            videos_display.append(grid)
        vid_tensor = torch.stack(videos_display, dim=0).unsqueeze(0)
        writer.add_video(tag, vid_tensor, fps=10, global_step=epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):

            input, target, mean, std = data
            input = input.to(args.device)
            target = target.unsqueeze(2).to(args.device)

            save_image(target, 'Target')
            if epoch != 0:
                output = model(input)


                output = output.unsqueeze(2)

                corrupted = model.subsampling(input).unsqueeze(2)
                cor_all = transforms.root_sum_of_squares(corrupted, -1)

                save_image(output, 'Reconstruction')
                save_image(corrupted[..., 0], 'Corrupted_real')
                save_image(corrupted[..., 1], 'Corrupted_im')
                save_image(cor_all, 'Corrupted')
                save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, name=None):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=(exp_dir + '/model.pt') if name is None else name
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')


def build_model(args):
    model = Subsampling_Model(
        in_chans=args.num_frames_per_example,
        out_chans=args.num_frames_per_example,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR,
        projection_iters=args.proj_iters,
        project=args.project,
        n_shots=args.n_shots,
        interp_gap=args.interp_gap,
        multiple_trajectories=args.multi_traj
    ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, model):
    optimizer = torch.optim.Adam([{'params': model.subsampling.parameters(), 'lr': args.sub_lr},
                                  {'params': model.reconstruction_model.parameters()}], args.lr)
    return optimizer


def train(args,order=None, init_trajs=[]):
    args.v_max = args.gamma * args.G_max * args.FOV * args.dt
    args.a_max = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3

    args.exp_dir = f'summary_{order}/{args.test_name}'if order is not None else f'summary/{args.test_name}'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)

    if args.resume: #load trained model (for evaluation or to keep training)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model)
        start_epoch = 0
    logging.info(args)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    enum_train = itertools.cycle(enumerate(train_loader))
    enum_val = itertools.cycle(enumerate(dev_loader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    dev_loss, dev_time, psnr_dev = evaluate(args, 0, model, dev_loader, writer, len(dev_loader))
    best_dev_loss = float('inf')
    visualize(args, 0, model, display_loader, writer)

    ####################################################################################
    if args.recons_resets and args.traj_freeze:
        for i in range(len(init_trajs)):
            model.subsampling.x.data[i] = init_trajs[i]
        if len(init_trajs) and order <args.num_frames_per_example:
            model.subsampling.x.data[order] = init_trajs[-1]
    elif args.recons_resets:
        if len(init_trajs):
            model.subsampling.x.data = init_trajs


    model.subsampling.curr_frame = 0 if order is None else order

    ####################################################################################

    for epoch in range(start_epoch, args.num_epochs):

        if args.traj_freeze and not args.recons_resets:
            if epoch and not epoch % (args.num_epochs // (args.num_frames_per_example + 1)):

                if model.subsampling.curr_frame != args.num_frames_per_example - 1:
                    model.subsampling.x.data[model.subsampling.curr_frame + 1] = model.subsampling.x.data[
                        model.subsampling.curr_frame]
                model.subsampling.curr_frame = (model.subsampling.curr_frame + 1) % args.num_frames_per_example
                optimizer.param_groups[0]['lr'] = 0.2

            if model.subsampling.curr_frame < args.num_frames_per_example:
                print(f"Optimizing Frame: {model.subsampling.curr_frame}")


        start = time.time()

        if epoch and not epoch % args.lr_step_size:
            optimizer.param_groups[1]['lr'] *= args.lr_gamma

        if epoch and not epoch % args.sub_lr_time:
            optimizer.param_groups[0]['lr'] = max(args.sub_lr_stepsize * optimizer.param_groups[0]['lr'], 5e-4)

        train_loss, train_time, train_rec_loss, train_vel_loss, train_acc_loss, psnr_train = train_epoch(
            args, epoch,
            model,
            enum_train,
            optimizer,
            writer,
            len(train_loader))
        dev_loss, dev_time, psnr_dev = evaluate(args, epoch + 1, model, enum_val, writer, len(dev_loader),
                                                          train_loss,
                                                          train_rec_loss, train_vel_loss, train_acc_loss, psnr_train,
                                            )

        visualize(args, epoch + 1, model, display_loader, writer)

        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s'
            f'DevPSNR = {psnr_dev:.4g}'
        )
        end = time.time() - start
        print(f'epoch time: {end}')
    print(args.test_name)
    print(f'Training done for order {order}, best epoch: {best_epoch}')
    writer.close()

    if args.recons_resets and args.traj_freeze and model.subsampling.curr_frame < args.num_frames_per_example:
        init_trajs.append(model.subsampling.x.data[order])
    elif args.recons_resets:
        init_trajs = model.subsampling.x.data
    return init_trajs

def run():
    args = create_arg_parser().parse_args()

    if args.recons_resets:
        traj_ls = []
        for i in range(args.num_frames_per_example+1):
            traj_ls = train(args,i, traj_ls)

    else:
        args.num_epochs *= (args.num_frames_per_example+1)
        train(args)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=55555, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--data-path', type=pathlib.Path,
                      default='/home/tamir.shor/MRI/aug', help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.,
                      help='Fraction of total volumes to include')
    parser.add_argument('--test-name', type=str, default='test/', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='output/',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--ocmr-path',type=str,default='/home/tamir/OCMR/OCMR/ocmr_data_attributes.csv', help="Path to ocmr attributes csv")
    parser.add_argument('--fsim',action='store_true',help="calculate fsim values (advised to only use this over a trained model, not in training - computing fsim takes ~30 secs)")
    parser.add_argument('--vif', action='store_true',
                        help="calculate vif values (advised to only use this over a trained model, not in training - computing vif takes ~30 secs)")

    # model parameters
    parser.add_argument('--num-pools', type=int, default=4, help='Number of Reconstruction model pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=64, help='Number of Reconstruction model channels')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', default=10, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')

    # optimization parameters
    parser.add_argument('--batch-size', default=12, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=35, help='Number of training epochs per frame')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for reconstruction model')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Period of learning rate decay for reconstruction model')
    parser.add_argument('--lr-gamma', type=float, default=0.005,
                        help='Multiplicative factor of reconstruction model learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    # trajectory learning parameters
    parser.add_argument('--sub-lr', type=float, default=0.2, help='learning rate of the sub-samping layer')
    parser.add_argument('--sub-lr-time', type=float, default=3,
                        help='learning rate decay timestep of the sub-sampling layer')
    parser.add_argument('--sub-lr-stepsize', type=float, default=0.7,
                        help='learning rate decay step size of the sub-sampling layer')

    parser.add_argument('--trajectory-learning', default=True,
                        help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')

    #MRI Machine Parameters
    parser.add_argument('--acc-weight', type=float, default=1e-2, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--gamma', type=float, default=42576, help='gyro magnetic ratio - kHz/T')
    parser.add_argument('--G-max', type=float, default=40, help='maximum gradient (peak current) - mT/m')
    parser.add_argument('--S-max', type=float, default=180, help='maximum slew-rate - T/m/s')
    parser.add_argument('--FOV', type=float, default=0.2, help='Field Of View - in m')
    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--a-max', type=float, default=0.17, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=3.4, help='maximum velocity')
    parser.add_argument('--initialization', type=str, default='radial',
                        help='Trajectory initialization when using PILOT (spiral, EPI, rosette, uniform, gaussian).')
    parser.add_argument('--SNR', action='store_true', default=False,
                        help='add SNR decay')
    #modelization parameters
    parser.add_argument('--n-shots', type=int, default=16,
                        help='Number of shots')
    parser.add_argument('--interp_gap', type=int, default=10,
                        help='number of interpolated points between 2 parameter points in the trajectory')
    parser.add_argument('--num_frames_per_example', type=int, default=8, help='num frames per example')
    parser.add_argument('--boost', action='store_true', default=False, help='boost to equalize num examples per file')

    parser.add_argument('--project', action='store_true', default=True, help='Use projection to impose kinematic constraints.'
                                                                             'If false, use interpolation and penalty (original PILOT paper).')
    parser.add_argument('--proj_iters', default=10e1, help='Number of iterations for each projection run.')
    parser.add_argument('--multi_traj', action='store_true', default=True, help='allow different trajectory per frame')
    parser.add_argument('--augment', action='store_true', default=False, help='Use augmented files. data-path argument from should lead to augmented '
                                                                             'files generated by the augment script. If false,'
                                                                             'path should lead to relevant ocmr dataset')
    parser.add_argument('--noise', action='store_true', default=False, help='add noise to traj.')
    parser.add_argument('--traj-dropout', action='store_true', default=False, help='randomly fix traj coordinates.')
    parser.add_argument('--recons_resets', action='store_true', default=False, help='Use Reconstruction Resets.')
    parser.add_argument('--traj_freeze', action='store_true', default=False, help='Use Trajectory Freezing.')
    return parser




if __name__ == '__main__':
    run()

