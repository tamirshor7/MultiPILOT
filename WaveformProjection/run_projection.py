from WaveformProjection import utils
from WaveformProjection.Projector import Projector
from WaveformProjection.Constraints import Constraints
from WaveformProjection import Evaluator
from scipy import io
from WaveformProjection.utils import interpolate
from math import ceil,floor
import torch
from matplotlib import pyplot as plt
import time


def pad_traj(tr,dest):
    dist = dest - tr.shape[1]
    out = torch.Tensor([]).to(tr.device)
    if dist < 0:
        stepsize = floor(tr.shape[1]/dest)
        for i in range(0,tr.shape[1],stepsize):
            if len(out) == dest-1:
                break
            out = torch.cat((out,torch.mean(tr[:,i:i+stepsize],dim=-1).unsqueeze(0)),dim=0)
        out = torch.cat((out,torch.mean(tr[:,i:],dim=-1).unsqueeze(0)),dim=0)
        return out
    if dist > 0:
        return torch.cat((tr, (tr[:, 0].unsqueeze(1)).repeat(dist, 1).reshape(-1, 2, 1).T.squeeze(0)), dim=1).T
    return tr.T



#Scanner Params([Lustig et al, IEEE TMI 2008])
def proj_handler(s0,num_iters, alpha = 3.4, beta = 0.17,disp=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Scanner Params([Lustig et al, IEEE TMI 2008])
    Gmax = 40e-3  # T/m
    Smax = 150e-3  # T/m/ms
    Kmax = 600  # m^-1

    gamma = 42.576 * 1e3  # kHz/T

    alpha = gamma * Gmax
    beta = gamma * Smax
    dt = 0.004  # sampling time in ms
    start = time.time()

    multitraj = False




    if disp:
        # plot initial interpolated trajectory (before projection)
        plt.style.use('seaborn')
        fig, ax = plt.subplots()
        ax.plot(s0[:, 0].to('cpu'), s0[:, 1].to('cpu'), color='#000000')
        ax.set_title('Before Projection')
        plt.tight_layout()
        plt.show()

    proj = Projector(num_iters=num_iters, device=device, display_res=disp, eps_inf=0, eps2=0)
    kc = [Constraints(Evaluator.LInf2Norm(), alpha, dt, 0), Constraints(Evaluator.LInf2Norm(), beta * dt, dt, 1)]
    proj.setKinematic(kc)


    if len(s0.shape) == 4:
        s1 = torch.zeros_like(s0)
        for i in range(s0.shape[0]):
            s1[i] = proj(s0[i])
    else:
        s1 = proj(s0)
    end = time.time()
    if disp:
        print(f'runtime: {end - start}')


    return s1
