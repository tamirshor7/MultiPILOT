import torch.nn as nn
from tricorder.torch.transforms import Interpolator

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

import sys

import numpy as np
import torch
import DRCN.torchkbnufft as tkbn

import numpy as np
import torch

from typing import Optional, Sequence, Union

import torch
import DRCN.torchkbnufft.functional as tkbnF
from torch import Tensor

from models.rec_models.resnet_utils import init_fn,kb_table_interp,kb_table_interp_adjoint


def calc_density_compensation_function(
    ktraj: Tensor,
    im_size: Sequence[int],
    num_iterations: int = 10,
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2**10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
) -> Tensor:
    """Numerical density compensation estimation.
    This function has optional parameters for initializing a NUFFT object. See
    :py:class:`~torchkbnufft.KbInterp` for details.
    * :attr:`ktraj` should be of size ``(len(grid_size), klength)`` or
      ``(N, len(grid_size), klength)``, where ``klength`` is the length of the
      k-space trajectory.
    Based on the `method of Pipe
    <https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V>`_.
    This code was contributed by Chaithya G.R.
    Args:
        ktraj: k-space trajectory (in radians/voxel).
        im_size: Size of image with length being the number of dimensions.
        num_iterations: Number of iterations.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension. Default: ``6``
        n_shift: Size for fftshift. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
    Returns:
        The density compensation coefficients for ``ktraj``.
    Examples:
        >>> data = torch.randn(1, 1, 12) + 1j * torch.randn(1, 1, 12)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> dcomp = tkbn.calc_density_compensation_function(omega, (8, 8))
        >>> adjkb_ob = tkbn.KbNufftAdjoint(im_size=(8, 8))
        >>> image = adjkb_ob(data * dcomp, omega)
    """
    device = ktraj.device
    batch_size = 1

    if ktraj.ndim not in (2, 3):
        raise ValueError("ktraj must have 2 or 3 dimensions")

    if ktraj.ndim == 3:
        if ktraj.shape[0] == 1:
            ktraj = ktraj[0]
        else:
            batch_size = ktraj.shape[0]

    # init nufft variables
    (
        tables,
        _,
        grid_size_t,
        n_shift_t,
        numpoints_t,
        offsets_t,
        table_oversamp_t,
        _,
        _,
    ) = init_fn(
        im_size=im_size,
        grid_size=grid_size,
        numpoints=numpoints,
        n_shift=n_shift,
        table_oversamp=table_oversamp,
        kbwidth=kbwidth,
        order=order,
        dtype=ktraj.dtype,
        device=device,
    )

    test_sig = torch.ones(
        [batch_size, 1, ktraj.shape[-1]], dtype=tables[0].dtype, device=device
    )
    for _ in range(num_iterations):

        new_sig = kb_table_interp(
            image=kb_table_interp_adjoint(
                data=test_sig,
                omega=ktraj,
                tables=tables,
                n_shift=n_shift_t,
                numpoints=numpoints_t,
                table_oversamp=table_oversamp_t,
                offsets=offsets_t,
                grid_size=grid_size_t,
            ),
            omega=ktraj,
            tables=tables,
            n_shift=n_shift_t,
            numpoints=numpoints_t,
            table_oversamp=table_oversamp_t,
            offsets=offsets_t,
        )

        test_sig = test_sig / torch.abs(new_sig)

    return test_sig

#########
# Fourier Transforms
#########
def fftNc_pyt(data, dim=(-2, -1), norm="ortho"):
    data = torch.ifftshift(data, dim=dim)
    data = torch.fftn(data, dim=dim, norm=norm)
    data = torch.fftshift(data, dim=dim)
    return data


def ifftNc_pyt(data, dim=(-2, -1), norm="ortho"):
    data = torch.ifftshift(data, dim=dim)
    data = torch.ifftn(data, dim=dim, norm=norm)
    data = torch.fftshift(data, dim=dim)
    return data


def fftNc_np(data, axes=(-2, -1), norm="ortho"):
    data = np.fft.ifftshift(data, axes=axes)
    data = np.fft.fftn(data, axes=axes, norm=norm)
    data = np.fft.fftshift(data, axes=axes)
    return data


def ifftNc_np(data, axes=(-2, -1), norm="ortho"):
    data = np.fft.ifftshift(data, axes=axes)
    data = np.fft.ifftn(data, axes=axes, norm=norm)
    data = np.fft.fftshift(data, axes=axes)
    return data


def fftNc(data, dim=(-2, -1), norm="ortho"):
    if type(data) is torch.Tensor:
        return fftNc_pyt(data=data, dim=dim, norm=norm)
    else:
        return fftNc_np(data=data, axes=dim, norm=norm)


def ifftNc(data, dim=(-2, -1), norm="ortho"):
    if type(data) is torch.Tensor:
        return ifftNc_pyt(data=data, dim=dim, norm=norm)
    else:
        # TODO: handle no centering cases
        return ifftNc_np(data=data, axes=dim, norm=norm)
#########

#########
# Normalizations
#########


def fnorm_pyt(x):
    return x/torch.abs(x).max()


def fnorm_np(x):
    return x/np.abs(x).max()


def fnorm(x):
    if type(x) is torch.Tensor:
        return fnorm_pyt(x=x)
    else:
        return fnorm_np(x=x)

class DataConsistency():
    def __init__(self, isRadial=False, metadict=None):
        self.isRadial = isRadial
        self.metadict = metadict
        # if isRadial:
        #     sys.exit("DataConsistency: Not working for Radial yet, due to raw kSpace troubles")

    # def cartesian_fastmri(self, out_ksp, full_ksp, under_ksp, mask): #torch.where doesn't work with complex
    #     if mask is None:
    #         mask = self.mask
    #     mask = mask.to(out_ksp.device)
    #     missing_mask = 1-mask
    #     missing_ksp = torch.where(missing_mask == 0, torch.Tensor([0]).to(out_ksp.device), out_ksp)
    #     if under_ksp is None:
    #         under_ksp = torch.where(mask == 0, torch.Tensor([0]).to(full_ksp.device), full_ksp)
    #     out_corrected_ksp = under_ksp + missing_ksp
    #     return out_corrected_ksp

    def cartesian_Ndmask(self, out_ksp, full_ksp, under_ksp, metadict):
        mask = metadict["mask"] if type(metadict) is dict else metadict
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.to(out_ksp.device)
        if len(full_ksp.shape) == 3 and len(mask.shape) == 2:  # TODO: do it nicely, its too strict now
            mask = mask.unsqueeze(-1)
        missing_mask = 1-mask
        missing_ksp = out_ksp * missing_mask
        if under_ksp is None:
            under_ksp = full_ksp * mask
        out_corrected_ksp = under_ksp + missing_ksp
        return out_corrected_ksp

    def radial(self, out_ksp, full_ksp, under_ksp, metadict, device="cpu"): #TODO: switch to cuda
        # om = torch.from_numpy(metadict['om'].transpose()).to(torch.float).to(device)
        # invom = torch.from_numpy(metadict['invom'].transpose()).to(torch.float).to(device)
        # fullom = torch.from_numpy(metadict['fullom'].transpose()).to(torch.float).to(device)
        # # dcf = torch.from_numpy(metadict['dcf'].squeeze())
        # dcfFullRes = torch.from_numpy(metadict['dcfFullRes'].squeeze()).to(torch.float).to(device)
        baseresolution = out_ksp.shape[0]*2
        Nd = (baseresolution, baseresolution)
        imsize = (384,144)#out_ksp.shape[:2]

        nufft_ob = tkbn.KbNufft(
            im_size=imsize,
            grid_size=Nd,
        )#.to(torch.complex64).to(device)
        adjnufft_ob = tkbn.AdjKbNufft(
            im_size=imsize,
            grid_size=Nd,
        )#.to(torch.complex64).to(device)

        # intrp_ob = tkbn.KbInterp(
        #     im_size=imsize,
        #     grid_size=Nd,
        # ).to(torch.complex64).to(device)

        out_img = torch.Tensor(ifftNc(data=out_ksp, dim=(0, 1), norm="ortho")).to(device)
        full_img = torch.Tensor(ifftNc(data=full_ksp, dim=(0, 1), norm="ortho")).to(device)

        if len(out_img.shape) == 3:
            out_img = torch.permute(out_img, dims=(2, 0, 1)).unsqueeze(1)
            full_img = torch.permute(full_img, dims=(2, 0, 1)).unsqueeze(1)
        else:
            out_img = out_img.unsqueeze(0).unsqueeze(0)
            full_img = full_img.unsqueeze(0).unsqueeze(0)

        # out_img = torch.permute(out_ksp, dims=(2,0,1)).unsqueeze(1).to(device)
        # full_img = torch.permute(full_ksp, dims=(2,0,1)).unsqueeze(1).to(device)

        spokelength = full_img.shape[-1] * 2
        grid_size = (spokelength, spokelength)
        nspokes = 512

        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
        kx = np.zeros(shape=(spokelength, nspokes))
        ky = np.zeros(shape=(spokelength, nspokes))
        ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
        for i in range(1, nspokes):
            kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
            ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

        ky = np.transpose(ky)
        kx = np.transpose(kx)

        fullom = torch.from_numpy(np.stack((ky.flatten(), kx.flatten()), axis=0)).to(
            torch.float).to(device)
        om = fullom[:, :30720]
        invom = fullom[:, 30720:]
        dcfFullRes = calc_density_compensation_function(
            ktraj=fullom, im_size=imsize).to(device)

        yUnder = nufft_ob(full_img, om, norm="ortho")
        yMissing = nufft_ob(out_img, invom, norm="ortho")
        # yUnder = intrp_ob(full_img, om)
        # yMissing = intrp_ob(out_img, invom)
        yCorrected = torch.concat((yUnder, yMissing), dim=-1)
        yCorrected = dcfFullRes * yCorrected
        out_corrected_img = adjnufft_ob(
            yCorrected, fullom, norm="ortho").squeeze()

        out_corrected_img = torch.abs(out_corrected_img)
        out_corrected_img = (out_corrected_img - out_corrected_img.min()) / \
            (out_corrected_img.max() - out_corrected_img.min())

        if len(out_corrected_img.shape) == 3:
            out_corrected_img = torch.permute(out_corrected_img, dims=(1, 2, 0))

        out_corrected_ksp = fftNc(
            data=out_corrected_img, dim=(0, 1), norm="ortho").cpu()
        return out_corrected_ksp

    def apply(self, out_ksp, full_ksp, under_ksp, metadict=None):
        if metadict is None:
            metadict = self.metadict
        if self.isRadial:
            return self.radial(out_ksp, full_ksp, under_ksp, metadict)
        else:
            return self.cartesian_Ndmask(out_ksp, full_ksp, under_ksp, metadict)
class ResidualBlock(nn.Module):
    def __init__(self, in_features, drop_prob=0.2):
        super(ResidualBlock, self).__init__()

        conv_block = [layer_pad(1),
                      layer_conv(in_features, in_features, 3),
                      layer_norm(in_features),
                      act_relu(),
                      layer_drop(p=drop_prob, inplace=True),
                      layer_pad(1),
                      layer_conv(in_features, in_features, 3),
                      layer_norm(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownsamplingBlock, self).__init__()

        conv_block = [layer_conv(in_features, out_features, 3, stride=2, padding=1),
                      layer_norm(out_features),
                      act_relu()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, mode="convtrans", interpolator=None, post_interp_convtrans=False):
        super(UpsamplingBlock, self).__init__()

        self.interpolator = interpolator
        self.mode = mode
        self.post_interp_convtrans = post_interp_convtrans
        if self.post_interp_convtrans:
            self.post_conv = layer_conv(out_features, out_features, 1)

        if mode == "convtrans":
            conv_block = [layer_convtrans(
                in_features, out_features, 3, stride=2, padding=1, output_padding=1), ]
        else:
            conv_block = [layer_pad(1),
                          layer_conv(in_features, out_features, 3), ]
        conv_block += [layer_norm(out_features),
                       act_relu()]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x, out_shape=None):
        if self.mode == "convtrans":
            if self.post_interp_convtrans:
                x = self.conv_block(x)
                if x.shape[2:] != out_shape:
                    return self.post_conv(self.interpolator(x, out_shape))
                else:
                    return x
            else:
                return self.conv_block(x)
        else:
            return self.conv_block(self.interpolator(x, out_shape))


class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, do_batchnorm=False, res_drop_prob=0.2,
                 is_replicatepad=0, out_act="sigmoid", forwardV=0, upinterp_algo='convtrans', post_interp_convtrans=False, is3D=False):  # should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256
        super(ResNet, self).__init__()

        layers = {}
        if is3D:
            layers["layer_conv"] = nn.Conv3d
            layers["layer_convtrans"] = nn.ConvTranspose3d
            if do_batchnorm:
                layers["layer_norm"] = nn.BatchNorm3d
            else:
                layers["layer_norm"] = nn.InstanceNorm3d
            layers["layer_drop"] = nn.Dropout3d
            if is_replicatepad == 0:
                layers["layer_pad"] = nn.ReflectionPad3d
            elif is_replicatepad == 1:
                layers["layer_pad"] = nn.ReplicationPad3d
            layers["interp_mode"] = 'trilinear'
        else:
            layers["layer_conv"] = nn.Conv2d
            layers["layer_convtrans"] = nn.ConvTranspose2d
            if do_batchnorm:
                layers["layer_norm"] = nn.BatchNorm2d
            else:
                layers["layer_norm"] = nn.InstanceNorm2d
            layers["layer_drop"] = nn.Dropout2d
            if is_replicatepad == 0:
                layers["layer_pad"] = nn.ReflectionPad2d
            elif is_replicatepad == 1:
                layers["layer_pad"] = nn.ReplicationPad2d
            layers["interp_mode"] = 'bilinear'
        if is_relu_leaky:
            layers["act_relu"] = nn.PReLU
        else:
            layers["act_relu"] = nn.ReLU
        globals().update(layers)

        self.forwardV = forwardV
        self.upinterp_algo = upinterp_algo

        interpolator = Interpolator(
            mode=layers["interp_mode"] if self.upinterp_algo == "convtrans" else self.upinterp_algo)

        # Initial convolution block
        intialConv = [layer_pad(3),
                      layer_conv(in_channels, starting_nfeatures, 7),
                      layer_norm(starting_nfeatures),
                      act_relu()]

        # Downsampling [need to save the shape for upsample]
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2
        for _ in range(updown_blocks):
            downsam.append(DownsamplingBlock(in_features, out_features))
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        resblocks = []
        for _ in range(res_blocks):
            resblocks += [ResidualBlock(in_features, res_drop_prob)]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            upsam.append(UpsamplingBlock(in_features, out_features,
                         self.upinterp_algo, interpolator, post_interp_convtrans))
            in_features = out_features
            out_features = in_features//2

        # Output layer
        finalconv = [layer_pad(3),
                     layer_conv(starting_nfeatures, out_channels, 7), ]

        if out_act == "sigmoid":
            finalconv += [nn.Sigmoid(), ]
        elif out_act == "relu":
            finalconv += [act_relu(), ]
        elif out_act == "tanh":
            finalconv += [nn.Tanh(), ]

        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = nn.ModuleList(downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = nn.ModuleList(upsam)
        self.finalconv = nn.Sequential(*finalconv)

        if self.forwardV == 0:
            self.forward = self.forwardV0
        elif self.forwardV == 1:
            self.forward = self.forwardV1
        elif self.forwardV == 2:
            self.forward = self.forwardV2
        elif self.forwardV == 3:
            self.forward = self.forwardV3
        elif self.forwardV == 4:
            self.forward = self.forwardV4
        elif self.forwardV == 5:
            self.forward = self.forwardV5

    def forwardV0(self, x):
        # v0: Original Version
        x = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(x.shape[2:])
            x = downblock(x)
        x = self.resblocks(x)
        for i, upblock in enumerate(self.upsam):
            x = upblock(x, shapes[-1-i])
        return self.finalconv(x)

    def forwardV1(self, x):
        # v1: input is added to the final output
        out = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return x + self.finalconv(out)

    def forwardV2(self, x):
        # v2: residual of v1 + input to the residual blocks added back with the output
        out = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return x + self.finalconv(out)

    def forwardV3(self, x):
        # v3: residual of v2 + input of the initial conv added back with the output
        out = x + self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return x + self.finalconv(out)

    def forwardV4(self, x):
        # v4: residual of v3 + output of the initial conv added back with the input of final conv
        iniconv = x + self.intialConv(x)
        shapes = []
        if len(self.downsam) > 0:
            for i, downblock in enumerate(self.downsam):
                if i == 0:
                    shapes.append(iniconv.shape[2:])
                    out = downblock(iniconv)
                else:
                    shapes.append(out.shape[2:])
                    out = downblock(out)
        else:
            out = iniconv
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        out = iniconv + out
        return x + self.finalconv(out)

    def forwardV5(self, x):
        # v5: residual of v4 + individual down blocks with individual up blocks
        outs = [x + self.intialConv(x)]
        shapes = []
        for i, downblock in enumerate(self.downsam):
            shapes.append(outs[-1].shape[2:])
            outs.append(downblock(outs[-1]))
        outs[-1] = outs[-1] + self.resblocks(outs[-1])
        for i, upblock in enumerate(self.upsam):
            outs[-1] = upblock(outs[-1], shapes[-1-i])
            outs[-1] = outs[-2] + outs.pop()
        return x + self.finalconv(outs.pop())