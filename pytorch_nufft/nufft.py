from pytorch_nufft import util
import pytorch_nufft.interp as interp
import numpy
import torch
from data import transforms


def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))


def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)


def resize(input, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.
    Args:
        input (array): Input array.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.
    Returns:
        array: Zero-padded or cropped result.
    """

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape1, dtype=input.dtype)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output.reshape(oshape)


def fft(input, oshape=None, axes=None, norm='ortho'):
    ndim = input.ndim
    axes = _normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = resize(input, oshape)
    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
    output = torch.fft.fftshift(tmp, dim=axes)
    return output


def ifft(input, oshape=None, axes=None, norm='ortho'):
    ndim = input.ndim
    axes = _normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = resize(input, oshape)
    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
    output = torch.fft.fftshift(tmp, dim=axes)
    return output


def nufft(input, coord, oversamp=1.25, width=4.0, n=128, device='cuda'):
    #torch.use_deterministic_algorithms(True)
    ndim = coord.shape[-1]
    beta = numpy.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    output = input.clone()

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)

    # Zero-pad
    output = output / util.prod(input.shape[-ndim:]) ** 0.5
    output = util.resize(output, os_shape, device=device)

    # FFT
    output = output.permute(0, 1, 3, 4, 2)
    output = torch.view_as_real(
        fft(
            torch.view_as_complex(output.contiguous())
            , axes=range(-ndim, 0), norm=None),
    )
    output = output.permute(0, 1, 4, 2, 3)

    # Interpolate
    coord = _scale_coord(coord, input.shape, oversamp, device)
    kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
    output = interp.interpolate(output, width, kernel, coord, device)

    return output
    #torch.use_deterministic_algorithms(False)

def nufft_adjoint(input, coord, out_shape, oversamp=1.25, width=4.0, n=128, device='cuda'):
    ndim = coord.shape[-1]
    beta = numpy.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    out_shape = list(out_shape)

    os_shape = _get_oversamp_shape(out_shape, ndim, oversamp)

    # Gridding
    out_shape2 = out_shape.copy()
    os_shape2 = os_shape.copy()
    coord = _scale_coord(coord, out_shape2, oversamp, device)
    kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
    output = interp.gridding(input, os_shape2, width, kernel, coord, device)

    # IFFT
    output = output.permute(0, 1, 3, 4, 2)
    output = torch.view_as_real(
        ifft(
            torch.view_as_complex(output.contiguous())
            , axes=range(-ndim, 0), norm=None),
    )
    output = output.permute(0, 1, 4, 2, 3)

    # Crop
    output = util.resize(output, out_shape2, device=device)
    a = util.prod(os_shape2[-ndim:]) / util.prod(out_shape2[-ndim:]) ** 0.5
    output = output * a

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)

    return output


def _get_kaiser_bessel_kernel(n, width, beta, dtype, device):
    x = torch.arange(n, dtype=dtype) / n
    kernel = 1 / width * torch.tensor(numpy.i0(beta * (1 - x ** 2) ** 0.5), dtype=dtype)
    return kernel.to(device)


def _scale_coord(coord, shape, oversamp, device):
    ndim = coord.shape[-1]
    scale = torch.tensor(
        [_get_ugly_number(oversamp * i) / i for i in shape[-ndim:]], device=device)
    shift = torch.tensor(
        [_get_ugly_number(oversamp * i) // 2 for i in shape[-ndim:]], device=device, dtype=torch.float32)

    coord = scale * coord + shift

    return coord


def _get_ugly_number(n):
    if n <= 1:
        return n

    ugly_nums = [1]
    i2, i3, i5 = 0, 0, 0
    while (True):

        ugly_num = min(ugly_nums[i2] * 2,
                       ugly_nums[i3] * 3,
                       ugly_nums[i5] * 5)

        if ugly_num >= n:
            return ugly_num

        ugly_nums.append(ugly_num)
        if ugly_num == ugly_nums[i2] * 2:
            i2 += 1
        elif ugly_num == ugly_nums[i3] * 3:
            i3 += 1
        elif ugly_num == ugly_nums[i5] * 5:
            i5 += 1


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [_get_ugly_number(oversamp * i)
                                  for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta, device):
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = _get_ugly_number(oversamp * i)
        idx = torch.arange(i, dtype=output.dtype, device=device)

        # Calculate apodization
        apod = (beta ** 2 - (numpy.pi * width * (idx - i // 2) / os_i) ** 2) ** 0.5
        apod = apod / torch.sinh(apod)
        output = output * apod.reshape([i] + [1] * (-a - 1))

    return output