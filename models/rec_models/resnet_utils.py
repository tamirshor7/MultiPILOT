import itertools
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import special
from scipy.sparse import coo_matrix
from torch import Tensor

DTYPE_MAP = [
    (torch.complex128, torch.float64),
    (torch.complex64, torch.float32),
    (torch.complex32, torch.float16),
]

from typing import List, Tuple

import torch
from torch import Tensor

import torch
from torch.autograd import Function

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def absolute(val: Tensor, dim: int = -1) -> Tensor:
    """Complex absolute value.
    Args:
        val: A tensor to have its absolute value computed.
        dim: An integer indicating the complex dimension (for real inputs
            only).
    Returns:
        The absolute value of ``val``.
    """
    if torch.is_complex(val):
        abs_val = torch.abs(val)
    else:
        if not val.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

        abs_val = torch.sqrt(
            val.select(dim, 0) ** 2 + val.select(dim, 1) ** 2
        ).unsqueeze(dim)

    return abs_val


def complex_mult(val1: Tensor, val2: Tensor, dim: int = -1) -> Tensor:
    """Complex multiplication.
    Args:
        val1: A tensor to be multiplied.
        val2: A second tensor to be multiplied.
        dim: An integer indicating the complex dimension (for real inputs
            only).
    Returns:
        ``val1 * val2``, where ``*`` executes complex multiplication.
    """
    if not val1.dtype == val2.dtype:
        raise ValueError("val1 has different dtype than val2.")

    if torch.is_complex(val1):
        val3 = val1 * val2
    else:
        if not val1.shape[dim] == val2.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

        real_a = val1.select(dim, 0)
        imag_a = val1.select(dim, 1)
        real_b = val2.select(dim, 0)
        imag_b = val2.select(dim, 1)

        val3 = torch.stack(
            (real_a * real_b - imag_a * imag_b, imag_a * real_b + real_a * imag_b), dim
        )

    return val3


def complex_sign(val: Tensor, dim: int = -1) -> Tensor:
    """Complex sign function value.
    Args:
        val: A tensor to have its complex sign computed.
        dim: An integer indicating the complex dimension (for real inputs
            only).
    Returns:
        The complex sign of ``val``.
    """
    is_complex = False
    if torch.is_complex(val):
        is_complex = True
        val = torch.view_as_real(val)
        dim = -1
    elif not val.shape[dim] == 2:
        raise ValueError("Real input does not have dimension size 2 at dim.")

    sign_val = torch.atan2(val.select(dim, 1), val.select(dim, 0))
    sign_val = imag_exp(sign_val, dim=dim, return_complex=is_complex)

    return sign_val


def conj_complex_mult(val1: Tensor, val2: Tensor, dim: int = -1) -> Tensor:
    """Complex multiplication, conjugating second input.
    Args:
        val1: A tensor to be multiplied.
        val2: A second tensor to be conjugated then multiplied.
        dim: An integer indicating the complex dimension (for real inputs
            only).
    Returns:
        ``val3 = val1 * conj(val2)``, where * executes complex multiplication.
    """
    if not val1.dtype == val2.dtype:
        raise ValueError("val1 has different dtype than val2.")

    if torch.is_complex(val1):
        val3 = val1 * val2.conj()
    else:
        if not val1.shape[dim] == val2.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

        real_a = val1.select(dim, 0)
        imag_a = val1.select(dim, 1)
        real_b = val2.select(dim, 0)
        imag_b = val2.select(dim, 1)

        val3 = torch.stack(
            (real_a * real_b + imag_a * imag_b, imag_a * real_b - real_a * imag_b), dim
        )

    return val3


def imag_exp(val: Tensor, dim: int = -1, return_complex: bool = True) -> Tensor:
    r"""Imaginary exponential.
    Args:
        val: A tensor to be exponentiated.
        dim: An integer indicating the complex dimension of the output (for
            real outputs only).
    Returns:
        ``val2 = exp(i*val)``, where ``i`` is ``sqrt(-1)``.
    """
    val2 = torch.stack((torch.cos(val), torch.sin(val)), -1)
    if return_complex:
        val2 = torch.view_as_complex(val2)

    return val2


def inner_product(val1: Tensor, val2: Tensor, dim: int = -1) -> Tensor:
    """Complex inner product.
    Args:
        val1: A tensor for the inner product.
        val2: A second tensor for the inner product.
        dim: An integer indicating the complex dimension (for real inputs
            only).
    Returns:
        The complex inner product of ``val1`` and ``val2``.
    """
    if not val1.dtype == val2.dtype:
        raise ValueError("val1 has different dtype than val2.")

    if not torch.is_complex(val1):
        if not val1.shape[dim] == val2.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

    inprod = conj_complex_mult(val2, val1, dim=dim)

    if not torch.is_complex(val1):
        inprod = torch.cat(
            (inprod.select(dim, 0).sum().view(1), inprod.select(dim, 1).sum().view(1))
        )
    else:
        inprod = torch.sum(inprod)

    return

# a little hacky but we don't have a function for detecting OMP
USING_OMP = "USE_OPENMP=ON" in torch.__config__.show()


def spmat_interp(
    image: Tensor, interp_mats: Union[Tensor, Tuple[Tensor, Tensor]]
) -> Tensor:
    """Sparse matrix interpolation backend."""
    if not isinstance(interp_mats, tuple):
        raise TypeError("interp_mats must be 2-tuple of (real_mat, imag_mat.")

    coef_mat_real, coef_mat_imag = interp_mats
    batch_size, num_coils = image.shape[:2]

    # sparse matrix multiply requires real
    image = torch.view_as_real(image)
    output_size = [batch_size, num_coils, -1]

    # we have to do these transposes because torch.mm requires first to be spmatrix
    image = image.reshape(batch_size * num_coils, -1, 2)
    real_griddat = image.select(-1, 0).t().contiguous()
    imag_griddat = image.select(-1, 1).t().contiguous()

    # apply multiplies
    kdat = torch.stack(
        [
            (
                torch.mm(coef_mat_real, real_griddat)
                - torch.mm(coef_mat_imag, imag_griddat)
            ).t(),
            (
                torch.mm(coef_mat_real, imag_griddat)
                + torch.mm(coef_mat_imag, real_griddat)
            ).t(),
        ],
        dim=-1,
    )

    return torch.view_as_complex(kdat).reshape(*output_size)


def spmat_interp_adjoint(
    data: Tensor,
    interp_mats: Union[Tensor, Tuple[Tensor, Tensor]],
    grid_size: Tensor,
) -> Tensor:
    """Sparse matrix interpolation adjoint backend."""
    if not isinstance(interp_mats, tuple):
        raise TypeError("interp_mats must be 2-tuple of (real_mat, imag_mat.")

    coef_mat_real, coef_mat_imag = interp_mats
    batch_size, num_coils = data.shape[:2]

    # sparse matrix multiply requires real
    data = torch.view_as_real(data)
    output_size = [batch_size, num_coils] + grid_size.tolist()

    # we have to do these transposes because torch.mm requires first to be spmatrix
    real_kdat = data.select(-1, 0).view(-1, data.shape[-2]).t().contiguous()
    imag_kdat = data.select(-1, 1).view(-1, data.shape[-2]).t().contiguous()
    coef_mat_real = coef_mat_real.t()
    coef_mat_imag = coef_mat_imag.t()

    # apply multiplies with complex conjugate
    image = torch.stack(
        [
            (
                torch.mm(coef_mat_real, real_kdat) + torch.mm(coef_mat_imag, imag_kdat)
            ).t(),
            (
                torch.mm(coef_mat_real, imag_kdat) - torch.mm(coef_mat_imag, real_kdat)
            ).t(),
        ],
        dim=-1,
    )

    return torch.view_as_complex(image).reshape(*output_size)


@torch.jit.script
def calc_coef_and_indices(
    tm: Tensor,
    base_offset: Tensor,
    offset_increments: Tensor,
    tables: List[Tensor],
    centers: Tensor,
    table_oversamp: Tensor,
    grid_size: Tensor,
    conjcoef: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Calculates interpolation coefficients and on-grid indices.
    Args:
        tm: Normalized frequency locations.
        base_offset: A tensor with offset locations to first elements in list
            of nearest neighbors.
        offset_increments: A tensor for how much to increment offsets.
        tables: A list of tensors tabulating a Kaiser-Bessel interpolation
            kernel.
        centers: A tensor with the center locations of the table for each
            dimension.
        table_oversamp: A tensor with the table size in each dimension.
        grid_size: A tensor with image dimensions.
        conjcoef: A boolean for whether to compute normal or complex conjugate
            interpolation coefficients (conjugate needed for adjoint).
    Returns:
        A tuple with interpolation coefficients and indices.
    """
    assert len(tables) == len(offset_increments)
    assert len(tables) == len(centers)

    # type values
    dtype = tables[0].dtype
    device = tm.device
    int_type = torch.long

    ktraj_len = tm.shape[1]

    # indexing locations
    gridind = base_offset + offset_increments.unsqueeze(1)
    distind = torch.round((tm - gridind.to(tm)) * table_oversamp.unsqueeze(1)).to(
        dtype=int_type
    )
    arr_ind = torch.zeros(ktraj_len, dtype=int_type, device=device)

    # give complex numbers if requested
    coef = torch.ones(ktraj_len, dtype=dtype, device=device)

    for d, (table, it_distind, center, it_gridind, it_grid_size) in enumerate(
        zip(tables, distind, centers, gridind, grid_size)
    ):  # spatial dimension
        if conjcoef:
            for i in range(it_distind.shape[0]):
                if not (it_distind[i] >=0 and it_distind[i]+center<len(table)):
                    it_distind[i] = 0

            coef = coef * table[it_distind + center].conj()
        else:
            coef = coef * table[it_distind + center]

        arr_ind = arr_ind + torch.remainder(it_gridind, it_grid_size).view(
            -1
        ) * torch.prod(grid_size[d + 1 :])

    return coef, arr_ind


@torch.jit.script
def table_interp_one_batch(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Table interpolation backend (see ``table_interp()``)."""
    dtype = image.dtype
    device = image.device
    int_type = torch.long

    grid_size = torch.tensor(image.shape[2:], dtype=int_type, device=device)

    # convert to normalized freq locs
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))

    # compute interpolation centers
    centers = torch.floor(torch.floor_divide(numpoints * table_oversamp,2)).to(dtype=int_type)

    # offset from k-space to first coef loc
    base_offset = 1 + (torch.floor_divide(tm - numpoints.unsqueeze(-1), 2.0)).to(dtype=int_type)

    # flatten image dimensions
    image = image.reshape(image.shape[0], image.shape[1], -1)
    kdat = torch.zeros(
        image.shape[0], image.shape[1], tm.shape[-1], dtype=dtype, device=device
    )
    # loop over offsets and take advantage of broadcasting
    for offset in offsets:
        coef, arr_ind = calc_coef_and_indices(
            tm=tm,
            base_offset=base_offset,
            offset_increments=offset,
            tables=tables,
            centers=centers,
            table_oversamp=table_oversamp,
            grid_size=grid_size,
        )

        # gather and multiply coefficients
        kdat += coef * image[:, :, arr_ind]

    # phase for fftshift
    return kdat * imag_exp(
        torch.sum(omega * n_shift.unsqueeze(-1), dim=-2, keepdim=True),
        return_complex=True,
    )


@torch.jit.script
def table_interp_multiple_batches(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Table interpolation with for loop over batch dimension."""
    kdat = []
    for (it_image, it_omega) in zip(image, omega):
        kdat.append(
            table_interp_one_batch(
                it_image.unsqueeze(0),
                it_omega,
                tables,
                n_shift,
                numpoints,
                table_oversamp,
                offsets,
            )
        )

    return torch.cat(kdat)


@torch.jit.script
def table_interp_fork_over_batchdim(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    num_forks: int,
) -> Tensor:
    """Table interpolation with forking over k-space."""
    # initialize the fork processes
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for (image_chunk, omega_chunk) in zip(
        image.split(num_forks), omega.split(num_forks)
    ):
        futures.append(
            torch.jit.fork(
                table_interp_multiple_batches,
                image_chunk,
                omega_chunk,
                tables,
                n_shift,
                numpoints,
                table_oversamp,
                offsets,
            )
        )

    # collect the results
    return torch.cat([torch.jit.wait(future) for future in futures])


@torch.jit.script
def table_interp_fork_over_kspace(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    num_forks: int,
) -> Tensor:
    """Table interpolation backend (see table_interp())."""
    # indexing is worst when we have repeated indices - let's spread them out
    klength = omega.shape[1]
    omega_chunks = [omega[:, ind:klength:num_forks] for ind in range(num_forks)]

    # initialize the fork processes
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for omega_chunk in omega_chunks:
        futures.append(
            torch.jit.fork(
                table_interp_one_batch,
                image,
                omega_chunk,
                tables,
                n_shift,
                numpoints,
                table_oversamp,
                offsets,
            )
        )

    kdat = torch.zeros(
        image.shape[0],
        image.shape[1],
        omega.shape[1],
        dtype=image.dtype,
        device=image.device,
    )

    # collect the results
    for ind, future in enumerate(futures):
        kdat[:, :, ind:klength:num_forks] = torch.jit.wait(future)

    return kdat


def table_interp(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    min_kspace_per_fork: int = 1024,
) -> Tensor:
    """Table interpolation backend.
    This interpolates from a gridded set of data to off-grid of data given by
    the coordinates in ``omega``.
    Args:
        image: Gridded data to interpolate from.
        omega: Fourier coordinates to interpolate to (in radians/voxel, -pi to
            pi).
        tables: List of tables for each image dimension.
        n_shift: Size of desired fftshift.
        numpoints: Number of neighbors in each dimension.
        table_oversamp: Size of table in each dimension.
        offsets: A list of offset values for interpolation.
        min_kspace_per_fork: Minimum number of k-space samples to use in each
            process fork. Only used for single trajectory on CPU.
    Returns:
        ``image`` interpolated to k-space locations at ``omega``.
    """
    if omega.ndim not in (2, 3):
        raise ValueError("omega must have 2 or 3 dimensions.")

    if omega.ndim == 3:
        if omega.shape[0] == 1:
            omega = omega[0]  # broadcast a single traj

    if omega.ndim == 3:
        if not omega.shape[0] == image.shape[0]:
            raise ValueError(
                "If omega has batch dim, omega batch dimension must match image."
            )

    # we fork processes for accumulation, so we need to do a bit of thread
    # management for OMP to make sure we don't oversubscribe (managment not
    # necessary for non-OMP)
    num_threads = torch.get_num_threads()
    factors = torch.arange(1, num_threads + 1)
    factors = factors[torch.remainder(torch.tensor(num_threads), factors) == 0]
    threads_per_fork = num_threads  # default fallback

    if omega.ndim == 3:
        # increase number of forks as long as it's not greater than batch size
        for factor in factors.flip(0):
            if num_threads // factor <= omega.shape[0]:
                threads_per_fork = int(factor)

        num_forks = num_threads // threads_per_fork

        if USING_OMP and image.device == torch.device("cpu"):
            torch.set_num_threads(threads_per_fork)
        kdat = table_interp_fork_over_batchdim(
            image, omega, tables, n_shift, numpoints, table_oversamp, offsets, num_forks
        )
        if USING_OMP and image.device == torch.device("cpu"):
            torch.set_num_threads(num_threads)
    elif image.device == torch.device("cpu"):
        # determine number of process forks while keeping a minimum amount of
        # k-space per fork
        for factor in factors.flip(0):
            if omega.shape[1] / (num_threads // factor) >= min_kspace_per_fork:
                threads_per_fork = int(factor)

        num_forks = num_threads // threads_per_fork

        if USING_OMP:
            torch.set_num_threads(threads_per_fork)
        kdat = table_interp_fork_over_kspace(
            image, omega, tables, n_shift, numpoints, table_oversamp, offsets, num_forks
        )
        if USING_OMP:
            torch.set_num_threads(num_threads)
    else:
        # no forking for batchless omega on GPU
        kdat = table_interp_one_batch(
            image, omega, tables, n_shift, numpoints, table_oversamp, offsets
        )

    return kdat


@torch.jit.script
def accum_tensor_index_add(
    image: Tensor, arr_ind: Tensor, data: Tensor, batched_nufft: bool
) -> Tensor:
    """We fork this function for the adjoint accumulation."""
    if batched_nufft:
        for (image_batch, arr_ind_batch, data_batch) in zip(image, arr_ind, data):
            for (image_coil, data_coil) in zip(image_batch, data_batch):
                image_coil.index_add_(0, arr_ind_batch, data_coil)
    else:
        for (image_it, data_it) in zip(image, data):
            image_it.to(torch.float32).index_add_(0, arr_ind, data_it)

    return image


@torch.jit.script
def fork_and_accum(
    image: Tensor, arr_ind: Tensor, data: Tensor, num_forks: int, batched_nufft: bool
) -> Tensor:
    """Process forking and per batch/coil accumulation function."""
    # initialize the fork processes
    futures: List[torch.jit.Future[torch.Tensor]] = []
    if batched_nufft:
        for (image_chunk, arr_ind_chunk, data_chunk) in zip(
            image.split(num_forks),
            arr_ind.split(num_forks),
            data.split(num_forks),
        ):
            futures.append(
                torch.jit.fork(
                    accum_tensor_index_add,
                    image_chunk,
                    arr_ind_chunk,
                    data_chunk,
                    batched_nufft,
                )
            )
    else:
        for (image_chunk, data_chunk) in zip(
            image.split(num_forks), data.split(num_forks)
        ):
            futures.append(
                torch.jit.fork(
                    accum_tensor_index_add,
                    image_chunk,
                    arr_ind,
                    data_chunk,
                    batched_nufft,
                )
            )

    # wait for processes to finish
    # results in-place
    _ = [torch.jit.wait(future) for future in futures]

    return image


@torch.jit.script
def calc_coef_and_indices_batch(
    tm: Tensor,
    base_offset: Tensor,
    offset_increments: Tensor,
    tables: List[Tensor],
    centers: Tensor,
    table_oversamp: Tensor,
    grid_size: Tensor,
    conjcoef: bool,
) -> Tuple[Tensor, Tensor]:
    """For loop coef calculation over batch dim."""
    coef = []
    arr_ind = []
    for (tm_it, base_offset_it) in zip(tm, base_offset):
        coef_it, arr_ind_it = calc_coef_and_indices(
            tm_it,
            base_offset_it,
            offset_increments,
            tables,
            centers,
            table_oversamp,
            grid_size,
            conjcoef,
        )

        coef.append(coef_it)
        arr_ind.append(arr_ind_it)

    return (torch.stack(coef), torch.stack(arr_ind))


@torch.jit.script
def calc_coef_and_indices_fork_over_batches(
    tm: Tensor,
    base_offset: Tensor,
    offset_increments: Tensor,
    tables: List[Tensor],
    centers: Tensor,
    table_oversamp: Tensor,
    grid_size: Tensor,
    conjcoef: bool,
    num_forks: int,
    batched_nufft: bool,
) -> Tuple[Tensor, Tensor]:
    """Split work across batchdim, fork processes."""
    if batched_nufft:
        # initialize the fork processes
        futures: List[torch.jit.Future[Tuple[Tensor, Tensor]]] = []
        for (tm_chunk, base_offset_chunk) in zip(
            tm.split(num_forks),
            base_offset.split(num_forks),
        ):
            futures.append(
                torch.jit.fork(
                    calc_coef_and_indices_batch,
                    tm_chunk,
                    base_offset_chunk,
                    offset_increments,
                    tables,
                    centers,
                    table_oversamp,
                    grid_size,
                    conjcoef,
                )
            )

        # collect the results
        results = [torch.jit.wait(future) for future in futures]
        coef = torch.cat([result[0] for result in results])
        arr_ind = torch.cat([result[1] for result in results])
    else:
        coef, arr_ind = calc_coef_and_indices(
            tm,
            base_offset,
            offset_increments,
            tables,
            centers,
            table_oversamp,
            grid_size,
            conjcoef,
        )

    return coef, arr_ind


@torch.jit.script
def sort_one_batch(
    tm: Tensor, omega: Tensor, data: Tensor, grid_size: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sort input tensors by ordered values of tm."""
    tmp = torch.zeros(omega.shape[1], dtype=omega.dtype, device=omega.device)
    for d, dim in enumerate(grid_size):
        tmp = tmp + torch.remainder(tm[d], dim) * torch.prod(grid_size[d + 1 :])

    _, indices = torch.sort(tmp)

    return tm[:, indices], omega[:, indices], data[:, :, indices]


@torch.jit.script
def sort_data(
    tm: Tensor, omega: Tensor, data: Tensor, grid_size: Tensor, batched_nufft: bool
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sort input tensors by ordered values of tm."""
    if batched_nufft:
        # loop over batch dimension to get sorted k-space
        results: List[Tuple[Tensor, Tensor, Tensor]] = []
        for (tm_it, omega_it, data_it) in zip(tm, omega, data):
            results.append(
                sort_one_batch(tm_it, omega_it, data_it.unsqueeze(0), grid_size)
            )

        tm_ret = torch.stack([result[0] for result in results])
        omega_ret = torch.stack([result[1] for result in results])
        data_ret = torch.cat([result[2] for result in results])
    else:
        tm_ret, omega_ret, data_ret = sort_one_batch(tm, omega, data, grid_size)

    return tm_ret, omega_ret, data_ret


def table_interp_adjoint(
    data: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    grid_size: Tensor,
) -> Tensor:
    """Table interpolation adjoint backend.
    This interpolates from an off-grid set of data at coordinates given by
    ``omega`` to on-grid locations.
    Args:
        data: Off-grid data to interpolate from.
        omega: Fourier coordinates to interpolate to (in radians/voxel, -pi to
            pi).
        tables: List of tables for each image dimension.
        n_shift: Size of desired fftshift.
        numpoints: Number of neighbors in each dimension.
        table_oversamp: Size of table in each dimension.
        offsets: A list of offset values for interpolation.
        grid_size: Size of grid to interpolate to.
    Returns:
        ``data`` interpolated to gridded locations.
    """
    dtype = data.dtype
    device = data.device
    int_type = torch.long
    batched_nufft = False

    if omega.ndim not in (2, 3):
        raise ValueError("omega must have 2 or 3 dimensions.")

    if omega.ndim == 3:
        if omega.shape[0] == 1:
            omega = omega[0]  # broadcast a single traj

    if omega.ndim == 3:
        batched_nufft = True
        if not omega.shape[0] == data.shape[0]:
            raise ValueError(
                "If omega has batch dim, omega batch dimension must match data."
            )

    # we fork processes for accumulation, so we need to do a bit of thread
    # management for OMP to make sure we don't oversubscribe (managment not
    # necessary for non-OMP)
    num_threads = torch.get_num_threads()
    factors = torch.arange(1, num_threads + 1)
    factors = factors[torch.remainder(torch.tensor(num_threads), factors) == 0]
    threads_per_fork = num_threads  # default fallback

    if batched_nufft:
        # increase number of forks as long as it's not greater than batch size
        for factor in factors.flip(0):
            if num_threads // factor <= omega.shape[0]:
                threads_per_fork = int(factor)
    else:
        # increase forks as long as it's less/eq than batch * coildim
        for factor in factors.flip(0):
            if num_threads // factor <= data.shape[0] * data.shape[1]:
                threads_per_fork = int(factor)

    num_forks = num_threads // threads_per_fork

    # calculate output size
    output_prod = int(torch.prod(grid_size))
    output_size = [data.shape[0], data.shape[1]]
    for el in grid_size:
        output_size.append(int(el))

    # convert to normalized freq locs and sort
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))
    tm, omega, data = sort_data(tm, omega, data, grid_size, batched_nufft)

    # compute interpolation centers
    centers = torch.floor_divide(numpoints * table_oversamp,2).to(dtype=int_type)

    # offset from k-space to first coef loc
    base_offset = 1 + torch.floor_divide(tm - numpoints.unsqueeze(-1),2.0).to(dtype=int_type)

    # initialized flattened image
    image = torch.zeros(
        size=(data.shape[0], data.shape[1], output_prod),
        dtype=dtype,
        device=device,
    )

    # phase for fftshift
    data = (
        data
        * imag_exp(
            torch.sum(omega * n_shift.unsqueeze(-1), dim=-2, keepdim=True),
            return_complex=True,
        ).conj()
    )

    # loop over offsets
    for offset in offsets:
        # TODO: see if we can fix thread counts in forking
        coef, arr_ind = calc_coef_and_indices_fork_over_batches(
            tm,
            base_offset,
            offset,
            tables,
            centers,
            table_oversamp,
            grid_size,
            True,
            num_forks,
            batched_nufft,
        )

        # multiply coefs to data
        if coef.ndim == 2:
            coef = coef.unsqueeze(1)
            assert coef.ndim == data.ndim

        # this is a much faster way of doing index accumulation
        if batched_nufft:
            # fork just over batches
            image = fork_and_accum(
                image, arr_ind, coef * data, num_forks, batched_nufft
            )
        else:
            # fork over coils and batches
            image = image.view(data.shape[0] * data.shape[1], output_prod)
            image = fork_and_accum(
                image,
                arr_ind,
                (coef * data).view(data.shape[0] * data.shape[1], -1),
                num_forks,
                batched_nufft,
            ).view(data.shape[0], data.shape[1], output_prod)

    return image.view(output_size)

class KbSpmatInterpForward(Function):
    @staticmethod
    def forward(ctx, image, interp_mats):
        """Apply sparse matrix interpolation.
        This is a wrapper for for PyTorch autograd.
        """
        grid_size = torch.tensor(image.shape[2:], device=image.device)
        output = spmat_interp(image, interp_mats)

        if isinstance(interp_mats, tuple):
            ctx.save_for_backward(interp_mats[0], interp_mats[1], grid_size)
        else:
            ctx.save_for_backward(interp_mats, grid_size)

        return output

    @staticmethod
    def backward(ctx, data):
        """Apply sparse matrix interpolation adjoint for gradient calculation.
        This is a wrapper for for PyTorch autograd.
        """
        if len(ctx.saved_tensors) == 3:
            interp_mats = ctx.saved_tensors[:2]
            grid_size = ctx.saved_tensors[2]
        else:
            (interp_mats, grid_size) = ctx.saved_tensors

        x = spmat_interp_adjoint(data, interp_mats, grid_size)

        return x, None


class KbSpmatInterpAdjoint(Function):
    @staticmethod
    def forward(ctx, data, interp_mats, grid_size):
        """Apply sparse matrix interpolation adjoint.
        This is a wrapper for for PyTorch autograd.
        """
        image = spmat_interp_adjoint(data, interp_mats, grid_size)

        if isinstance(interp_mats, tuple):
            ctx.save_for_backward(interp_mats[0], interp_mats[1])
        else:
            ctx.save_for_backward(interp_mats)

        return image

    @staticmethod
    def backward(ctx, image):
        """Apply sparse matrix interpolation for gradient calculation.
        This is a wrapper for for PyTorch autograd.
        """
        if len(ctx.saved_tensors) == 2:
            interp_mats = ctx.saved_tensors
        else:
            (interp_mats,) = ctx.saved_tensors

        y = spmat_interp(image, interp_mats)

        return y, None, None


class KbTableInterpForward(Function):
    @staticmethod
    def forward(ctx, image, omega, tables, n_shift, numpoints, table_oversamp, offsets):
        """Apply table interpolation.
        This is a wrapper for for PyTorch autograd.
        """
        grid_size = torch.tensor(image.shape[2:], device=image.device)

        output = table_interp(
            image=image,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
        )

        ctx.save_for_backward(
            omega, n_shift, numpoints, table_oversamp, offsets, grid_size, *tables
        )

        return output

    @staticmethod
    def backward(ctx, data):
        """Apply table interpolation adjoint for gradient calculation.
        This is a wrapper for for PyTorch autograd.
        """
        (
            omega,
            n_shift,
            numpoints,
            table_oversamp,
            offsets,
            grid_size,
        ) = ctx.saved_tensors[:6]
        tables = [table for table in ctx.saved_tensors[6:]]

        image = table_interp_adjoint(
            data=data,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
            grid_size=grid_size,
        )

        return image, None, None, None, None, None, None


class KbTableInterpAdjoint(Function):
    @staticmethod
    def forward(
        ctx, data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
    ):
        """Apply table interpolation adjoint.
        This is a wrapper for for PyTorch autograd.
        """
        image = table_interp_adjoint(
            data=data,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
            grid_size=grid_size,
        )

        ctx.save_for_backward(
            omega, n_shift, numpoints, table_oversamp, offsets, *tables
        )

        return image

    @staticmethod
    def backward(ctx, image):
        """Apply table interpolation for gradient calculation.
        This is a wrapper for for PyTorch autograd.
        """
        (omega, n_shift, numpoints, table_oversamp, offsets) = ctx.saved_tensors[:5]
        tables = [table for table in ctx.saved_tensors[5:]]

        data = table_interp(
            image=image,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
        )

        return data, None, None, None, None, None, None, None

def kb_spmat_interp(image: Tensor, interp_mats: Tuple[Tensor, Tensor]) -> Tensor:
    """Kaiser-Bessel sparse matrix interpolation.
    See :py:class:`~torchkbnufft.KbInterp` for an overall description of
    interpolation.
    To calculate the sparse matrix tuple, see
    :py:meth:`~torchkbnufft.calc_tensor_spmatrix`.
    Args:
        image: Gridded data to be interpolated to scattered data.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.
    Returns:
        ``image`` calculated at scattered locations.
    """
    is_complex = True
    if not image.is_complex():
        if not image.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        image = torch.view_as_complex(image)

    data = KbSpmatInterpForward.apply(image, interp_mats)

    if is_complex is False:
        data = torch.view_as_real(data)

    return data


def kb_spmat_interp_adjoint(
    data: Tensor, interp_mats: Tuple[Tensor, Tensor], grid_size: Tensor
) -> Tensor:
    """Kaiser-Bessel sparse matrix interpolation adjoint.
    See :py:class:`~torchkbnufft.KbInterpAdjoint` for an overall description of
    adjoint interpolation.
    To calculate the sparse matrix tuple, see
    :py:meth:`~torchkbnufft.calc_tensor_spmatrix`.
    Args:
        data: Scattered data to be interpolated to gridded data.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.
    Returns:
        ``data`` calculated at gridded locations.
    """
    is_complex = True
    if not data.is_complex():
        if not data.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        data = torch.view_as_complex(data)

    image = KbSpmatInterpAdjoint.apply(data, interp_mats, grid_size)

    if is_complex is False:
        image = torch.view_as_real(image)

    return image


def kb_table_interp(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Kaiser-Bessel table interpolation.
    See :py:class:`~torchkbnufft.KbInterp` for an overall description of
    interpolation and how to construct the function arguments.
    Args:
        image: Gridded data to be interpolated to scattered data.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift: Size for fftshift, usually ``im_size // 2``.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            ``numpoints``.
    Returns:
        ``image`` calculated at scattered locations.
    """
    is_complex = True
    if not image.is_complex():
        if not image.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        image = torch.view_as_complex(image)

    data = KbTableInterpForward.apply(
        image, omega, tables, n_shift, numpoints, table_oversamp, offsets
    )

    if is_complex is False:
        data = torch.view_as_real(data)

    return data


def kb_table_interp_adjoint(
    data: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    grid_size: Tensor,
) -> Tensor:
    """Kaiser-Bessel table interpolation adjoint.
    See :py:class:`~torchkbnufft.KbInterpAdjoint` for an overall description of
    adjoint interpolation.
    Args:
        data: Scattered data to be interpolated to gridded data.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift: Size for fftshift, usually ``im_size // 2``.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            ``numpoints``.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``.
    Returns:
        ``data`` calculated at gridded locations.
    """
    is_complex = True
    if not data.is_complex():
        if not data.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        data = torch.view_as_complex(data)

    image = KbTableInterpAdjoint.apply(
        data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
    )

    if is_complex is False:
        image = torch.view_as_real(image)

    return image
def build_numpy_spmatrix(
    omega: np.ndarray,
    numpoints: Sequence[int],
    im_size: Sequence[int],
    grid_size: Sequence[int],
    n_shift: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> coo_matrix:
    """Builds a sparse matrix with the interpolation coefficients.
    Args:
        omega: An array of coordinates to interpolate to (radians/voxel).
        numpoints: Number of points to use for interpolation in each dimension.
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        n_shift: Number of points to shift for fftshifts.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.
    Returns:
        A scipy sparse interpolation matrix.
    """
    spmat = -1

    ndims = omega.shape[0]
    klength = omega.shape[1]

    # calculate interpolation coefficients using kb kernel
    def interp_coeff(om, npts, grdsz, alpha, order):
        gam = 2 * np.pi / grdsz
        interp_dist = om / gam - np.floor(om / gam - npts / 2)
        Jvec = np.reshape(np.array(range(1, npts + 1)), (1, npts))
        kern_in = -1 * Jvec + np.expand_dims(interp_dist, 1)

        cur_coeff = np.zeros(shape=kern_in.shape, dtype=np.complex)
        indices = np.absolute(kern_in) < npts / 2
        bess_arg = np.sqrt(1 - (kern_in[indices] / (npts / 2)) ** 2)
        denom = special.iv(order, alpha)
        cur_coeff[indices] = special.iv(order, alpha * bess_arg) / denom
        cur_coeff = np.real(cur_coeff)

        return cur_coeff, kern_in

    full_coef = []
    kd = []
    for (
        it_om,
        it_im_size,
        it_grid_size,
        it_numpoints,
        it_om,
        it_alpha,
        it_order,
    ) in zip(omega, im_size, grid_size, numpoints, omega, alpha, order):
        # get the interpolation coefficients
        coef, kern_in = interp_coeff(
            it_om, it_numpoints, it_grid_size, it_alpha, it_order
        )

        gam = 2 * np.pi / it_grid_size
        phase_scale = 1j * gam * (it_im_size - 1) / 2

        phase = np.exp(phase_scale * kern_in)
        full_coef.append(phase * coef)

        # nufft_offset
        koff = np.expand_dims(np.floor(it_om / gam - it_numpoints / 2), 1)
        Jvec = np.reshape(np.arange(1, it_numpoints + 1), (1, it_numpoints))
        kd.append(np.mod(Jvec + koff, it_grid_size) + 1)

    for i in range(len(kd)):
        kd[i] = (kd[i] - 1) * np.prod(grid_size[i + 1 :])

    # build the sparse matrix
    kk = kd[0]
    spmat_coef = full_coef[0]
    for i in range(1, ndims):
        Jprod = int(np.prod(numpoints[: i + 1]))
        # block outer sum
        kk = np.reshape(
            np.expand_dims(kk, 1) + np.expand_dims(kd[i], 2), (klength, Jprod)
        )
        # block outer prod
        spmat_coef = np.reshape(
            np.expand_dims(spmat_coef, 1) * np.expand_dims(full_coef[i], 2),
            (klength, Jprod),
        )

    # build in fftshift
    phase = np.exp(1j * np.dot(np.transpose(omega), np.expand_dims(n_shift, 1)))
    spmat_coef = np.conj(spmat_coef) * phase

    # get coordinates in sparse matrix
    trajind = np.expand_dims(np.arange(klength), 1)
    trajind = np.repeat(trajind, int(np.prod(numpoints)), axis=1)

    # build the sparse matrix
    spmat = coo_matrix(
        (spmat_coef.flatten(), (trajind.flatten(), kk.flatten())),
        shape=(klength, np.prod(grid_size)),
    )

    return spmat


def build_table(
    im_size: Sequence[int],
    grid_size: Sequence[int],
    numpoints: Sequence[int],
    table_oversamp: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> List[Tensor]:
    """Builds an interpolation table.
    Args:
        numpoints: Number of points to use for interpolation in each dimension.
        table_oversamp: Table oversampling factor.
        grid_size: Size of the grid to interpolate from.
        im_size: Size of base image.
        ndims: Number of image dimensions.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.
    Returns:
        A list of tables for each dimension.
    """
    table = []

    # build one table for each dimension
    for (
        it_im_size,
        it_grid_size,
        it_numpoints,
        it_table_oversamp,
        it_order,
        it_alpha,
    ) in zip(im_size, grid_size, numpoints, table_oversamp, order, alpha):
        # The following is a trick of Fessler.
        # It uses broadcasting semantics to quickly build the table.
        t1 = (
            it_numpoints / 2
            - 1
            + np.array(range(it_table_oversamp)) / it_table_oversamp
        )  # [L]
        om1 = t1 * 2 * np.pi / it_grid_size  # gam
        s1 = build_numpy_spmatrix(
            np.expand_dims(om1, 0),
            numpoints=(it_numpoints,),
            im_size=(it_im_size,),
            grid_size=(it_grid_size,),
            n_shift=(0,),
            order=(it_order,),
            alpha=(it_alpha,),
        )
        h = np.array(s1.getcol(it_numpoints - 1).todense())
        for col in range(it_numpoints - 2, -1, -1):
            h = np.concatenate((h, np.array(s1.getcol(col).todense())), axis=0)
        h = np.concatenate((h.flatten(), np.array([0])))

        table.append(torch.tensor(h))

    return table


def kaiser_bessel_ft(
    omega: np.ndarray, numpoints: int, alpha: float, order: float, d: int
) -> np.ndarray:
    """Computes FT of KB function for scaling in image domain.
    Args:
        omega: An array of coordinates to interpolate to.
        numpoints: Number of points to use for interpolation in each dimension.
        alpha: KB parameter.
        order: Order of Kaiser-Bessel kernel.
        d (int):  # TODO: find what d is
    Returns:
        The scaling coefficients.
    """
    z = np.sqrt((2 * np.pi * (numpoints / 2) * omega) ** 2 - alpha**2 + 0j)
    nu = d / 2 + order
    scaling_coef = (
        (2 * np.pi) ** (d / 2)
        * ((numpoints / 2) ** d)
        * (alpha**order)
        / special.iv(order, alpha)
        * special.jv(nu, z)
        / (z**nu)
    )
    scaling_coef = np.real(scaling_coef)

    return scaling_coef


def compute_scaling_coefs(
    im_size: Sequence[int],
    grid_size: Sequence[int],
    numpoints: Sequence[int],
    alpha: Sequence[float],
    order: Sequence[float],
) -> Tensor:
    """Computes scaling coefficients for NUFFT operation.
    Args:
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        numpoints: Number of points to use for interpolation in each dimension.
        alpha: KB parameter.
        order: Order of Kaiser-Bessel kernel.
    Returns:
        The scaling coefficients.
    """
    num_coefs = np.array(range(im_size[0])) - (im_size[0] - 1) / 2
    scaling_coef = 1 / kaiser_bessel_ft(
        num_coefs / grid_size[0], numpoints[0], alpha[0], order[0], 1
    )

    if numpoints[0] == 1:
        scaling_coef = np.ones(scaling_coef.shape)

    for i in range(1, len(im_size)):
        indlist = np.array(range(im_size[i])) - (im_size[i] - 1) / 2
        scaling_coef = np.expand_dims(scaling_coef, axis=-1)
        tmp = 1 / kaiser_bessel_ft(
            indlist / grid_size[i], numpoints[i], alpha[i], order[i], 1
        )

        for _ in range(i):
            tmp = tmp[np.newaxis]

        if numpoints[i] == 1:
            tmp = np.ones(tmp.shape)

        scaling_coef = scaling_coef * tmp

    return torch.tensor(scaling_coef)


def init_fn(
    im_size: Sequence[int],
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2**10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tuple[
    List[Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    """Initialization function for NUFFT objects.
    Args:
        im_size: Size of image.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.
        numpoints: Number of neighbors to use for interpolation.
        n_shift; Optional: Size for fftshift, usually `im_size // 2`.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers.
        device: Which device to create tensors on.
    Returns:
        Tuple containing all variables recast as Tensors:
            tables (List of tensors)
            im_size
            grid_size
            n_shift
            numpoints
            offset_list
            table_oversamp
            order
            alpha
    """
    (
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        order,
        alpha,
        dtype,
        device,
    ) = validate_args(
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        kbwidth,
        order,
        dtype,
        device,
    )

    tables = build_table(
        numpoints=numpoints,
        table_oversamp=table_oversamp,
        grid_size=grid_size,
        im_size=im_size,
        order=order,
        alpha=alpha,
    )
    assert len(tables) == len(im_size)

    # precompute interpolation offsets
    offset_list = list(itertools.product(*[range(numpoint) for numpoint in numpoints]))

    if dtype.is_floating_point:
        real_dtype = dtype
        for pair in DTYPE_MAP:
            if pair[1] == real_dtype:
                complex_dtype = pair[0]
                break
    elif dtype.is_complex:
        complex_dtype = dtype
        for pair in DTYPE_MAP:
            if pair[0] == complex_dtype:
                real_dtype = pair[1]
                break
    else:
        raise TypeError("Unrecognized dtype.")

    tables = [table.to(dtype=complex_dtype, device=device) for table in tables]

    return (
        tables,
        torch.tensor(im_size, dtype=torch.long, device=device),
        torch.tensor(grid_size, dtype=torch.long, device=device),
        torch.tensor(n_shift, dtype=real_dtype, device=device),
        torch.tensor(numpoints, dtype=torch.long, device=device),
        torch.tensor(offset_list, dtype=torch.long, device=device),
        torch.tensor(table_oversamp, dtype=torch.long, device=device),
        torch.tensor(order, dtype=real_dtype, device=device),
        torch.tensor(alpha, dtype=real_dtype, device=device),
    )


def validate_args(
    im_size: Sequence[int],
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2**10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tuple[
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[float],
    Sequence[float],
    torch.dtype,
    torch.device,
]:
    im_size = tuple(im_size)
    if grid_size is None:
        grid_size = tuple([dim * 2 for dim in im_size])
    else:
        grid_size = tuple(grid_size)
    if isinstance(numpoints, int):
        numpoints = tuple([numpoints for _ in range(len(grid_size))])
    else:
        numpoints = tuple(numpoints)
    if n_shift is None:
        n_shift = tuple([dim // 2 for dim in im_size])
    else:
        n_shift = tuple(n_shift)
    if isinstance(table_oversamp, int):
        table_oversamp = tuple(table_oversamp for _ in range(len(grid_size)))
    else:
        table_oversamp = tuple(table_oversamp)
    alpha = tuple(kbwidth * numpoint for numpoint in numpoints)
    if isinstance(order, float):
        order = tuple(order for _ in range(len(grid_size)))
    else:
        order = tuple(order)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")

    # dimension checking
    assert len(grid_size) == len(im_size)
    assert len(n_shift) == len(im_size)
    assert len(numpoints) == len(im_size)
    assert len(alpha) == len(im_size)
    assert len(order) == len(im_size)
    assert len(table_oversamp) == len(im_size)

    return (
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        order,
        alpha,
        dtype,
        device,
    )