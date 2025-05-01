import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Shaped


def coerce_to_3d(x: Array) -> Shaped[Array, "?batch dim1 dim2"]:
    n_dim = x.ndim
    if n_dim == 2:
        x = jnp.expand_dims(x, axis=0)
    elif n_dim < 2 or n_dim > 3:
        raise ValueError("The rank of x must be 2 or 3. Instead got rank {n_dim}")
    return x


def standardize(
    array: Float[Array, "?batch time dim"],
    other: Float[Array, "?batch_other time dim"] | None = None,
    axis: int = -2,
) -> tuple[
    Float[Array, "?batch time dim"], Float[Array, "?batch_other time dim"] | None
]:
    """
    Standardizes a given 2D or 3D array along some axis.
    For 2D arrays, axis specifies the axis over which the mean and standard deviations
    are to be calculated.
    For 3D arrays, the mean and standard deviation reductions are additionally performed
    over the leading axis, which is assumed to correspond to the batch dimension.

    The argument `other` is used to pass in an additional array, which is standardized
    using the statistics of the first argument `array`.

    This is useful when preprocessing train and validation/test sets for model training,
    as using the combined statistics for both train and validation/test sets for
    standardization leads to data leakage.
    """
    orig_shape = array.shape
    array = coerce_to_3d(array)
    mean = jnp.mean(array, axis=(0, axis), keepdims=True)
    std = jnp.std(array, axis=(0, axis), keepdims=True)

    arr_stdized = jnp.reshape((array - mean) / std, orig_shape)
    if other is None:
        return arr_stdized
    else:
        othr_stdized = jnp.reshape((other - mean) / std, other.shape)
        return arr_stdized, othr_stdized


def add_noise(
    array: Float[Array, "*rest time dim"],
    rel_noise_strength: float = 0.05,
    preserve_first: bool = False,
    key: Array | int = 0,
) -> Float[Array, "*rest time dim"]:
    """
    Adds Gaussian noise to given array per data dimension, which is assumed to be
    along the last axis.

    The noise strength is set to be a multiple of the array standard deviation,
    calculated over the first two axes.

    preserve_first determines whether to keep the first timepoint of the array
    (array[:,0]) un-noised or not.
    This can be useful when experimenting with autoregressive models whose outputs are
    sensitive to the initial condition used.
    """
    if rel_noise_strength < 0:
        raise ValueError("Relative noise strength must be a nonnegative number.")
    elif rel_noise_strength == 0:
        array_noised = array
    else:
        std: Float[Array, " dim"] = jnp.std(array, axis=tuple(range(array.ndim - 1)))

        key = jax.random.PRNGKey(key)
        noise = jax.random.normal(key, array.shape) * std * rel_noise_strength
        array_noised = array + noise

        if preserve_first:
            array_noised = array_noised.at[:, 0].set(array[:, 0])

    return array_noised


def downsample(
    data: Float[Array, "*rest time dim"], keep_every: int = 1
) -> Float[Array, "*rest time dim//keep_every"]:
    """
    Downsample the given data array by an integer factor along the time dimension,
    which is assumed to be along the second-to-last axis.
    """
    if keep_every <= 0:
        raise ValueError("Downsampling factor must be an integer larger than 1.")
    elif keep_every == 1:
        return data
    else:
        return data[..., ::keep_every, :]


def split_into_chunks(
    sequence: Float[Array, " time ?dim"], chunk_size: int, overlap: int = 0
) -> Float[Array, "batch chunk_size ?dim"]:
    """
    Split an array into possibly overlapping chunks.

    The number of resulting chunks is derived from the following inequality:
    (n-1)*(chunk_size-overlap)+chunk_size <= len(sequence)
                                            < n*(chunk_size-overlap)+chunk_size

    overlap is an integer ranging from (-len(sequence)+1, len(sequence)-1).

    If the given chunk_size, overlap parameters do not cleanly divide the length
    of the sequence array, the remaining end bits are discarded.
    """
    if overlap < 0:
        overlap = chunk_size + overlap
    assert 0 <= overlap < chunk_size, "Overlap must be smaller than the chunk_size"

    num_chunks = (len(sequence) - chunk_size) // (chunk_size - overlap) + 1

    def slice_chunk(start_ind: int, arg=None):
        del arg

        return start_ind + chunk_size - overlap, jax.lax.dynamic_slice_in_dim(
            sequence, start_ind, chunk_size, axis=0
        )

    _, chunks = jax.lax.scan(slice_chunk, 0, length=num_chunks)
    return chunks
