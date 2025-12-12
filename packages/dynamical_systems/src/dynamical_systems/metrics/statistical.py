import jax.numpy as jnp
from jax.scipy.signal import correlate
from jaxtyping import Array, Float


def gaussian_kernel_1d(sigma: float, radius: int) -> Float[Array, " 2*{radius}"]:
    """JAX version of the gaussian kernel to be used in the gaussian_filter1d function.

    Note that the implementation parallels that of scipy.ndimage at
    https://github.com/scipy/scipy/blob/v1.16.2/scipy/ndimage/_filters.py#L656."""
    x = jnp.arange(-radius, radius + 1)
    sigma2 = sigma * sigma
    phi_x = jnp.exp(-0.5 * x**2 / sigma2)
    return phi_x / jnp.sum(phi_x)


def gaussian_filter1d(
    x: Float[Array, " N"], sigma: float, truncate: float = 4.0
) -> Float[Array, " N"]:
    """JAX implementation of scipy.ndimage.gaussian_filter1d for order=1.

    Since jax.scipy.signal.correlate does not provide reflect padding, this is
    manually done via jax.numpy.pad."""
    radius = int(truncate * sigma + 0.5)
    weights = gaussian_kernel_1d(sigma, radius)[::-1]
    x_padded = jnp.pad(x, (radius, radius), mode="reflect")

    return correlate(x_padded, weights, mode="valid", method="fft")


def power_spectrum1d(
    x: Float[Array, " N"], standardize: bool = True, smoothing: int | None = 20
) -> Float[Array, " N"]:
    if standardize:
        x = x - jnp.mean(x) / jnp.std(x)
    x_fft = jnp.fft.rfft(x)
    ps = 2 * jnp.abs(x_fft) ** 2 / len(x)
    if smoothing is not None:
        ps = gaussian_filter1d(ps, smoothing)
    return ps / jnp.sum(ps)


def hellinger_distance(
    p: Float[Array, " N"], q: Float[Array, " N"]
) -> Float[Array, ""]:
    return jnp.sqrt(1 - jnp.sum(jnp.sqrt(p * q)))


def mean_spectral_hellinger_distance(
    x1: Float[Array, " time"],
    x2: Float[Array, " time"],
    standardize: bool = True,
    smoothing: int | None = 20,
) -> Float[Array, ""]:
    """This is a pure JAX implementation of the mean spectral hellinger distance.

    To use the function on data with multiple dimensions, simply vmap this function
    over the dimension axis and average the result.

    The implementation is heavily inspired from that of Durstewitz's group (see [1],
    [2]) where the power spectrum is smooothed before being used to calculate the
    hellinger distance.

    This is different from what is done is the `dysts` library [3], which uses
    `scipy.signal.welch` to estimate the power spectrum of the time series.

    We choose the former approch as it has less algorithm parameters, and also is easier
    to port to JAX.

    [1] https://github.com/DurstewitzLab/DynaMix-python/blob/main/src/dynamix/metrics/metrics.py#L83
    [2] https://github.com/DurstewitzLab/ChaosRNN/blob/main/evaluation/pse.py#L65
    [3] https://github.com/GilpinLab/dysts/blob/master/dysts/metrics.py#L922
    """
    ps1 = power_spectrum1d(x1, standardize=standardize, smoothing=smoothing)
    ps2 = power_spectrum1d(x2, standardize=standardize, smoothing=smoothing)
    return hellinger_distance(ps1, ps2)
