from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy.signal import find_peaks, peak_prominences
from tqdm import tqdm

from dynamical_systems.metrics.statistical import (
    power_spectrum1d,
)


def average_power_spectrum(
    t: Float[Array, " time"],
    u_batch: Float[Array, "batch time dim"],
    smoothing: int = 5,
    batch_size=100,
) -> tuple[Float[Array, " freq"], Float[Array, "dim freq"]]:
    def _power_spectrum(u: Float[Array, "time dim"]) -> Float[Array, "dim freq"]:
        return jax.vmap(partial(power_spectrum1d, smoothing=smoothing), in_axes=1)(u)

    freqs = jnp.fft.rfftfreq(len(t), d=t[1] - t[0])
    ps = jax.lax.map(_power_spectrum, u_batch, batch_size=batch_size)
    mean_ps: Float[Array, "dim freq"] = jnp.mean(ps, axis=0)
    return freqs, mean_ps


def period_from_average_power_spectrum(
    t: Float[Array, " time"],
    u_batch: Float[Array, "batch time dim"],
    smoothing: int = 5,
    batch_size=100,
):
    freqs, mean_ps = average_power_spectrum(t, u_batch, smoothing, batch_size)
    mean_log_ps: Float[Array, "dim freqs"] = jnp.log10(mean_ps)

    periods = []
    for log_ps_i in tqdm(mean_log_ps):
        peak_locs = find_peaks(log_ps_i)[0]
        if len(peak_locs) >= 1:
            prominences = peak_prominences(log_ps_i, peak_locs)[0]
            periods.append(1 / freqs[peak_locs[np.argmax(prominences)]])
    return jnp.median(jnp.asarray(periods))
