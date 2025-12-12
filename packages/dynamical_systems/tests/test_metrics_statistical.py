import jax
import numpy as np
import scipy.ndimage as scimage
from dynamical_systems.metrics.statistical import gaussian_filter1d
from scipy.signal import square


jax.config.update("jax_enable_x64", True)


def test_gaussian_filter1d():
    t = np.linspace(0, 1, 500, endpoint=False)
    waveform = square(2 * np.pi * 2 * t)
    waveform_scipy = scimage.gaussian_filter1d(waveform, 20)
    waveform_jax = gaussian_filter1d(waveform, 20)
    assert np.allclose(waveform_scipy, waveform_jax)
