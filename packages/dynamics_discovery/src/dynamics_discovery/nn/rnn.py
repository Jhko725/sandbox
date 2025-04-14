import math
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init, named_scope
from jaxtyping import Array, PRNGKeyArray


class AbstractRNNCell(Protocol):
    """
    Protocol used to typecheck for all equinox modules implementing a single step
    of a RNN variant.

    Protocol is used instead of abstract base class is used so that this can be used
    to match `equinox.nn.GRUCell` and `equinox.nn.LSTMCell`.

    However, we cannot make AbstractRNNCell also inherit from `equinox.Module`
    (see https://github.com/patrick-kidger/equinox/pull/401), though in reality,
    it will almost always be the case that the RNNCell is implemented as a
    subclass of `equinox.Module`.
    """

    def __call__(
        self, input: Array, hidden: Array, *, key: PRNGKeyArray | None = None
    ): ...


class RNNCell(eqx.Module, strict=True):
    """A single step of a 1 layer Elman RNN.

    Analogous to `torch.nn.RNN(num_layers=1, bidirectional=False,
    nonlinearity=activation)`
    One difference from the PyTorch implementation is a single bias vector is
    used instead of two.

    !!! example

        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            cell: RNNCell

            def __init__(self, **kwargs):
                self.cell = RNNCell(**kwargs)

            def __call__(self, xs):
                scan_fn = lambda state, input: (self.cell(input, state), None)
                init_state = jnp.zeros(self.cell.hidden_size)
                final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
                return final_state
        ```
    """

    weight_ih: Array
    weight_hh: Array
    bias: Array | None
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        activation: Callable = jax.nn.tanh,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `dtype`: The dtype to use for all weights and biases in this GRU cell.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
            whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey = jax.random.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        ihshape = (hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)
        if use_bias:
            self.bias = default_init(bkey, (hidden_size,), dtype, lim)
        else:
            self.bias = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.activation = activation

    @named_scope("nn.RNNCell")
    def __call__(self, input: Array, hidden: Array, *, key: PRNGKeyArray | None = None):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a JAX array of shape
            `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """
        bias = self.bias if self.use_bias else 0
        ih = self.weight_ih @ input
        hh = self.weight_hh @ hidden
        return self.activation(ih + hh + bias)
