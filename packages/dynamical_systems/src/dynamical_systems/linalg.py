import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# adapted from https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
def gram_schmidt(
    vectors: Float[Array, "dim num_vector"],
) -> tuple[Float[Array, "dim num_vector"], Float[Array, " num_vector"]]:
    """Implementation of the modified Gram-Schmidt orthonormalization algorithm.

    The code is slightedly modified from the original version at
    https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
    By removing an unnecessary jnp.linalg.norm in the main loop, the code has been sped
    up by about 20%.

    Unlike the original version, the code also outputs the norm of the orthogonalized
    vectors, which is useful in computing the Lyapunov exponents.

    Below is the documentation from the original code:

    We assume here that the vectors are linearly independent. Zero vectors will be
    left unchanged, but will also consume an iteration against `num_vectors`.

    From [1]: "MGS is numerically equivalent to Householder QR factorization
    applied to the matrix A augmented with a square matrix of zero elements on
    top."

    Historical note, see [1]: "modified" Gram-Schmidt was derived by Laplace [2],
    for elimination and not as an orthogonalization algorithm. "Classical"
    Gram-Schmidt actually came later [2]. Classical Gram-Schmidt has a sometimes
    catastrophic loss of orthogonality for badly conditioned matrices, which is
    discussed further in [1].

    #### References

    [1] Bjorck, A. (1994). Numerics of gram-schmidt orthogonalization. Linear
        Algebra and Its Applications, 197, 297-316.

    [2] P. S. Laplace, Thiorie Analytique des Probabilites. Premier Supple'ment,
        Mme. Courtier, Paris, 1816.

    [3] E. Schmidt, über die Auflosung linearer Gleichungen mit unendlich vielen
        Unbekannten, Rend. Circ. Mat. Pulermo (1) 25:53-77 (1908).

    Args:
      vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
        orthonormalize.

    Returns:
      bases: A Tensor of shape `[d, n]` corresponding to the orthonormalization.
      norms: Norm of the basis vectors
    """
    n_vectors = vectors.shape[-1]

    def body_fn(i, vecs):
        # Slice out the vector w.r.t. which we're orthogonalizing the rest.
        u = vecs[:, i]  # (d, )
        # Find weights by dotting the d x 1 against the d x n.
        weights = u @ vecs  # (n,)
        # Project out vector `u` from the trailing vectors.
        masked_weights = jnp.where(jnp.arange(n_vectors) > i, weights, 0.0) / weights[i]
        vecs = vecs - jnp.outer(u, masked_weights)
        return vecs

    bases = jax.lax.fori_loop(0, n_vectors - 1, body_fn, vectors)
    norms = jnp.linalg.norm(bases, axis=0, keepdims=True)
    return bases / norms, norms[0]
