import jax
from jaxtyping import Array, Float


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
