from pathlib import Path

from jaxtyping import Array, Float, Int, PRNGKeyArray


IntScalar = Int[Array, ""]
IntScalarLike = int | IntScalar
FloatScalar = Float[Array, ""]
FloatScalarLike = float | FloatScalar

PRNGKeyArrayLike = IntScalarLike | PRNGKeyArray

PathLike = str | Path
