"""Davidson algorithm.

Adapted from PySCF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _put
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, TypeVar, Union

    from mypy_extensions import DefaultArg
    from numpy import complexfloating, floating, integer
    from numpy.typing import NDArray

    T = TypeVar("T", floating, complexfloating)

    PreconditionerType = Callable[[NDArray[T], Union[NDArray[T], T]], NDArray[T]]
    PickType = Callable[
        [
            NDArray[T],
            NDArray[T],
            int,
            DefaultArg(Optional[NDArray[T]], "basis_vectors"),  # noqa: F821
        ],
        tuple[NDArray[floating], NDArray[T], NDArray[integer]],
    ]


def make_diagonal_preconditioner(
    diag: NDArray[floating], level_shift: float = 0.0, tol: float = 1e-8
) -> PreconditionerType[T]:
    """Generate the preconditioner function.

    Args:
        diag: The diagonal of the matrix.
        level_shift: The level shift to use.
        tol: The tolerance to use to avoid division by zero.

    Returns:
        The preconditioner function.

    Notes:
        The default preconditioner approximates the inverse of the matrix by its diagonal. If the
        basis generated by the preconditioner is not linearly dependent, a level shift can be
        applied to break the correlation between the matrix and its diagonal.
    """

    def preconditioner(dx: NDArray[T], e: Union[NDArray[T], T]) -> NDArray[T]:
        """Precondition the solver."""
        diag_shifted = diag - (e - level_shift)
        indices = np.ix_(np.abs(diag_shifted) < tol)
        _put(
            diag_shifted,
            indices,  # type: ignore
            np.full(indices[0].shape, tol),
        )
        return dx / diag_shifted

    return preconditioner


def pick_real_eigenvalues(
    w: NDArray[T],
    v: NDArray[T],
    nroots: int,
    basis_vectors: Optional[NDArray[T]] = None,
) -> tuple[NDArray[floating], NDArray[T], NDArray[integer]]:
    """Search for eigenvalues that are real or close to real.

    Args:
        w: The eigenvalues.
        v: The eigenvectors.
        nroots: The number of roots to find.
        basis_vectors: The basis vectors. Not used in this function.

    Returns:
        The eigenvalues, eigenvectors, and the indices of the chosen eigenvalues.
    """
    imaginary_tol = 1e-3
    abs_imag = np.abs(np.imag(w))
    max_imag_tol = max(imaginary_tol, np.sort(abs_imag)[min(w.size, nroots) - 1])
    real_idx = np.where(abs_imag < max_imag_tol)[0]
    return make_eigenvectors_real(w, v, real_idx, real_eigenvectors=True)


def make_eigenvectors_real(
    w: NDArray[T],
    v: NDArray[T],
    real_idx: NDArray[integer],
    real_eigenvectors: bool = True,
) -> tuple[NDArray[floating], NDArray[T], NDArray[integer]]:
    """Transform the eigenvectors to be real-valued.

    If a complex eigenvalue has a small imaginary part, both the real and imaginary parts of the
    eigenvector can be used as the real eigenvector.

    Args:
        w: The eigenvalues.
        v: The eigenvectors.
        real_idx: The indices of the real eigenvalues.
        real_eigenvectors: Whether to return real eigenvectors.

    Returns:
        The eigenvalues, eigenvectors, and the indices of the chosen eigenvalues.

    Notes:
        If `real_eigenvectors` is set to `True`, this function can only be used for real matrices
        and real eigenvectors. It discards the imaginary part of the eigenvectors and returns only
        the real part of the eigenvectors.
    """
    idx = real_idx[np.argsort(np.real(w[real_idx]))]
    w = w[idx]
    v = v[:, idx]

    if real_eigenvectors:
        degen = np.where(np.imag(w) != 0)[0]
        if degen.size > 0:
            indices = np.ix_(np.arange(v.shape[0]), degen[1::2])
            _put(v, indices, np.imag(v[indices]))
        v = np.real(v)

    return np.real(w), v, idx


def _fill_subspace_matrix(
    matrix_prev: NDArray[T],
    basis_vectors: NDArray[T],
    matrix_basis_vectors: NDArray[T],
    trial_vectors: NDArray[T],
    matrix_trial_vectors: NDArray[T],
) -> NDArray[T]:
    """Fill the subspace matrix for the Davidson algorithm.

    Args:
        matrix_prev: The previous subspace matrix.
        basis_vectors: The basis vectors.
        matrix_basis_vectors: The matrix-vector products of the basis vectors.
        trial_vectors: The trial vectors.
        matrix_trial_vectors: The matrix-vector products of the trial vectors.

    Returns:
        The filled subspace matrix.
    """
    row1 = matrix_basis_vectors.shape[0]
    row0 = row1 - matrix_trial_vectors.shape[0]
    num = row1 - row0
    itrial = np.arange(row0, row1)
    ibasis = np.arange(row0)

    matrix = _put(
        matrix_prev,
        np.ix_(itrial, itrial),
        util.einsum("ik,jk->ij", np.conj(trial_vectors), matrix_trial_vectors),
    )

    matrix = _put(
        matrix,
        np.ix_(ibasis, itrial),
        util.einsum("ik,jk->ij", np.conj(basis_vectors[:row0]), matrix_trial_vectors[:num]),
    )

    matrix = _put(
        matrix,
        np.ix_(itrial, ibasis),
        util.einsum("ik,jk->ij", np.conj(trial_vectors[:num]), matrix_basis_vectors[:row0]),
    )

    return matrix


def _sort_eigenvalues(
    w_prev: NDArray[floating],
    conv_prev: list[bool],
    v_prev: NDArray[T],
    v: NDArray[T],
) -> tuple[NDArray[floating], list[bool]]:
    """Sort the eigenvalues.

    Reorders the eigenvalues of the last iteration to make them comparable to the eigenvalues of
    the current iteration, as the eigenstates may be flipped during the Davidson iterations.

    Args:
        w_prev: The eigenvalues of the last iteration.
        conv_prev: The convergence flags of the last iteration.
        v_prev: The eigenvectors of the last iteration.
        v: The eigenvectors of the current iteration.

    Returns:
        The sorted eigenvalues and convergence flags.
    """
    head, nroots = v_prev.shape
    ovlp = np.abs(np.dot(np.transpose(np.conj(v[:head])), v_prev))
    idx = np.argmax(ovlp, axis=1)
    return w_prev[idx], [conv_prev[i] for i in idx]


def _orthonormalise_vectors(vectors: NDArray[T], lindep_tol: float = 1e-14) -> NDArray[T]:
    """Orthonormalise a list of vectors, returning only the linearly independent vectors.

    Args:
        vectors: The vectors to orthonormalise.
        lindep_tol: The tolerance for linear dependence.

    Returns:
        The orthonormalised vectors.
    """
    vectors_out: list[NDArray[T]] = []
    for i in range(vectors.shape[0]):
        vector = np.copy(vectors[i])
        for vector_out in vectors_out:
            vector -= vector_out * np.dot(np.conj(vector_out), vector)
        norm = np.real(np.linalg.norm(vector, ord=2))
        if norm**2 > lindep_tol:
            vectors_out.append(vector / norm)
    return np.stack(vectors_out)


def _project_vectors(
    vectors: NDArray[T],
    basis_vectors: NDArray[T],
) -> NDArray[T]:
    """Project out existing basis vectors from a list of vectors.

    Args:
        vectors: The vectors to project.
        basis_vectors: The basis vectors to project out.

    Returns:
        The projected vectors.
    """
    projection: NDArray[T] = util.einsum(
        "ik,lk,lj->ij", vectors, np.conj(basis_vectors), basis_vectors
    )
    return vectors - projection


def _normalise_vectors(vectors: NDArray[T], lindep_tol: float = 1e-14) -> NDArray[T]:
    """Normalize vectors and remove vectors with small norms.

    Args:
        vectors: The vectors to normalize.
        lindep_tol: The tolerance for linear dependence.

    Returns:
        The normalized vectors.
    """
    norms: NDArray[floating] = np.real(np.linalg.norm(vectors, axis=1, ord=2))
    mask = norms**2 > lindep_tol
    return vectors[mask] / norms[mask][:, None]


def _outer_product_to_subspace(vectors: NDArray[T], basis_vectors: NDArray[T]) -> NDArray[T]:
    """Find the outer product of a list of vectors with a list of basis vectors.

    Args:
        vectors: The vectors.
        basis_vectors: The basis vectors.

    Returns:
        The outer product of the vectors
    """
    vectors_out: NDArray[T] = util.einsum("xi,xj->ij", vectors, basis_vectors)
    return vectors_out


def davidson(
    matvec: Callable[[NDArray[T]], NDArray[T]],
    vectors: NDArray[T],
    diagonal: NDArray[floating],
    nroots: int = 1,
    max_iter: int = 50,
    max_space: int = 20,
    max_trials: int = 40,
    e_tol: float = 1e-12,
    r_tol: Optional[float] = None,
    lindep_tol: float = 1e-14,
    left: bool = False,
    pick: Optional[PickType[T]] = None,
    follow_state: bool = False,
    level_shift: float = 0.0,
    callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[list[bool], NDArray[floating], NDArray[T]]:
    """Davidson algorithm for finding eigenvalues and eigenvectors."""
    # Parse arguments
    loose_tol = r_tol if r_tol is not None else e_tol**0.5
    max_space += (nroots - 1) * 6
    preconditioner: PreconditionerType[T] = make_diagonal_preconditioner(diagonal)
    if pick is None:
        pick = pick_real_eigenvalues
    dim = diagonal.size

    def _empty_vector() -> NDArray[T]:
        """Return an empty vector."""
        return np.empty((0, dim), dtype=types[complex])

    # Initialise variables
    w: NDArray[floating] = np.empty(0, dtype=types[float])
    v: NDArray[T] = _empty_vector()
    conv: list[bool] = [False] * nroots
    w_prev = np.copy(w)
    v_prev = np.copy(v)
    conv_prev = conv.copy()
    matrix: NDArray[T] = np.empty((max_space + nroots, max_space + nroots), dtype=types[complex])
    restart = True
    basis_vectors: NDArray[T] = _empty_vector()
    matrix_basis_vectors: NDArray[T] = _empty_vector()
    space = 0

    for cycle in range(max_iter):
        # Get the trial vectors
        if restart:
            basis_vectors = _empty_vector()
            matrix_basis_vectors = _empty_vector()
            space = 0
            trial_vectors = _orthonormalise_vectors(vectors, lindep_tol=lindep_tol)
            del vectors
        elif trial_vectors.shape[0] > 1:
            trial_vectors = _orthonormalise_vectors(trial_vectors, lindep_tol=lindep_tol)
            trial_vectors = trial_vectors[:max_trials]

        # Find the matrix-vector products for the trial vectors
        matrix_trial_vectors = np.stack(
            [matvec(trial_vectors[i]) for i in range(trial_vectors.shape[0])]
        )

        # Add the trial vectors to the basis
        basis_vectors = np.concatenate((basis_vectors, trial_vectors))
        matrix_basis_vectors = np.concatenate((matrix_basis_vectors, matrix_trial_vectors))
        space += trial_vectors.shape[0]

        # Store the previous iteration results
        w_prev = w
        v_prev = v
        conv_prev = conv

        # Fill the matrix
        matrix = _fill_subspace_matrix(
            matrix, basis_vectors, matrix_basis_vectors, trial_vectors, matrix_trial_vectors
        )
        del trial_vectors, matrix_trial_vectors

        # Solve the eigenvalue problem
        w, v, idx = pick(
            *np.linalg.eig(matrix[:space, :space]), nroots, basis_vectors=basis_vectors
        )
        if not w.size:
            raise RuntimeError("Not enough eigenvalues found.")

        # Sort the eigenvalues and eigenvectors
        w = w[:nroots]
        v = v[:, :nroots]
        conv = [False] * nroots
        if not restart:
            w_prev, conv_prev = _sort_eigenvalues(w_prev, conv_prev, v_prev, v)
        dw = w - w_prev if w_prev.size == w.size else w

        # Find the subspace vectors and matrix-vector products
        vectors = _outer_product_to_subspace(v, basis_vectors)
        matrix_vectors = _outer_product_to_subspace(v, matrix_basis_vectors)

        # Check convergence
        trial_vectors = matrix_vectors - w[:, None] * vectors
        norms = np.real(np.linalg.norm(trial_vectors, axis=1, ord=2))
        conv = [np.abs(dw[k]) < e_tol and norms[k] < loose_tol for k in range(w.size)]
        del matrix_vectors
        if all(conv):
            break

        # Check for restart
        if follow_state and np.max(norms) > 1 and space > nroots + 4:
            vectors = _outer_product_to_subspace(v_prev, basis_vectors)
            restart = True
            continue

        # Remove subspace linear dependency and project out existing basis vectors
        trial_vectors = preconditioner(trial_vectors, w[0] - level_shift)
        trial_vectors = _project_vectors(trial_vectors, basis_vectors)
        trial_vectors = _normalise_vectors(trial_vectors, lindep_tol=lindep_tol)
        if trial_vectors.shape[0] == 0:
            conv = [conv[k] or (norm < loose_tol) for k, norm in enumerate(norms)]
            break

        # Check for restart
        restart = space + nroots > max_space

        # Call the callback
        if callback is not None:
            callback(locals())

    if left:
        # Get the left eigenvectors instead
        wl, vl, v = scipy.linalg.eig(matrix[:space, :space], left=True)
        w, v, idx = pick(wl, v, nroots, basis_vectors=basis_vectors)
        if not w.size:
            raise RuntimeError("Not enough eigenvalues found.")
        w = w[:nroots]
        vectors = _outer_product_to_subspace(vl[:, idx[:nroots]], basis_vectors)

    return conv, w, np.transpose(vectors)
