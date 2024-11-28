"""Externally corrected coupled cluster."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ebcc import init_logging
from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _inflate, _put
from ebcc.codegen import GecCC, RecCC, UecCC
from ebcc.core.logging import ANSI
from ebcc.util import _BaseOptions

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Optional

    from mypy_extensions import DefaultArg
    from numpy import bool_
    from numpy.typing import NDArray

    from ebcc.cc import gebcc, rebcc, uebcc
    from ebcc.cc.base import BaseEBCC, SpinArrayType, ERIsInputType
    from ebcc.util import Namespace
    from ebcc.ham import Space


@dataclass
class BaseOptions(_BaseOptions):
    """Options for ecCC calculations.

    Args:
        mixed_terms_strategy: Strategy for mixed terms. If "fixed", the mixed terms are calculated
            once and kept fixed. If "update", the mixed terms are recalculated at each iteration.
            If "ignore", the mixed terms are not calculated at all.
    """

    mixed_terms_strategy: Literal["fixed", "update", "ignore"] = "fixed"


def _mask(space: Space, char: str) -> NDArray[bool_]:
    """Get the mask for the given character."""
    return space.omask(char) if char in "ioO" else space.vmask(char)


class BaseExternalCorrection(AbstractContextManager[None]):
    """Context manager for externally correcting an EBCC calculation.

    This context manager is used to externally correct an EBCC calculation by updating the T1 and T2
    amplitudes according to some T3 and T4 amplitudes passed as an argument. The T3 and T4
    amplitudes are assumed to span the active space as defined by `ebcc.space`.
    """

    # Types
    Options: type[BaseOptions] = BaseOptions

    # Attributes
    cc: BaseEBCC
    _update_amps_old: Optional[
        Callable[
            [
                DefaultArg(Optional[ERIsInputType], "eris"),  # noqa: F821
                DefaultArg(Optional[Namespace[SpinArrayType]], "amplitudes"),  # noqa: F821
            ],
            Namespace[SpinArrayType],
        ]
    ]

    def __init__(
        self,
        cc: BaseEBCC,
        amplitudes: Namespace[SpinArrayType],
        options: Optional[BaseOptions] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the context manager.

        Args:
            cc: EBCC object.
            amplitudes: Cluster amplitudes.
            options: Options for the ecCC calculation.
            **kwargs: Additional keyword arguments used to update `options`.
        """
        # Options:
        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)

        # Parameters:
        self.cc = cc
        self.amplitudes = amplitudes

        # Attributes:
        self._update_amps_old = None

        # Logging:
        init_logging(self.cc.log)
        self.cc.log.info(f"Applying {ANSI.m}external corrections{ANSI.R}.")
        self.cc.log.info(
            f" > mixed_terms_strategy:  {ANSI.y}{self.options.mixed_terms_strategy}{ANSI.R}"
        )
        self.cc.log.debug("")

    def __enter__(self) -> None:
        """Enter the context manager."""
        # Save the original update_amps method
        self._update_amps_old = self.cc.update_amps

        # Get the ERIs
        eris = self.cc.get_eris()

        # Get the fixed external corrections
        amplitudes = self._get_external_amplitudes()
        ext = self._update_external_corrections(eris=eris, amplitudes=amplitudes, which="external")
        if self.options.mixed_terms_strategy != "ignore":
            mix = self._update_external_corrections(eris=eris, amplitudes=amplitudes, which="mixed")

        def update_amps(
            eris: Optional[Any] = None,
            amplitudes: Optional[Namespace[SpinArrayType]] = None,
        ) -> Namespace[SpinArrayType]:
            """Update the cluster amplitudes."""
            # Update the mixed terms if necessary
            nonlocal mix
            if self.options.mixed_terms_strategy == "update" or (
                self.options.mixed_terms_strategy == "fixed" and mix is None
            ):
                if amplitudes is None:
                    raise ValueError(
                        "Cannot update mixed terms of the external corrections without "
                        "providing the cluster amplitudes."
                    )
                amplitudes_full = amplitudes.copy()
                for key, val in self.amplitudes.items():
                    if key not in amplitudes_full:
                        amplitudes_full[key] = val
                mix = self._update_external_corrections(
                    eris=eris, amplitudes=amplitudes_full, which="mixed"
                )

            # Perform the original update_amps method
            assert self._update_amps_old is not None
            amps = self._update_amps_old(eris=eris, amplitudes=amplitudes)

            # Add the external corrections
            self._add_to_amps(amps, ext)
            if self.options.mixed_terms_strategy != "ignore":
                self._add_to_amps(amps, mix)

            return amps

        # Replace the original update_amps method
        update_amps.__doc__ = self._update_amps_old.__doc__
        self.cc.update_amps = update_amps  # type: ignore[method-assign]

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager."""
        # Restore the original update_amps method
        assert self._update_amps_old is not None
        self.cc.update_amps = self._update_amps_old  # type: ignore[method-assign]
        self._update_amps_old = None

    @abstractmethod
    def _add_to_amps(
        self,
        amps: Namespace[SpinArrayType],
        exts: Namespace[Namespace[SpinArrayType]],
    ) -> None:
        """Add the external corrections to the cluster amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            exts: External corrections.
        """
        pass

    @abstractmethod
    def _update_external_corrections(
        self,
        eris: Optional[Any] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        which: Literal["external", "mixed"] = "external",
    ) -> Namespace[SpinArrayType]:
        """Update the external corrections to the cluster amplitudes.

        Args:
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            which: Type of external corrections to calculate. If "external", return the fully
                external corrections. If "mixed", return the mixed corrections that may also depend
                on the cluster amplitudes.

        Returns:
            External corrections to the cluster amplitudes.
        """
        pass

    @abstractmethod
    def _get_external_amplitudes(self) -> Namespace[SpinArrayType]:
        """Get the external amplitudes.

        Pre-processes the external corrections by inflating any amplitudes that belong to this
        ansatz into the full space. Amplitudes that do not belong to this ansatz remain in the
        active space.

        Returns:
            External amplitudes.
        """
        pass


class ExternalCorrectionREBCC(BaseExternalCorrection):
    """Context manager for externally correcting a REBCC calculation.

    This context manager is used to externally correct an EBCC calculation by updating the T1 and T2
    amplitudes according to some T3 and T4 amplitudes passed as an argument. The T3 and T4
    amplitudes are assumed to span the active space as defined by `ebcc.space`.
    """

    # Attributes
    cc: rebcc.REBCC
    _update_amps_old: Optional[
        Callable[
            [
                DefaultArg(Optional[rebcc.ERIsInputType], "eris"),  # noqa: F821
                DefaultArg(Optional[Namespace[rebcc.SpinArrayType]], "amplitudes"),  # noqa: F821
            ],
            Namespace[rebcc.SpinArrayType],
        ]
    ]

    def _add_to_amps(
        self,
        amps: Namespace[rebcc.SpinArrayType],
        exts: Namespace[Namespace[rebcc.SpinArrayType]],
    ) -> None:
        """Add the external corrections to the cluster amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            exts: External corrections.
        """
        for name in exts.keys():
            if name not in amps:
                continue
            for key in exts[name].keys():
                masks = tuple(_mask(self.cc.space, k) for k in key)
                indices = np.ix_(*masks)
                _put(
                    amps[name],
                    indices,  # type: ignore
                    amps[name][indices] + exts[name][key],
                )

    def _update_external_corrections(
        self,
        eris: Optional[Any] = None,
        amplitudes: Optional[Namespace[rebcc.SpinArrayType]] = None,
        which: Literal["external", "mixed"] = "external",
    ) -> Namespace[rebcc.SpinArrayType]:
        """Update the external corrections to the cluster amplitudes.

        Args:
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            which: Type of external corrections to calculate. If "external", return the fully
                external corrections. If "mixed", return the mixed corrections that may also depend
                on the cluster amplitudes.

        Returns:
            External corrections to the cluster amplitudes.
        """
        # Calculate residuals for corrections to the T1 and T2 amplitudes
        if amplitudes is None:
            kwargs = self.cc._pack_codegen_kwargs(eris=eris)
        else:
            kwargs = self.cc._pack_codegen_kwargs(dict(amplitudes), eris=eris)
        ext = getattr(RecCC, f"update_amps_{which}")(**kwargs)
        ext = util.Namespace(**{key.rstrip("new"): val for key, val in ext.items()})

        # Get the amplitudes from the residuals
        for name, _, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.cc.spin_type):
            if name not in ext:
                continue
            for key in ext[name].keys():
                ext[name][key] /= self.cc.energy_sum(key)

        return ext

    def _get_external_amplitudes(self) -> Namespace[rebcc.SpinArrayType]:
        """Get the external amplitudes.

        Pre-processes the external corrections by inflating any amplitudes that belong to this
        ansatz into the full space. Amplitudes that do not belong to this ansatz remain in the
        active space.

        Returns:
            External amplitudes.
        """
        # Inflate the external amplitudes
        amplitudes_ext = self.amplitudes.copy()
        for name, target_key, n in self.cc.ansatz.fermionic_cluster_ranks(
            spin_type=self.cc.spin_type
        ):
            actual_key = target_key.upper()
            if name in amplitudes_ext.keys():
                target_shape = tuple(self.cc.space.size(k) for k in target_key)
                masks = tuple(_mask(self.cc.space, k) for k in actual_key)
                indices = np.ix_(*masks)
                amplitudes_ext[name] = _inflate(
                    target_shape,
                    indices,  # type: ignore
                    amplitudes_ext[name],
                )

        return amplitudes_ext


class ExternalCorrectionUEBCC(BaseExternalCorrection):
    """Context manager for externally correcting a UEBCC calculation.

    This context manager is used to externally correct an EBCC calculation by updating the T1 and T2
    amplitudes according to some T3 and T4 amplitudes passed as an argument. The T3 and T4
    amplitudes are assumed to span the active space as defined by `ebcc.space`.
    """

    # Attributes
    cc: uebcc.UEBCC
    _update_amps_old: Optional[
        Callable[
            [
                DefaultArg(Optional[uebcc.ERIsInputType], "eris"),  # noqa: F821
                DefaultArg(Optional[Namespace[uebcc.SpinArrayType]], "amplitudes"),  # noqa: F821
            ],
            Namespace[uebcc.SpinArrayType],
        ]
    ]

    def _add_to_amps(
        self,
        amps: Namespace[uebcc.SpinArrayType],
        exts: Namespace[Namespace[uebcc.SpinArrayType]],
    ) -> None:
        """Add the external corrections to the cluster amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            exts: External corrections.
        """
        for name in exts.keys():
            if name not in amps:
                continue
            for comb in exts[name].keys():
                for key in exts[name][comb].keys():
                    masks = tuple(_mask(self.cc.space["ab".index(s)], k) for s, k in zip(comb, key))
                    indices = np.ix_(*masks)
                    _put(
                        amps[name][comb],
                        indices,  # type: ignore
                        amps[name][comb][indices] + exts[name][comb][key],
                    )

    def _update_external_corrections(
        self,
        eris: Optional[Any] = None,
        amplitudes: Optional[Namespace[uebcc.SpinArrayType]] = None,
        which: Literal["external", "mixed"] = "external",
    ) -> Namespace[uebcc.SpinArrayType]:
        """Update the external corrections to the cluster amplitudes.

        Args:
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            which: Type of external corrections to calculate. If "external", return the fully
                external corrections. If "mixed", return the mixed corrections that may also depend
                on the cluster amplitudes.

        Returns:
            External corrections to the cluster amplitudes.
        """
        # Calculate residuals for corrections to the T1 and T2 amplitudes
        if amplitudes is None:
            kwargs = self.cc._pack_codegen_kwargs(eris=eris)
        else:
            kwargs = self.cc._pack_codegen_kwargs(dict(amplitudes), eris=eris)
        ext = getattr(UecCC, f"update_amps_{which}")(**kwargs)
        ext = util.Namespace(**{key.rstrip("new"): val for key, val in ext.items()})

        # Get the amplitudes from the residuals
        for name, _, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.cc.spin_type):
            if name not in ext:
                continue
            for comb in util.generate_spin_combinations(n, unique=True):
                for key in ext[name][comb].keys():
                    subscript, _ = util.combine_subscripts(key, comb)
                    ext[name][comb][key] /= self.cc.energy_sum(key, comb)
                    ext[name][comb][key] = util.symmetrise(
                        subscript, ext[name][comb][key], symmetry="-" * (2 * n)
                    )

        return ext

    def _get_external_amplitudes(self) -> Namespace[uebcc.SpinArrayType]:
        """Get the external amplitudes.

        Pre-processes the external corrections by inflating any amplitudes that belong to this
        ansatz into the full space. Amplitudes that do not belong to this ansatz remain in the
        active space.

        Returns:
            External amplitudes.
        """
        # Inflate the external amplitudes
        amplitudes_ext = self.amplitudes.copy()
        for name, target_key, n in self.cc.ansatz.fermionic_cluster_ranks(
            spin_type=self.cc.spin_type
        ):
            actual_key = target_key.upper()
            if name in amplitudes_ext.keys():
                for comb in util.generate_spin_combinations(n, unique=True):
                    spaces = tuple(self.cc.space["ab".index(s)] for s in comb)
                    target_shape = tuple(space.size(k) for space, k in zip(spaces, target_key))
                    masks = tuple(_mask(space, k) for space, k in zip(spaces, actual_key))
                    indices = np.ix_(*masks)
                    amplitudes_ext[name][comb] = _inflate(
                        target_shape,
                        indices,  # type: ignore
                        amplitudes_ext[name][comb],
                    )

        return amplitudes_ext


class ExternalCorrectionGEBCC(BaseExternalCorrection):
    """Context manager for externally correcting a GEBCC calculation.

    This context manager is used to externally correct an EBCC calculation by updating the T1 and T2
    amplitudes according to some T3 and T4 amplitudes passed as an argument. The T3 and T4
    amplitudes are assumed to span the active space as defined by `ebcc.space`.
    """

    # Attributes
    cc: gebcc.GEBCC
    _update_amps_old: Optional[
        Callable[
            [
                DefaultArg(Optional[gebcc.ERIsInputType], "eris"),  # noqa: F821
                DefaultArg(Optional[Namespace[gebcc.SpinArrayType]], "amplitudes"),  # noqa: F821
            ],
            Namespace[gebcc.SpinArrayType],
        ]
    ]

    def _add_to_amps(
        self,
        amps: Namespace[gebcc.SpinArrayType],
        exts: Namespace[Namespace[gebcc.SpinArrayType]],
    ) -> None:
        """Add the external corrections to the cluster amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            exts: External corrections.
        """
        for name in exts.keys():
            if name not in amps:
                continue
            for key in exts[name].keys():
                masks = tuple(_mask(self.cc.space, k) for k in key)
                indices = np.ix_(*masks)
                _put(
                    amps[name],
                    indices,  # type: ignore
                    amps[name][indices] + exts[name][key],
                )

    def _update_external_corrections(
        self,
        eris: Optional[Any] = None,
        amplitudes: Optional[Namespace[gebcc.SpinArrayType]] = None,
        which: Literal["external", "mixed"] = "external",
    ) -> Namespace[rebcc.SpinArrayType]:
        """Update the external corrections to the cluster amplitudes.

        Args:
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            which: Type of external corrections to calculate. If "external", return the fully
                external corrections. If "mixed", return the mixed corrections that may also depend
                on the cluster amplitudes.

        Returns:
            External corrections to the cluster amplitudes.
        """
        # Calculate residuals for corrections to the T1 and T2 amplitudes
        if amplitudes is None:
            kwargs = self.cc._pack_codegen_kwargs(eris=eris)
        else:
            kwargs = self.cc._pack_codegen_kwargs(dict(amplitudes), eris=eris)
        ext = getattr(GecCC, f"update_amps_{which}")(**kwargs)
        ext = util.Namespace(**{key.rstrip("new"): val for key, val in ext.items()})

        # Get the amplitudes from the residuals
        for name, _, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.cc.spin_type):
            if name not in ext:
                continue
            for key in ext[name].keys():
                ext[name][key] /= self.cc.energy_sum(key)

        return ext

    def _get_external_amplitudes(self) -> Namespace[gebcc.SpinArrayType]:
        """Get the external amplitudes.

        Pre-processes the external corrections by inflating any amplitudes that belong to this
        ansatz into the full space. Amplitudes that do not belong to this ansatz remain in the
        active space.

        Returns:
            External amplitudes.
        """
        # Inflate the external amplitudes
        amplitudes_ext = self.amplitudes.copy()
        for name, target_key, n in self.cc.ansatz.fermionic_cluster_ranks(
            spin_type=self.cc.spin_type
        ):
            actual_key = target_key.upper()
            if name in amplitudes_ext.keys():
                target_shape = tuple(self.cc.space.size(k) for k in target_key)
                masks = tuple(_mask(self.cc.space, k) for k in actual_key)
                indices = np.ix_(*masks)
                amplitudes_ext[name] = _inflate(
                    target_shape,
                    indices,  # type: ignore
                    amplitudes_ext[name],
                )

        return amplitudes_ext
