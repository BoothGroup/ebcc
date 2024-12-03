"""Tailored coupled cluster."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ebcc import init_logging
from ebcc import numpy as np
from ebcc.backend import _put
from ebcc.core.logging import ANSI
from ebcc.util import _BaseOptions

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    from mypy_extensions import DefaultArg
    from numpy import bool_
    from numpy.typing import NDArray

    from ebcc.cc import gebcc, rebcc, uebcc
    from ebcc.cc.base import BaseEBCC, ERIsInputType, SpinArrayType
    from ebcc.ham import Space
    from ebcc.util import Namespace


@dataclass
class BaseOptions(_BaseOptions):
    """Options for tCC calculations."""

    pass


def _mask(space: Space, char: str) -> NDArray[bool_]:
    """Get the mask for the given character."""
    return space.omask(char) if char in "ioO" else space.vmask(char)


class BaseTailor(AbstractContextManager[None]):
    """Context manager for tailoring an EBCC calculation.

    This context manager is used to tailor an EBCC calculation by updating the T1 and T2 amplitudes
    to constrain the active space to known T1 and T2 amplitudes spanning the active space as defined
    by `ebcc.space`.
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
            options: Options for the tCC calculation.
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
        self.cc.log.info(f"Applying {ANSI.m}tailoring{ANSI.R}.")
        self.cc.log.debug("")

    def __enter__(self) -> None:
        """Enter the context manager."""
        # Save the original update_amps method
        self._update_amps_old = self.cc.update_amps

        def update_amps(
            eris: Optional[Any] = None,
            amplitudes: Optional[Namespace[SpinArrayType]] = None,
        ) -> Namespace[SpinArrayType]:
            """Update the cluster amplitudes."""
            # Perform the original update_amps method
            assert self._update_amps_old is not None
            amps = self._update_amps_old(eris=eris, amplitudes=amplitudes)

            # Tailor the amplitudes
            self._set_active_space(amps, self.amplitudes)

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
    def _set_active_space(
        self,
        amps: Namespace[SpinArrayType],
        amps_active: Namespace[SpinArrayType],
    ) -> None:
        """Set the active space amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            amps_active: Cluster amplitudes for the active space.
        """
        pass


class TailorREBCC(BaseTailor):
    """Context manager for tailoring an REBCC calculation.

    This context manager is used to tailor an EBCC calculation by updating the T1 and T2 amplitudes
    to constrain the active space to known T1 and T2 amplitudes spanning the active space as defined
    by `ebcc.space`.
    """

    # Attributes
    cc: rebcc.REBCC
    _update_amps_old: Optional[
        Callable[
            [
                DefaultArg(Optional[rebcc.ERIsInputType], "eris"),  # noqa: F821
                DefaultArg(Optional[Namespace[rebcc.SpinArrayType]], "amplitudes"),  # noqa: F821
            ],
            Namespace[SpinArrayType],
        ]
    ]

    def _set_active_space(
        self,
        amps: Namespace[rebcc.SpinArrayType],
        amps_active: Namespace[rebcc.SpinArrayType],
    ) -> None:
        """Set the active space amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            amps_active: Cluster amplitudes for the active space.
        """
        for name, key, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.cc.spin_type):
            if name not in amps_active:
                continue
            masks = tuple(_mask(self.cc.space, k.upper()) for k in key)
            indices = np.ix_(*masks)
            _put(amps[name], indices, amps_active[name])  # type: ignore


class TailorUEBCC(BaseTailor):
    """Context manager for tailoring a UEBCC calculation.

    This context manager is used to tailor an EBCC calculation by updating the T1 and T2 amplitudes
    to constrain the active space to known T1 and T2 amplitudes spanning the active space as defined
    by `ebcc.space`.
    """

    # Attributes
    cc: uebcc.UEBCC
    _update_amps_old: Optional[
        Callable[
            [
                DefaultArg(Optional[uebcc.ERIsInputType], "eris"),  # noqa: F821
                DefaultArg(Optional[Namespace[uebcc.SpinArrayType]], "amplitudes"),  # noqa: F821
            ],
            Namespace[SpinArrayType],
        ]
    ]

    def _set_active_space(
        self,
        amps: Namespace[uebcc.SpinArrayType],
        amps_active: Namespace[uebcc.SpinArrayType],
    ) -> None:
        """Set the active space amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            amps_active: Cluster amplitudes for the active space.
        """
        for name, key, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.cc.spin_type):
            if name not in amps_active:
                continue
            for comb in amps[name].keys():
                masks = tuple(
                    _mask(self.cc.space["ab".index(s)], k.upper()) for s, k in zip(comb, key)
                )
                indices = np.ix_(*masks)
                _put(amps[name][comb], indices, amps_active[name][comb])  # type: ignore


class TailorGEBCC(BaseTailor):
    """Context manager for tailoring a GEBCC calculation.

    This context manager is used to tailor an EBCC calculation by updating the T1 and T2 amplitudes
    to constrain the active space to known T1 and T2 amplitudes spanning the active space as defined
    by `ebcc.space`.
    """

    # Attributes
    cc: gebcc.GEBCC
    _update_amps_old: Optional[
        Callable[
            [
                DefaultArg(Optional[gebcc.ERIsInputType], "eris"),  # noqa: F821
                DefaultArg(Optional[Namespace[gebcc.SpinArrayType]], "amplitudes"),  # noqa: F821
            ],
            Namespace[SpinArrayType],
        ]
    ]

    def _set_active_space(
        self,
        amps: Namespace[gebcc.SpinArrayType],
        amps_active: Namespace[gebcc.SpinArrayType],
    ) -> None:
        """Set the active space amplitudes in-place.

        Args:
            amps: Cluster amplitudes.
            amps_active: Cluster amplitudes for the active space.
        """
        for name, key, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.cc.spin_type):
            if name not in amps_active:
                continue
            masks = tuple(_mask(self.cc.space, k.upper()) for k in key)
            indices = np.ix_(*masks)
            _put(amps[name], indices, amps_active[name])  # type: ignore
