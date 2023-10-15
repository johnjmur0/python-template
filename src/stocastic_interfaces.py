import numpy as np
from typing import Protocol, Optional


# NOTE first time seeing protocols.
# Seem inferior to ABCs imo, but do allow for classes to implement multiple protocols (although loosely)
class Drift(Protocol):
    @property
    def sample_size(self) -> int:
        raise NotImplementedError("Must be implemented in extension class")

    @property
    def n_procs(self) -> int:
        raise NotImplementedError("Must be implemented in extension class")

    def get_mu(self, random_state: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError("Must be implemented in extension class")


class Sigma(Protocol):
    @property
    def sample_size(self) -> int:
        raise NotImplementedError("Must be implemented in extension class")

    @property
    def n_procs(self) -> int:
        raise NotImplementedError("Must be implemented in extension class")

    def get_sigma(self, random_state: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError("Must be implemented in extension class")


class Init_P(Protocol):
    @property
    def n_procs(self) -> int:
        raise NotImplementedError("Must be implemented in extension class")

    def get_P_0(self, random_state: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError("Must be implemented in extension class")
