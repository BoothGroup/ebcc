# ebcc: Coupled cluster calculations on electron-boson systems

[![CI](https://github.com/BoothGroup/ebcc/workflows/CI/badge.svg)](https://github.com/BoothGroup/ebcc/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/BoothGroup/ebcc/branch/master/graph/badge.svg?token=QZQZQ2ZQ2Z)](https://codecov.io/gh/BoothGroup/ebcc)
[![PyPI version](https://badge.fury.io/py/ebcc.svg)](https://badge.fury.io/py/ebcc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `ebcc` package implements various coupled cluster (CC) models for both purely electronic and coupled electron-boson models, with a focus on generality and model extensibility.

For a summary of the implemented models, see the [FEATURES.md](FEATURES.md) file.

### Installation

From PyPI:

```bash
pip install ebcc
```

From source:

```bash
git clone https://github.com/BoothGroup/ebcc
pip install .
```

### Usage

The implemented models are built upon the mean-field objects of [`pyscf`](https://github.com/pyscf/pyscf):

```python
from pyscf import gto, scf
from ebcc import EBCC
mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="cc-pvdz")
mf = scf.RHF(mol)
mf.kernel()
ccsd = EBCC(mf, ansatz="CCSD")
ccsd.kernel()
```

### Code generation

The models implemented are generated algorithmically from expressions over second quantized operators. The scripts for generating these models are found in the `codegen` directory on the `bootstrap` branch.
User-inputted models should operate seamlessly with the solvers by adding files under `ebcc/codegen`, so long as they satisfy the interface.
