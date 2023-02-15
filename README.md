# ebcc: Coupled cluster calculations on electron-boson systems

The `ebcc` package implements various coupled cluster (CC) models for both purely electronic and coupled electron-boson models, with a focus on generality and model extensibility.

### Installation

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
ccsd = EBCC(mf)
ccsd.kernel()
```

### Code generation

The models implemented are generated algorithmically from expressions over second quantized operators. Expressions are generated using [`qwick`](https://github.com/obackhouse/qwick) and [`pdaggerq`](https://github.com/edeprince3/pdaggerq) with optimisation of common subexpressions and contraction order achieved using [`drudge`](https://github.com/tschijnmo/drudge) and [`gristmill`](https://github.com/tschijnmo/gristmill).
User-inputted models should operate seamlessly with the solvers by adding files under `ebcc/codegen`, so long as they satisfy the interface.
