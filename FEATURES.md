- Coupled cluster calculations using a wide range of ansatzes, as summarised below
- Lambda equation solver
- Equation-of-motion solver
- Density matrices
- Brueckner orbital calculations
- Frozen and active space constraints

The following table summarises the available methods and routines for the ansatz currently treated by code generation, in the three spin cases:

| Ansatz      |   T   |   Λ   |  IP   |  EA   |  EE   |  DM1  |  DM2  |  BDM  |
| :---------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CCD         |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |   -   |
| CCSD        |  RUG  |  RUG  |  UG   |  UG   |  UG   |  RUG  |  RUG  |   -   |
| CCSDT       |  Rug  |       |       |       |       |       |       |   -   |
| CCSDTQ      |   g   |       |       |       |       |       |       |   -   |
| CCSD(T)     |  RuG  |       |       |       |       |       |       |   -   |
| CC2         |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |   -   |
| CC3         |  RUG  |       |       |       |       |       |       |   -   |
| QCISD       |  RUG  |       |       |       |       |       |       |   -   |
| CCSD-S-1-1  |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |  RUG  |
| CCSD-SD-1-1 |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |  RUG  |
| CCSD-SD-1-2 |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |  RUG  |

- R, U, G indicate availability of restricted, unrestricted, and generalised codes.
- Capital letters (R rather than r) indicates that the expressions are optimised for contraction order and subexpression elimination.
- T, Λ are the excitation and de-excitation amplitude availabilities, respectively, the former allowing the coupled cluster solver and the latter allowing the lambda solver.
- IP, EA, EE indicate availability of the corresponding equation of motion (EOM) functionalities.
- DM1, DM2 indicate availability of the one- and two-particle reduced density matrices.
- BDM availability includes the single boson density matrix, bosonic one-particle reduced density matrix, and the electron-boson coupling reduced density matrix.
