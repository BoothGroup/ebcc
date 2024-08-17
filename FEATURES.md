- Coupled cluster calculations using a wide range of ansatzes, as summarised below
- Lambda equation solver
- Equation-of-motion solver
- Density matrices
- Frozen and active space constraints
- Brueckner orbital calculations
- Frozen natural orbital calculations
- Single- and mixed-precision calculations

The following table summarises the available methods and routines for the ansatz currently treated by code generation, in the three spin cases:

| Ansatz      |   E   |   T   |   Λ   |  IP   |  EA   |  EE   |  DM1  |  DM2  |  BDM  |
| :---------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MP2         |  RUG  |   -   |   -   |  RUG  |  RUG  |  UG   |  RUG  |  RUG  |   -   |
| MP3         |  RUG  |       |       |       |       |       |       |       |   -   |
| CCD         |  RUG  |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |   -   |
| CCSD        |  RUG  |  RUG  |  RUG  |  RUG  |  RUG  |  UG   |  RUG  |  RUG  |   -   |
| CCSDT       |  RUG  |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |   -   |
| CCSDTQ      |   g   |   g   |       |       |       |       |       |       |   -   |
| CCSD(T)     |  RuG  |  RuG  |       |       |       |       |       |       |   -   |
| CCSDt       |  RG   |  RG   |       |       |       |       |       |       |   -   |
| CCSDt'      |  RUG  |  RUG  |       |       |       |       |       |       |   -   |
| CC2         |  RUG  |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |   -   |
| CC3         |  RUG  |  RUG  |       |       |       |       |       |       |   -   |
| QCISD       |  RUG  |  RUG  |       |       |       |       |       |       |   -   |
| DCD         |  RU   |  RU   |       |       |       |       |       |       |   -   |
| DCSD        |  RU   |  RU   |       |       |       |       |       |       |   -   |
| DF-CCD      |  RU   |  RU   |  RU   |       |       |       |       |       |   -   |
| DF-CCSD     |  RU   |  RU   |  RU   |       |       |       |  RU   |  RU   |   -   |
| DF-CC2      |  RU   |  RU   |  RU   |       |       |       |  RU   |  RU   |   -   |
| DF-QCISD    |  RU   |  RU   |       |       |       |       |       |       |   -   |
| DF-DCD      |  RU   |  RU   |       |       |       |       |       |       |   -   |
| DF-DCSD     |  RU   |  RU   |       |       |       |       |       |       |   -   |
| CCSD-S-1-1  |  RUG  |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |  RUG  |
| CCSD-SD-1-1 |  RUG  |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |  RUG  |
| CCSD-SD-1-2 |  RUG  |  RUG  |  RUG  |       |       |       |  RUG  |  RUG  |  RUG  |

- R, U, G indicate availability of restricted, unrestricted, and generalised codes.
- Capital letters (R rather than r) indicates that the expressions are optimised for contraction order and subexpression elimination.
- DF in the ansatz name indicates methods specialised for density-fitted integrals.
- E is the correlation energy.
- T, Λ are the excitation and de-excitation amplitude availabilities, respectively, the former allowing the coupled cluster solver and the latter allowing the lambda solver.
- IP, EA, EE indicate availability of the corresponding equation of motion (EOM) functionalities.
- DM1, DM2 indicate availability of the one- and two-particle reduced density matrices.
- BDM availability includes the single boson density matrix, bosonic one-particle reduced density matrix, and the electron-boson coupling reduced density matrix.
