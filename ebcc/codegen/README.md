# Code generation

The models implemented are generated algorithmically from expressions over second quantized operators. Expressions are generated using [`qwick`](https://github.com/obackhouse/qwick) with optimisation of common subexpressions and contraction order achieved using [`drudge`](https://github.com/tschijnmo/drudge) and [`gristmill`](https://github.com/tschijnmo/gristmill). The scripts contained in `ebcc.codegen` can be extended to new ansatz.

### Second quantized expressions

Expressions in terms of second quantized expressions are generated using [`qwick`](https://github.com/obackhouse/qwick), a `C++` version of the [`wick`](https://github.com/awhite862/wick) package, and [`pdagger`](https://github.com/edeprince3/pdaggerq). These expressions are generated in spin-orbital notation, providing the models for `GEBCC`.

### Spin integration

The expressions generated in spin-orbital notation can be spin-integrated to yield expressions in an unrestricted basis, providing the models for `UEBCC`, using `qwick.codegen`. These equations can be subsequently manipulated to give restricted basis expressions by applying symmetry in the fermionic spin index providing models for `REBCC`.

### Optimisation

All expressions are optimised for contraction order and common subexpressions using the [`drudge`](https://github.com/tschijnmo/drudge) and [`gristmill`](https://github.com/tschijnmo/gristmill) packages.
