# Code generation

The models implemented are generated algorithmically from expressions over second quantized operators. Expressions are generated using [`pdaggerq`](https://github.com/edeprince3/pdaggerq) with optimisation of common subexpressions and contraction order achieved using [`drudge`](https://github.com/tschijnmo/drudge) and [`gristmill`](https://github.com/tschijnmo/gristmill). The spin integrated and code synthesis is handled by [`albert`](https://github.com/obackhouse/albert). The scripts contained in `ebcc.codegen` can be extended to new ansatzes.
