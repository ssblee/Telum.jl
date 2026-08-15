# Telum.jl

`Telum.jl` (Tensor Library for Universal Many-body simulation) is a Julia
library for tensor-network simulations with non-Abelian symmetries.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ssblee/Telum.jl")
```

For local development alongside `LurCGT.jl`:

```julia
using Pkg
Pkg.develop(path = "../LurCGT.jl")
Pkg.develop(path = ".")
```

## Documentation

- [Getting started](getting-started.md): installation, local operators, tensor construction,
  contraction, and leg tags.
- [Tensor operations](tensor-operations.md): decompositions, direct sums, subspace selection,
  and leg utilities.
- [DMRG tutorial](dmrg-tutorial.md): a complete two-site DMRG calculation for the
  Majumdar--Ghosh model.
- [Local spaces](local-spaces.md): mixed symmetries and custom local-space definitions.
- [Advanced topics](advanced.md): shared tensor storage and LurCGT database usage.

## Quick start

```@example quickstart
using Telum

set_accumul_costs!(true)
read_reset_costs!()
```

The block is evaluated while the documentation is built; Documenter renders
the resulting list immediately below it. Use `@example name` for executable,
reproducible examples and plain `julia` fences for illustrative code that
should not run during a documentation build.

The guides above are derived from the maintained Telum tutorial notebooks.
