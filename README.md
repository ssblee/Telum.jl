# QSpaces.jl

`QSpaces.jl` is the tensor-space layer extracted from `CGTfromInts`. It owns
`QSpace`, `LurTensor`, local-space construction, tensor contraction, SVD/eigen
helpers, and QIndex manipulation utilities.

## Installation

Once this repository is pushed to its own remote, install it with:

```julia
using Pkg
Pkg.add(url="<QSpaces-repo-url>")
```

Because `QSpaces.jl` depends on `LurCGT.jl`, local development typically looks
like this:

```julia
using Pkg
Pkg.develop(path="../LurCGT.jl")
Pkg.develop(path=".")
```

Typical usage:

```julia
using LurCGT
using QSpaces
```

## Testing

```julia
using Pkg
Pkg.develop(path="../LurCGT.jl")
Pkg.test()
```

