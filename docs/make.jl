using Documenter
using Telum

makedocs(
    sitename = "Telum.jl",
    authors = "Kiyeon Kim",
    repo = "github.com/ssblee/Telum.jl/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Tensor operations" => "tensor-operations.md",
        "DMRG tutorial" => "dmrg-tutorial.md",
        "Local spaces" => "local-spaces.md",
        "Advanced topics" => "advanced.md",
    ],
)

deploydocs(
    repo = "github.com/ssblee/Telum.jl.git",
    devbranch = "main",
)
