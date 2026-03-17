# For now, consider only a single S=1/2 site with SU(2) symmetry.

function getSymmetryInfo(opts::SpinOptions)
    @assert opts.symmetry == SU{2} "SpinOptions currently only supports SU{2} symmetry"
    @assert opts.spin == 1//2 "SpinOptions currently only supports a single S=1/2 site"

    # Basis: |up>, |down>
    weights = ([(1,), (-1,)],)
    lowering_ops = ([sparse([0 0; 1 0])],)

    mwirops = Dict{Symbol, Tuple{AbstractMatrix{Int}, Float64}}()
    mwirops[:S] = (sparse([0 1; 0 0]), 1 / sqrt(2))
    mwirops[:I] = (sparse(I, 2, 2), 1.0)

    return (opts.symmetry,), weights, lowering_ops, mwirops
end
