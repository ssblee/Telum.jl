# For now, consider only a single S=1/2 site with SU(2) symmetry.

function getSymmetryInfo(opts::SpinOptions)
    @assert opts.symmetry == SU{2} || opts.symmetry == U1 
    "SpinOptions currently only supports SU{2} or U1 symmetry"
    @assert opts.spin == 1//2 "SpinOptions currently only supports a single S=1/2 site"

    # Basis: |up>, |down>
    weights = ([(1,), (-1,)],)
    lowering_ops = opts.symmetry == SU{2} ? ([sparse([0 0; 1 0])],) : (Matrix{Int}[],)

    mwirops = Dict{Symbol, Tuple{AbstractMatrix{Int}, Float64}}()
    if opts.symmetry == SU{2}
        mwirops[:S] = (sparse([0 1; 0 0]), 1 / sqrt(2))
    elseif opts.symmetry == U1
        mwirops[:Sp] = (sparse([0 1; 0 0]), 1 / sqrt(2))
        mwirops[:Sz] = (sparse([1 0; 0 -1]), 1 / 2)
        mwirops[:Sm] = (sparse([0 0; 1 0]), 1 / sqrt(2))
    end
    mwirops[:I] = (sparse(I, 2, 2), 1.0)

    return (opts.symmetry,), weights, lowering_ops, mwirops
end
