# For now, consider only a single S=1/2 site with SU(2) symmetry.

function getSymmetryInfo(opts::SpinOptions)
    @assert opts.symmetry == :SU2 || opts.symmetry == :U1 
    "SpinOptions currently only supports :SU2 or :U1 symmetry"
    @assert opts.spin == 1 "SpinOptions currently only supports a single S=1/2 site"

    # Basis: |up>, |down>
    weights = ([(1,), (-1,)],)
    lowering_ops = opts.symmetry == :SU2 ? ([sparse([0 0; 1 0])],) : (Matrix{Int}[],)

    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    if opts.symmetry == :SU2
        mwirops[:S] = (1 / sqrt(2)) * sparse([0 1; 0 0])
        symm = SU{2}
    elseif opts.symmetry == :U1
        mwirops[:Sp] = (-1 / sqrt(2)) * sparse([0 1; 0 0])
        mwirops[:Sz] = (1 / 2) * sparse([1 0; 0 -1])
        mwirops[:Sm] = (1 / sqrt(2)) * sparse([0 0; 1 0])
        symm = U1
    end
    mwirops[:I] = spdiagm(0 => ones(Float64, 2))

    return (symm,), weights, lowering_ops, mwirops
end
