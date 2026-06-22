# For now, consider only a single S=1/2 site with SU(2) symmetry.

function getSymmetryInfo(opts::SpinOptions)
    @assert opts.symmetry === nothing || opts.symmetry == :SU2 || opts.symmetry == :U1 
    "SpinOptions currently only supports :SU2, :U1, or no symmetry"
    @assert opts.spin == 1 "SpinOptions currently only supports a single S=1/2 site"

    if opts.symmetry === nothing
        mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
        mwirops[:Sp] = (-1 / sqrt(2)) * sparse([0.0 1.0; 0.0 0.0])
        mwirops[:Sz] = (1 / 2) * sparse([1.0 0.0; 0.0 -1.0])
        mwirops[:Sm] = (1 / sqrt(2)) * sparse([0.0 0.0; 1.0 0.0])
        mwirops[:I] = spdiagm(0 => ones(Float64, 2))
        return (), (), (), mwirops
    end

    # Basis: |up>, |down>
    weights = ([(1,), (-1,)],)
    lowering_ops = opts.symmetry == :SU2 ? ([sparse(Float64[0 0; 1 0])],) : (Matrix{Float64}[],)

    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    if opts.symmetry == :SU2
        mwirops[:S] = (1 / sqrt(2)) * sparse([0.0 1.0; 0.0 0.0])
        symm = SU{2}
    elseif opts.symmetry == :U1
        mwirops[:Sp] = (-1 / sqrt(2)) * sparse([0.0 1.0; 0.0 0.0])
        mwirops[:Sz] = (1 / 2) * sparse([1.0 0.0; 0.0 -1.0])
        mwirops[:Sm] = (1 / sqrt(2)) * sparse([0.0 0.0; 1.0 0.0])
        symm = U1
    end
    mwirops[:I] = spdiagm(0 => ones(Float64, 2))

    return (symm,), weights, lowering_ops, mwirops
end
