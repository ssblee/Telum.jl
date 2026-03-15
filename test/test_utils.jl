using LinearAlgebra
using SparseArrayKit
using Test
const _compress_sector = QSpaces._compress_sector
const _contract_om_axis = QSpaces._contract_om_axis
const _row_qlabel = QSpaces._row_qlabel
const change_dir = QSpaces.change_dir
const contract_v2 = QSpaces.contract_v2
⊗(a, b) = kron(b, a)

"""
    _linear_interp_extrap(x, y, xq)

Evaluate the piecewise-linear interpolant defined by sample points `(x, y)` at
the query points `xq`, using linear extrapolation outside the sampled domain.

This helper reproduces the behavior needed for the MATLAB routine
`interp1(..., "linear", "extrap")` used by the logarithmic discretization
code. The abscissae `x` must be sorted in strictly increasing order and contain
at least two points.
"""
function _linear_interp_extrap(x::AbstractVector, y::AbstractVector,
                               xq::AbstractVector)
    length(x) == length(y) ||
        throw(ArgumentError("`x` and `y` must have the same length."))
    length(x) ≥ 2 ||
        throw(ArgumentError("At least two sample points are required for interpolation."))

    T = promote_type(float(eltype(x)), float(eltype(y)), float(eltype(xq)))
    xp = T.(x)
    yp = T.(y)
    xqp = T.(xq)

    issorted(xp) || throw(ArgumentError("`x` must be sorted in increasing order."))
    all(diff(xp) .> zero(T)) ||
        throw(ArgumentError("`x` entries must be strictly increasing."))

    result = similar(xqp, T)

    @inbounds for i in eachindex(xqp)
        xi = xqp[i]
        seg = if xi <= xp[1]
            1
        elseif xi >= xp[end]
            length(xp) - 1
        else
            searchsortedlast(xp, xi)
        end

        x1 = xp[seg]
        x2 = xp[seg + 1]
        y1 = yp[seg]
        y2 = yp[seg + 1]
        result[i] = y1 + (y2 - y1) * (xi - x1) / (x2 - x1)
    end

    return result
end

"""
    doCLD_1side(oz, rho, nstep) -> repE, repT

Compute the representative energies and interval weights for one energy half
of the Campo-Oliveira logarithmic discretization.

Arguments
- `oz`: Positive logarithmic grid points in increasing order.
- `rho`: Hybridization values sampled on `oz`.
- `nstep`: Number of grid samples inside each discretization interval.

Returns
- `repE`: Representative energies for each interval.
- `repT`: Integrated hybridization weights for each interval.

This is a direct Julia translation of the helper routine from `doCLD (1).m`.
The returned quantities correspond to the symbols `\\mathcal{E}` and the
interval integrals used in the star-geometry representation.
"""
function doCLD_1side(oz::AbstractVector, rho::AbstractVector, nstep::Integer)
    length(oz) == length(rho) ||
        throw(ArgumentError("`oz` and `rho` must have the same length."))
    nstep > 0 || throw(ArgumentError("`nstep` must be positive."))

    T = promote_type(float(eltype(oz)), float(eltype(rho)))
    ozp = T.(oz)
    rhop = T.(rho)

    ids = length(rhop):-nstep:1
    repT = zeros(T, length(ids) - 1)
    repE = similar(repT)

    @inbounds for itx in eachindex(repT)
        lo = ids[itx + 1]
        hi = ids[itx]
        ozseg = @view ozp[lo:hi]
        rhoseg = @view rhop[lo:hi]

        dω = ozseg[2:end] .- ozseg[1:end-1]
        repT[itx] = sum((rhoseg[2:end] .+ rhoseg[1:end-1]) .* dω) / 2

        repE_denom = (rhoseg[end] - rhoseg[1]) +
                     sum(((ozseg[2:end] .* rhoseg[1:end-1] .-
                           ozseg[1:end-1] .* rhoseg[2:end]) ./ dω) .*
                         log.(abs.(ozseg[2:end] ./ ozseg[1:end-1])))
        repE[itx] = repT[itx] / repE_denom
    end

    return repE, repT
end

"""
    doCLD(ozin, rhoV2in, Lambda, N; estep=10, emax=maximum(abs, ozin), emin=100eps(Float64)) -> ff, gg

Perform Campo-Oliveira logarithmic discretization of a bath hybridization
function and map the resulting star geometry to a Wilson chain.

Arguments
- `ozin`: Frequency grid used to define the hybridization function
  `Δ(ω) ≡ rhoV2in(ω)`. The first and last entries determine the sampled
  bandwidth.
- `rhoV2in`: Hybridization values evaluated on `ozin`.
- `Lambda`: Logarithmic discretization parameter. Typical NRG applications use
  `Lambda > 1`.
- `N`: Requested Wilson-chain length.

Keyword arguments
- `estep`: Number of grid points used to resolve each logarithmic interval.
  This matches the MATLAB option `"estep"` and defaults to `10`.
- `emax`: Largest positive energy used for the logarithmic grid. The default is
  `maximum(abs, ozin)`.
- `emin`: Smallest positive energy used for the logarithmic grid. The default
  is `100eps(Float64)`, mirroring the MATLAB code's `100*eps`.

Returns
- `ff`: Wilson-chain hopping amplitudes obtained from the Lanczos
  tridiagonalization of the discretized star Hamiltonian.
- `gg`: Wilson-chain on-site energies.

Notes
- Negative values produced by interpolation are clipped to zero, following the
  MATLAB implementation and ensuring a positive hybridization density.
- The routine preserves the MATLAB indexing convention exactly: `ff` and `gg`
  both have length `min(N, nintervals)`, and the final entry of `gg` remains
  zero because the original code only writes `gg[it]` when the next Lanczos
  vector exists.
- If the number of discretization intervals is smaller than `N`, the returned
  chain is truncated and a warning is emitted.

This function is a faithful Julia port of `doCLD (1).m`, including the helper
interval formulas and the final Lanczos step used to build the Wilson chain.
"""
function doCLD(ozin::AbstractVector, rhoV2in::AbstractVector,
               Lambda::Real, N::Integer;
               estep::Integer = 10,
               emax = maximum(abs, ozin),
               emin = 100eps(Float64))
    isempty(ozin) && throw(ArgumentError("Empty frequency input `ozin`."))
    isempty(rhoV2in) && throw(ArgumentError("Empty hybridization input `rhoV2in`."))
    length(ozin) == length(rhoV2in) ||
        throw(ArgumentError("`ozin` and `rhoV2in` must have the same length."))
    length(ozin) ≥ 2 ||
        throw(ArgumentError("At least two bath samples are required."))
    estep > 0 || throw(ArgumentError("`estep` must be positive."))
    N > 0 || throw(ArgumentError("`N` must be positive."))
    Lambda > 1 || throw(ArgumentError("`Lambda` must be larger than 1."))
    emax > 0 || throw(ArgumentError("`emax` must be positive."))
    emin > 0 || throw(ArgumentError("`emin` must be positive."))
    emin < emax || throw(ArgumentError("`emin` must be smaller than `emax`."))

    T = promote_type(float(eltype(ozin)), float(eltype(rhoV2in)),
                     typeof(float(Lambda)), typeof(float(emax)), typeof(float(emin)))
    ozinp = T.(ozin)
    rhoinp = T.(rhoV2in)

    perm = sortperm(ozinp)
    ozsorted = ozinp[perm]
    rhosorted = rhoinp[perm]
    all(diff(ozsorted) .> zero(T)) ||
        throw(ArgumentError("`ozin` must contain strictly increasing frequencies after sorting."))

    xs_hi = log(T(emax)) / log(T(Lambda)) * estep
    xs_lo = log(T(emin)) / log(T(Lambda)) * estep
    xs = reverse(collect(xs_hi:-1:xs_lo)) ./ estep
    oz = T(Lambda) .^ xs

    rho1 = _linear_interp_extrap(ozsorted, rhosorted, oz)
    rho1 .= max.(rho1, zero(T))
    repE1, repT1 = doCLD_1side(oz, rho1, estep)

    rho2 = _linear_interp_extrap(ozsorted, rhosorted, .-oz)
    rho2 .= max.(rho2, zero(T))
    repE2, repT2 = doCLD_1side(oz, rho2, estep)

    nintervals = length(repE1) + length(repE2)
    N2 = if nintervals < N
        @warn "Number of discretization intervals is smaller than the requested chain length." nintervals N
        nintervals
    else
        N
    end

    ff = zeros(T, N2)
    gg = zeros(T, N2)

    Xis = vcat(repE1, .-repE2)
    Gammas = vcat(sqrt.(repT1), sqrt.(repT2))
    H = zeros(T, length(Xis) + 1, length(Xis) + 1)
    H[1, 2:end] .= Gammas
    H[2:end, 1] .= Gammas
    H[2:end, 2:end] .= Diagonal(Xis)

    U = zeros(T, size(H, 1), N2)
    U[1, 1] = one(T)

    for itN in 1:N2
        v = H * view(U, :, itN)
        v .-= U * (U' * v)
        v .-= U * (U' * v)  # repeat for numerical stability, as in MATLAB
        ff[itN] = LinearAlgebra.norm(v)

        if itN < N2 && ff[itN] > zero(T)
            U[:, itN + 1] .= v ./ ff[itN]
            gg[itN] = dot(view(U, :, itN + 1), H * view(U, :, itN + 1))
        end
    end

    return ff, gg
end

function NRG_IterDiag(H0::QSpace{T, 2, N}, 
    A0::QSpace{T, 3, N}, 
    λ::Float64, 
    ff::Vector{Float64}, 
    q0F::QSpace{T, 3, N}, 
    gg::Vector{Float64}, 
    n0::QSpace{T, 2, N}, 
    q0Z::QSpace{T, 2, N}, 
    Nkeep::Int) where {T, N}
    # Placeholder for the actual NRG iteration and diagonalization routine.
    # This function would perform the iterative diagonalization of the Wilson
    # chain Hamiltonian constructed from H0 and A0, using the hopping amplitudes
    # ff and on-site energies gg. The local space is defined by q0F and q0Z.
    # The function would return the final effective Hamiltonian or other relevant
    # data structures after NRG iterations with a truncation to Nkeep states.
    println("NRG start")

    Nsite = length(ff) + 1
    EScale = [1, (λ.^(((Nsite-2):-1:0)./2).*ff[end])...]
    AK = Vector{QSpace{T, 3, N}}(undef, Nsite)
    AD = Vector{QSpace{T, 3, N}}(undef, Nsite)
    EK = []
    ED = []
    E0 = Vector{Float64}(undef, Nsite)

    Nfac = 0.1
    Hprev = H0
    Fprev = q0F
    for itN=1:Nsite
        si = itN - 1
        Z = QSpace(q0Z, ("s,$si", "s,$si"))
        n0 = QSpace(n0, ("s,$si", "s,$si"))
        F = QSpace(q0F, ("s,$si", "s,$si", "op"))
        if itN == 1
            AK[itN] = A0
            # Do not update AD since there is no discarded states
            e = eigQS(H0)
            ek, ed = discard_eigQS(e, Nkeep, "K,$si", "D,$si")
            push!(EK, ek.eig_list)
            push!(ED, ed.eig_list)

        else
            l = findleg(Hprev; dir='-')
            Anow = getIdentity((Hprev, l), (Z, 2); itags="L,$si")
            
            Hnow = lock(Anow'; itags="L") * (Hprev * Anow)
            Hnow = Hnow * (EScale[itN-1] / EScale[itN])

            Fnow = F' * lock(Z, 2)
            Hhop = lock(Anow'; itags="L") * (Fprev * Anow * Fnow)
            Hhop = Hhop * (ff[itN-1] / EScale[itN])
            Hhop = Hhop + Hhop'

            Hon = lock(Anow'; itags="L") * (n0 * Anow)
            Hon = (gg[itN-1] / EScale[itN]) * Hon

            Hnow = Hnow + Hhop + Hon
            Hnow = (Hnow + Hnow') / 2 # Ensure Hermiticity

            # Eigendecomposition and discard spaces
            e = eigQS(Hnow)
            e0 = e.eig_list[1][1] # Ground state energy
            E0[itN] = e0

            ek, ed = discard_eigQS(e, Nkeep, "K,$si", "D,$si")

            # Shift energies to make the lowest energy value be 0
            Hprev = ek.D - e0
            klst = [(x-e0, k...) for (x, k...) in ek.eig_list]
            dlst = [(x-e0, d...) for (x, d...) in ed.eig_list]
            push!(EK, klst); push!(ED, dlst)
            AK[itN] = Anow * ek.V; AD[itN] = Anow * ed.V
        end

        if itN < Nsite
            ak = AK[itN]
            Fprev = ak' * lock(F * ak; itags="K,$si")
        end
    end


    return EK, AK, ED, AD, E0  
end

function NRG_test()
    U = 4e-3 # Coulomb interaction at the impurity
    epsd = -U/2 #  impurity on-site energy
    Γ = 8e-5*pi # hybridization strength

    λ = 2.5 # logarithmic discretization parameter
    N = 55 # Wilson chain length
    Nkeep = 300

    ff, gg = doCLD([-1, 1], [1, 1].*(Γ/pi), λ, N)

    option = FermionSOptions(U1, SU{2}, nothing, 1)
    q0 = getLocalSpace(option, ("s,0", "s,0", "op"))
    n0 = lock(q0.F', 2) * q0.F

    H0 = U/2*lock(n0, 1) * (n0 - q0.I) + epsd * n0 
    v = getvac(q0.I, ("K,vac", "K,vac"))
    A0 = permuteQS(getIdentity((v, 2), (H0, 2); itags="K,0"), (1, 3, 2))

    H0 = A0' * lock(A0 * H0, 2)

    return NRG_IterDiag(H0, A0, λ, ff, q0.F, gg, n0, q0.Z, Nkeep)
end

_nrg_sector_label(sector_qlabels) =
    join(["(" * join(ql, ",") * ")" for ql in sector_qlabels], " ")

function _nrg_plot_series(EK::AbstractVector, parity::Int, ymax::Real)
    series = Dict{Tuple{Any, Int}, Vector{Tuple{Int, Float64}}}()

    for (it, eig_entries) in enumerate(EK)
        isodd(it) == (parity == 1) || continue
        for (ev, _, sector_qlabels, sector_index) in eig_entries
            energy = Float64(real(ev))
            energy < ymax || continue

            key = (sector_qlabels, sector_index)
            push!(get!(series, key, Tuple{Int, Float64}[]), (it, energy))
        end
    end

    return series
end

function _nrg_sector_colors(EK::AbstractVector, ymax::Real)
    sector_order = Any[]
    seen_sectors = Set{Any}()

    for eig_entries in EK
        for (ev, _, sector_qlabels, _) in eig_entries
            Float64(real(ev)) < ymax || continue
            if sector_qlabels ∉ seen_sectors
                push!(sector_order, sector_qlabels)
                push!(seen_sectors, sector_qlabels)
            end
        end
    end

    palette = Plots.palette(:tab20, max(length(sector_order), 1))
    return Dict(sector => palette[i] for (i, sector) in enumerate(sector_order))
end

function _plot_nrg_parity(EK::AbstractVector, parity::Int;
                          sector_colors,
                          ymax::Real,
                          linewidth::Real,
                          show_legend::Bool)
    parity_name = parity == 1 ? "Odd iterations" : "Even iterations"
    series = _nrg_plot_series(EK, parity, ymax)

    p = Plots.plot(
        xlabel = "Iteration",
        ylabel = "Eigenvalue",
        title = parity_name,
        ylim = (0, ymax),
        legend = show_legend ? :outerright : false,
    )

    shown_labels = Set{Any}()
    for key in sort!(collect(keys(series)); by = x -> (_nrg_sector_label(x[1]), x[2]))
        points = sort!(series[key]; by = first)
        sector_qlabels, sector_index = key
        label = if show_legend && sector_qlabels ∉ shown_labels
            push!(shown_labels, sector_qlabels)
            _nrg_sector_label(sector_qlabels)
        else
            ""
        end

        Plots.plot!(
            p,
            first.(points),
            last.(points);
            color = sector_colors[sector_qlabels],
            linewidth = linewidth,
            label = label,
        )
    end

    return p
end

"""
    plot_NRG_EK(EK; ymax=3.0, linewidth=1.5, show_legend=true)
    plot_NRG_EK(result::Tuple; ymax=3.0, linewidth=1.5, show_legend=true)

Plot the NRG eigenvalue flow stored in `EK`, splitting odd and even iterations
into separate subplots. Only eigenvalues strictly smaller than `ymax` are
drawn, and all lines with the same symmetry-sector qlabels share the same
color.

This helper uses `Plots.jl` as an optional dependency. If `Plots` is not
installed in the active Julia environment, the function throws an error with a
short installation hint.
"""
function plot_NRG_EK(EK::AbstractVector;
                     ymax::Real = 3.0,
                     linewidth::Real = 1.5,
                     show_legend::Bool = true)
    Base.find_package("Plots") === nothing &&
        throw(ArgumentError("`plot_NRG_EK` requires Plots.jl. Install it with `import Pkg; Pkg.add(\"Plots\")`."))
    @eval import Plots

    sector_colors = _nrg_sector_colors(EK, ymax)
    odd_plot = _plot_nrg_parity(EK, 1;
        sector_colors,
        ymax, linewidth, show_legend)
    even_plot = _plot_nrg_parity(EK, 2;
        sector_colors,
        ymax, linewidth, show_legend)

    fig = Plots.plot(odd_plot, even_plot; layout = (2, 1), size = (950, 900))
    display(fig)
    return fig
end

plot_NRG_EK(result::Tuple; kwargs...) = plot_NRG_EK(result[1]; kwargs...)

# ─── _leg_kron ────────────────────────────────────────────────────────────────
# Kronecker product of two tensors, taken leg-by-leg.
#   A: (d_1a, ..., d_QDa, Ma)
#   B: (d_1b, ..., d_QDb, Mb)
#   Result: (d_1a*d_1b, ..., d_QDa*d_QDb, Ma*Mb)
# The last axis in each operand is treated as a "bond/weight" index and is
# also Kronecker-products so the result carries the full combined bond.
function _leg_kron(A::AbstractArray, B::AbstractArray, QD::Int)
    da = ntuple(l -> size(A, l), QD)
    db = ntuple(l -> size(B, l), QD)
    Ma, Mb = size(A, QD+1), size(B, QD+1)

    # Expand A: insert QD singleton dims between physical and bond, plus a 1 at end
    #   (da..., 1..._QD, Ma, 1)
    Ar = reshape(A, da..., ones(Int, QD)..., Ma, 1)
    # Expand B: insert QD singleton dims at the front, plus a 1 before Mb
    #   (1..._QD, db..., 1, Mb)
    Br = reshape(B, ones(Int, QD)..., db..., 1, Mb)

    # Broadcast multiply → (da..., db..., Ma, Mb)
    outer = Ar .* Br

    # Permute so matching legs are adjacent: (d1a,d1b, d2a,d2b, ..., Ma,Mb)
    perm = vcat([[l, QD+l] for l in 1:QD]..., 2QD+1, 2QD+2)
    outer_p = permutedims(outer, perm)

    # Reshape interleaved pairs into single legs
    return reshape(outer_p, ntuple(l -> da[l]*db[l], QD)..., Ma*Mb)
end

# ─── _legwise_kron ───────────────────────────────────────────────────────────
# Leg-wise Kronecker product of two QD-dimensional tensors (no bond leg).
#   A: (d_1, ..., d_QD),  B: (s_1, ..., s_QD)
#   Result: (d_1*s_1, ..., d_QD*s_QD)
# Convention: kron(A_l, B_l) per leg — A (CGT) is the slow index, B (RMT) fast.
function _legwise_kron(A::AbstractArray, B::AbstractArray)
    QD = ndims(A)
    da = ntuple(l -> size(A, l), QD)
    db = ntuple(l -> size(B, l), QD)
    # Expand A → (d1,1,d2,1,...), B → (1,s1,1,s2,...)
    Ar = reshape(A, Tuple(Iterators.flatten((da[l], 1) for l in 1:QD)))
    Br = reshape(B, Tuple(Iterators.flatten((1, db[l]) for l in 1:QD)))
    outer = Ar .* Br   # shape (d1, s1, d2, s2, ..., dQD, sQD)
    # Keep pair order (dl, sl) so that dl (A) is fast in the reshape.
    # Combined index per leg = (sl-1)*dl + rl  ↔  kron(B_l, A_l) w/ A fast
    perm = Tuple(Iterators.flatten((2l-1, 2l) for l in 1:QD))
    outer_p = permutedims(outer, perm)  # (d1, s1, d2, s2, ...)
    return reshape(outer_p, ntuple(l -> da[l]*db[l], QD)...)
end

# ─── contract_sparse ─────────────────────────────────────────────────────────
# Arbitrary contraction of two arrays over matching leg pairs.
# Remaining legs keep their original relative order:
#   result legs = (free legs of A in original order, free legs of B in original order)
#
#   A     : any AbstractArray  (works with SparseArray, plain Array, …)
#   B     : any AbstractArray
#   legs_A: legs of A to contract (1-based, length M)
#   legs_B: corresponding legs of B (must match sizes pairwise)
function contract_sparse(A::AbstractArray, B::AbstractArray,
                         legs_A::NTuple{M, Int},
                         legs_B::NTuple{M, Int}) where M
    keep_A = [i for i in 1:ndims(A) if !(i in legs_A)]
    keep_B = [i for i in 1:ndims(B) if !(i in legs_B)]

    @assert all(size(A, legs_A[k]) == size(B, legs_B[k]) for k in 1:M) "contracted leg size mismatch"

    # Permute A → (keep_A..., legs_A...)  and reshape to matrix (free × contr)
    Ap = permutedims(A, [keep_A; collect(legs_A)])
    sz_fA = [size(A, i) for i in keep_A]
    Amat  = reshape(Ap, prod(sz_fA; init=1), :)

    # Permute B → (legs_B..., keep_B...)  and reshape to matrix (contr × free)
    Bp = permutedims(B, [collect(legs_B); keep_B])
    sz_fB = [size(B, i) for i in keep_B]
    Bmat  = reshape(Bp, :, prod(sz_fB; init=1))

    # Contract via matrix multiply, then restore leg shapes
    return reshape(Amat * Bmat, sz_fA..., sz_fB...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Compute offset map from a splist (space list) for a given symmetry tuple.
# Returns (leg_offset::Dict{qlabel => UnitRange}, leg_total::Int)
function get_offset(symm::Tuple, splist::Vector)
    N = length(symm)
    leg_offset = Dict{Any, UnitRange{Int}}()
    sector_size = Dict{Any, Int}()
    
    for (RMTd, qlabels) in splist
        irep_dim = prod(
            isabelian(symm[n]) ? 1 :
            dimension(getNsave_irep(symm[n], BigInt, qlabels[n]))
            for n in 1:N)
        sz = irep_dim * RMTd
        sector_size[qlabels] = sz
    end

    # Assign offsets in sorted qlabel order.
    off = 1
    for ql in sort(collect(keys(sector_size)))
        sz = sector_size[ql]
        leg_offset[ql] = off:off+sz-1
        off += sz
    end
    leg_total = off - 1
    return (leg_offset, leg_total)
end


# ─── to_sparse_array ───────────────────────────────────────────────────────────
# Convert a QSpace to a sparse array using its spaces field for offset computation.
function to_sparse_array(q::QSpace{T, QD, N, RD},
    ::Type{FT} = Float64) where {T, QD, N, RD, FT}

    symm = q.symm
    rows = q.rows

    # ── Step 1: offset map ──────────────────────────────────────────────────
    # Use spaces field directly for each leg's offset computation.
    leg_info = [get_offset(symm, q.spaces[l]) for l in 1:QD]
    leg_offsets = [li[1] for li in leg_info]
    leg_total = [li[2] for li in leg_info]

    # ── Step 2: allocate output array ───────────────────────────────────────
    result = SparseArray(zeros(FT, leg_total...))

    # ── Step 3: accumulate each row's contribution ───────────────────────────
    for r in rows
        # For each symmetry n, build the CGT contracted with its w-matrix.
        # After contracting, cgt_wmats[n] has shape
        #   (d_phys_1^(n), ..., d_phys_QD^(n), M_n)
        # where d_phys_l^(n) = irrep dim at physical leg l for symmetry n,
        # and M_n = size(wmat, 2) (compressed bond dimension after SVD).
        cgt_wmats = Vector{Array{FT}}(undef, N)
        for n in 1:N
            S   = symm[n]
            cgr = r.cgrs[n]
            M   = size(cgr.wmat.data, 2)

            if isabelian(S)
                # Abelian symmetry: irrep dim = 1 at every leg, outer multiplicity = 1.
                # CGT is trivially the scalar 1; the contracted result is just the
                # w-matrix row reshaped to (1,...,1, M).
                @assert M == 1 "Unexpected bond dimension for abelian symmetry $n"
                cgt_wmats[n] = reshape(FT.(cgr.wmat.data), ones(Int, QD)..., M)
            else
                # a. Extract CGT qlabels from this CGR.
                nin, nout = cgr.legdir
                insp  = Tuple(cgr.qlabels[i] for i in 1:nin)
                outsp = Tuple(cgr.qlabels[i] for i in nin+1:QD)
                insp_, _ = remove_zeros(S, insp)
                outsp_, _ = remove_zeros(S, outsp)

                CGTom = get_CGTom(S, insp_, outsp_)
                om    = CGTom.totalOM
                @assert om == size(cgr.wmat.data, 1) "outer multiplicity mismatch for symmetry $n"

                canbasis = get_canonical_basis(S, insp, outsp, CGTom)

                # b. Allocate (CGT_shape..., om).
                cgt_shape = size(Array(canbasis[1]))   # (d_in_1,...,d_out_NO) in canonical order
                cgt_arr   = zeros(FT, cgt_shape..., om)

                # c. Fill each om-slice with the corresponding canonical basis element.
                for i in 1:om cgt_arr[fill(:, QD)..., i] .= Array(canbasis[i]) end

                # d. Contract last (om) leg of cgt_arr with first (om) leg of wmat.
                #    cgt_arr: (...cgt_shape..., om)  →  flatten to (prod_shape, om)
                #    wmat:    (om, M)
                #    result:  (prod_shape, M)  →  reshaped to (cgt_shape..., M)
                wmat_data   = cgr.wmat.data                        # (om, M)
                cgt_flat    = reshape(cgt_arr, :, om)              # (prod(cgt_shape), om)
                result_flat = cgt_flat * wmat_data                 # (prod(cgt_shape), M)
                cgt_wmat_canon = reshape(result_flat, cgt_shape..., M)

                # Permute from canonical leg order to physical leg order using cgp.
                # cgr.cgp[l] = canonical axis that corresponds to physical leg l,
                # so permutedims with perm = (cgp..., QD+1) maps canonical → physical.
                perm = (cgr.cgp..., QD+1)
                cgt_wmats[n] = permutedims(cgt_wmat_canon, perm)
            end
        end

        # e. Kronecker product over symmetries to get the combined CGT block.
        #    cgt_wmats[n]: (d_1^(n), ..., d_QD^(n), M_n)
        #    cgt_block:    (irep_dim_1, ..., irep_dim_QD, M_total)
        cgt_block = cgt_wmats[1]
        for n in 2:N
            cgt_block = _leg_kron(cgt_block, cgt_wmats[n], QD)
        end
        # cgt_block has shape (irep_dim_1, ..., irep_dim_QD, M_total)
        # where irep_dim_l = prod_n d_l^(n)  and  M_total = prod_n M_n.
        chi = size(cgt_block, QD + 1)   # = M_total

        # ── Step 3a: Merge all OM (non-physical) legs of RMT ─────────────────
        # RMT shape: (s_1, ..., s_QD, M_1, ..., M_N)  →  (s_1, ..., s_QD, chi)
        # Julia column-major reshape: M_1 varies fastest, consistent with how
        # _leg_kron built up the bond dimension in cgt_block.
        rmt_merged = reshape(r.RMT.data, size(r.RMT.data)[1:QD]..., chi)

        # ── Step 3b: Σ_i kron(CGT[:,...,:,i], RMT[:,...,:,i]) ────────────────
        d_phys = ntuple(l -> size(cgt_block,  l), QD)   # irep dims per leg
        s_phys = ntuple(l -> size(rmt_merged, l), QD)   # RMT free dims per leg
        block  = zeros(FT, ntuple(l -> d_phys[l] * s_phys[l], QD))
        col    = fill(:, QD)   # helper for slicing all physical legs
        for i in 1:chi
            block .+= _legwise_kron(cgt_block[col..., i], rmt_merged[col..., i])
        end

        # ── Step 3c: Scatter block into result at the correct qlabel offsets ─
        ranges = Tuple(leg_offsets[l][_row_qlabel(r, l)] for l in 1:QD)
        result[ranges...] .+= block
    end

    return result
end

# ─── test_compress_sector ────────────────────────────────────────────────────
# Invariant: the effective tensor is preserved by the SVD compression (tol=0).
#   Direct:        D[..., om3_1,...,om3_N] = Σ_p Σ_{om12} (⊗_n W[n][p]) * RMT[p][..., om12]
#   Reconstructed: contract result_RMT with each U[n] along axis QD_out+n
#
# Parameters:
#   N        : number of symmetries
#   K        : number of matched pairs (= length of new_wmats[n] and new_RMTs)
#   QD_out   : number of free (non-OM) axes in each RMT
#   free_sizes : sizes of the QD_out free axes  (default: random 2–5)
#   OM3_sizes  : output OM size per symmetry    (default: random 1–4)
#   om12_sizes : input OM size [n, p]           (default: random 1–4)
function test_compress_sector(N::Int = 2, K::Int = 2, QD_out::Int = 2;
                               seed::Int       = 420,
                               free_sizes      = rand(2:5, QD_out),
                               OM3_sizes       = rand(1:4, N),
                               om12_sizes      = rand(1:4, N, K),
                               verbose::Bool   = false)
    Random.seed!(seed)

    # W[n][p] : (OM3_sizes[n], om12_sizes[n,p])
    W    = [[randn(OM3_sizes[n], om12_sizes[n, p]) for p in 1:K] for n in 1:N]
    # RMT[p] : (free_sizes..., om12_sizes[1,p], ..., om12_sizes[N,p])
    RMTs = [randn(free_sizes..., [om12_sizes[n, p] for n in 1:N]...) for p in 1:K]

    if verbose
        println("  Parameters: N=$N, K=$K, QD_out=$QD_out")
        println("  free_sizes  = $free_sizes")
        println("  OM3_sizes   = $OM3_sizes")
        println("  om12_sizes  = $om12_sizes  (rows=symmetry, cols=pair)")
        for n in 1:N, p in 1:K
            println("  W[$n][$p]   size = $(size(W[n][p]))")
        end
        for p in 1:K
            println("  RMTs[$p]    size = $(size(RMTs[p]))")
        end
    end

    new_wmats = Tuple(QTensor{Float64, 2}[QTensor(W[n][p]) for p in 1:K]
                      for n in 1:N)
    new_RMTs  = [QTensor(RMTs[p]) for p in 1:K]

    U_mats, result_RMT = _compress_sector(new_wmats, new_RMTs, QD_out, 0.0)

    if verbose
        for n in 1:N
            println("  U_mats[$n]  size = $(size(U_mats[n]))")
        end
        println("  result_RMT  size = $(size(result_RMT))")
    end

    # Reconstruct: contract result_RMT with U[n] along axis QD_out+n for each n.
    reconstructed = result_RMT.data
    for n in 1:N
        reconstructed = _contract_om_axis(reconstructed, U_mats[n].data, QD_out + n)
    end
    # shape: (free_sizes..., OM3_sizes...)

    # Direct: for each pair p, contract W[n][p] along axis QD_out+n, then sum over p.
    direct = zeros(free_sizes..., OM3_sizes...)
    for p in 1:K
        contrib = RMTs[p]
        for n in 1:N
            contrib = _contract_om_axis(contrib, W[n][p], QD_out + n)
        end
        direct .+= contrib
    end

    if verbose
        println("  reconstructed size = $(size(reconstructed))")
        println("  direct        size = $(size(direct))")
    end

    max_diff = maximum(abs, reconstructed .- direct)
    @assert max_diff < 1e-10 "test_compress_sector FAILED: max diff = $(max_diff)"
    println("test_compress_sector passed (N=$N, K=$K, QD_out=$QD_out).")
end

function test_FAcont(option::LocalSpaceOptions)
    q = getLocalSpace(option);
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    qf = QSpace(q.F, ("lur2", "lur2", "op"))
    a = getIdentity((qi1, 2), (qi2, 2));

    ct = qf * a
    Farr = to_sparse_array(q.F)
    Aarr = to_sparse_array(a)

    ctarr1 = contract_sparse(Farr, Aarr, (2,), (2,))
    ctarr2 = to_sparse_array(ct)

    println(norm(ctarr1 - ctarr2))
    @test norm(ctarr1 - ctarr2) < 1e-10

    return q, a, ct, ctarr1, ctarr2
end

function test_1jpair(option::LocalSpaceOptions)
    q = getLocalSpace(option);
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))

    q1, q2 = get1jpair(qi1, 2)
    mult = contract(q1, 2, q2, 1)
    arr1 = to_sparse_array(q.I)
    arr2 = to_sparse_array(mult)

    println(norm(arr1 - arr2))
    @test norm(arr1 - arr2) < 1e-10

    a = getIdentity((qi1, 2), (qi2, 2))
    q1, q2 = get1jpair(a, 3)
    mult = contract(q1, 2, q2, 1)

    arr1 = to_sparse_array(mult)
    sz = size(arr1)[1]
    arr2 = SparseArray(Matrix(I, sz, sz))

    println(norm(arr1 - arr2))
    @test norm(arr1 - arr2) < 1e-10

end

# ─── test_conj ───────────────────────────────────────────────────────────────
# Invariant: conj(q) represented as a dense/sparse array equals the
# elementwise complex conjugate of q's sparse array.
#
# Tested QSpace objects (drawn from existing test helpers):
#   1. q.I       — 2-leg identity operator
#   2. q.F       — 3-leg creation/annihilation operator
#   3. a         — 4-leg identity from getIdentity (tensor product of two I legs)
#   4. ct        — 4-leg contraction result F ⊗ a from test_FAcont
#
# For each, we verify:
#   norm( to_sparse_array(qs)  −  conj(to_sparse_array(conj(qs))) )  < tol
#
# Physical qlabels do not change under conjugation (only stored CGR ordering
# flips), so both sparse arrays have the same shape and qlabel offsets.
# ─────────────────────────────────────────────────────────────────────────────
function test_conj(option::LocalSpaceOptions)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2))
    qf  = QSpace(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a

    # Pairs (label, QSpace) to test
    cases = [
        ("I",  q.I),
        ("F",  q.F),
        ("a",  a),
        ("ct", ct),
    ]

    for (label, qs) in cases
        qc   = conj(qs)

        # Convert both to plain dense arrays for comparison.
        arr  = Array(to_sparse_array(qs))
        arrc = Array(to_sparse_array(qc))

        # conj(q) as a sparse array should equal conj(q as sparse array).
        diff = norm(arr .- conj.(arrc))
        println("test_conj [$label]: ‖arr − conj(arrc)‖ = $diff")
        @test diff < 1e-10
    end
end

# ─── test_svdQS ────────────────────────────────────────────────────────────────
# Test SVD by performing decomposition and reconstructing the original QSpace.
#
# Arguments:
#   q         : QSpace to decompose
#   left_legs : legs to group on the "U" side (same as svdQS)
#   cutoff    : singular value cutoff (default 1e-12)
#   tol       : tolerance for reconstruction error (default 1e-9)
#
# Returns: (diff_norm, max_rmt_diff) where
#   diff_norm     : norm of (original - reconstructed) as sparse array
#   max_rmt_diff  : maximum per-RMT norm of difference (after matching rows)
#
# Algorithm:
#   1. Perform svdQS(q, left_legs; cutoff)
#   2. Contract U * S to get US
#   3. Contract US * Vd to get reconstructed QSpace
#   4. Permute reconstructed to match original leg order
#   5. Convert both to sparse arrays and compute norm of difference
# ─────────────────────────────────────────────────────────────────────────────
function test_svdQS(q::QSpace{T, QD, N, RD},
                    left_legs;
                    cutoff::Float64 = 1e-12,
                    tol::Float64 = 1e-9,
                    verbose::Bool = true) where {T, QD, N, RD}
    
    left_legs = collect(Int, left_legs)
    right_legs = [l for l in 1:QD if l ∉ left_legs]
    NL, NR = length(left_legs), length(right_legs)
    
    # Step 1: Perform SVD
    U, S, Vd = svdQS(q, left_legs; cutoff=cutoff)
    
    if verbose
        println("SVD completed:")
        println("  U  : $(length(U.rows)) rows, $(NL+1) legs")
        println("  S  : $(length(S.rows)) rows, 2 legs")
        println("  Vd : $(length(Vd.rows)) rows, $(NR+1) legs")
    end
    
    # Step 2: Contract U * S
    # U has legs (left_legs..., bond_left'+'), bond leg is last with direction '+'
    # S has legs (left_tag'-', right_tag'-'), both directions '-'
    # U's bond leg (+) should contract with S's first leg (-)
    US = U * S
    
    if verbose
        println("  US : $(length(US.rows)) rows after U*S")
    end
    
    # Step 3: Contract US * Vd
    # US has legs (left_legs..., right_tag'-')
    # Vd has legs (bond'+', right_legs...)
    # US's last leg (-) should contract with Vd's first leg (+)
    rec = US * Vd
    
    if verbose
        println("  rec: $(length(rec.rows)) rows after (U*S)*Vd")
    end
    
    # Step 4: Permute reconstructed to match original leg order
    # After contraction, rec has legs in order (left_legs..., right_legs...)
    # We need to permute back to original order (1, 2, ..., QD)
    # Build inverse permutation: orig_order[i] tells where leg i should go
    combined_order = vcat(left_legs, right_legs)
    inv_perm = zeros(Int, QD)
    for (new_pos, orig_leg) in enumerate(combined_order)
        inv_perm[orig_leg] = new_pos
    end
    # Now inv_perm[orig_leg] = current_pos, so we need perm where perm[final_pos] = current_pos
    # final_pos = orig_leg, so perm[orig_leg] = inv_perm[orig_leg] is wrong
    # Actually: combined_order[new_pos] = orig_leg means leg orig_leg is at new_pos
    # We want permutation p such that after permute, leg in position i came from combined_order[p[i]]
    # i.e., we want result[i] = orig[p[i]] where orig has legs in combined_order
    # Since combined_order[new_pos] = orig_leg, we want p[orig_leg] = new_pos
    # That's what inv_perm computes: inv_perm[orig_leg] = new_pos
    # So we apply permutation inv_perm to get legs in order (1, 2, ..., QD)
    
    # Wait, let me think again:
    # rec has legs ordered as combined_order = (left_legs..., right_legs...)
    # rec.legs[new_pos] corresponds to original leg combined_order[new_pos]
    # We want final.legs[orig_leg] = rec.legs[pos_in_rec_of_orig_leg]
    # pos_in_rec_of_orig_leg = inv_perm[orig_leg] where inv_perm[combined_order[i]] = i
    # So perm for permuteQS: perm[final_pos] = old_pos means final.leg[final_pos] = rec.leg[old_pos]
    # We want final.leg[orig_leg] = rec.leg[inv_perm[orig_leg]]
    # So perm[orig_leg] = inv_perm[orig_leg]? No, that's circular.
    # 
    # Let's be concrete: if left_legs = [1,3], right_legs = [2,4]
    # combined_order = [1,3,2,4]
    # rec has legs: (orig_leg1, orig_leg3, orig_leg2, orig_leg4)
    # We want: (orig_leg1, orig_leg2, orig_leg3, orig_leg4)
    # So perm = [1, 3, 2, 4] meaning final[1]=rec[1], final[2]=rec[3], final[3]=rec[2], final[4]=rec[4]
    # In general: perm[orig_leg] = position_of_orig_leg_in_combined_order
    # inv_perm[combined_order[i]] = i, so inv_perm[orig_leg] = position_of_orig_leg_in_combined_order
    # So perm = inv_perm, and perm[orig_leg] = inv_perm[orig_leg]
    
    # Actually, permuteQS semantics: perm[new_pos] = old_pos
    # result.leg[new_pos] = input.leg[perm[new_pos]]
    # We want result.leg[orig_leg] = rec.leg[inv_perm[orig_leg]]
    # So perm[orig_leg] = inv_perm[orig_leg], i.e., perm = inv_perm
    
    perm = Tuple(inv_perm)
    rec_permuted = permuteQS(rec, perm)
    
    if verbose
        println("  rec_permuted: legs permuted to original order")
    end
    
    # Step 5: Convert to sparse arrays and compare
    arr_orig = Array(to_sparse_array(q))
    arr_rec  = Array(to_sparse_array(rec_permuted))
    
    diff_arr = arr_orig .- arr_rec
    diff_norm = norm(diff_arr)
    max_diff = maximum(abs, diff_arr)
    
    if verbose
        println("Reconstruction error:")
        println("  ‖orig - rec‖₂ = $diff_norm")
        println("  max|orig - rec| = $max_diff")
    end
    
    @assert diff_norm < tol "SVD reconstruction error ‖orig - rec‖ = $diff_norm exceeds tolerance $tol"
    
    return diff_norm, max_diff
end

# ─── test_norm ───────────────────────────────────────────────────────────────
# Verify norm(q) against two independent reference values:
#
#   (a) norm(to_sparse_array(q))  – dense/sparse Frobenius norm
#   (b) sqrt(abs((q * q')[]))    – scalar obtained by contracting q with conj(q)
#                                   over all legs (Wigner-Eckart inner product)
#
# Cases tested per call:
#   q.I  – rank-2 identity (QD = 2, exercises the dim·‖RMT‖² formula)
#   q.F  – rank-3 operator (QD = 3, standard CGT-orthonormal formula)
#   ct   – rank-4 tensor   (QD = 4, after F * identity contraction)
#
# For (b) to work every pair (q leg l, conj(q) leg l) must form a valid
# contraction pair: same itags, opposite directions.  conj() flips all
# directions, so the pairing is always leg-for-leg given that all legs of q
# already carry distinct QIndex values (full-rank contraction → 0D scalar).
# ─────────────────────────────────────────────────────────────────────────────
function test_norm(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itags="lurlur")
    qf  = QSpace(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a   # rank-4: legs (lur1_in, lur2_in, lur2_out, op)

    cases = [
        ("I",  q.I),
        ("F",  q.F),
        ("ct", ct),
    ]

    for (label, qs) in cases
        # ── (a) compare with sparse array norm ─────────────────────────────
        norm_qs     = norm(qs)
        norm_sparse = norm(Array(to_sparse_array(qs)))
        diff_a = abs(norm_qs - norm_sparse)
        println("test_norm [$label]: norm_qs=$norm_qs  norm_sparse=$norm_sparse  Δ=$diff_a")
        @test diff_a < tol

        # ── (b) compare norm² with scalar from qs * qs' ────────────────────
        #   qs' = conj(qs) flips all leg directions; contraction over all
        #   matching legs yields a 0D QSpace whose single element is ‖qs‖².
        scalar_qs     = qs * qs'
        println(qs)
        norm_sq_contr = abs(scalar_qs[])
        diff_b = abs(norm_qs^2 - norm_sq_contr)
        ref    = max(norm_sq_contr, 1.0)
        println("test_norm [$label]: ‖qs‖²=$(norm_qs^2)  (qs·qs')=$norm_sq_contr  Δ=$diff_b  (rel=$(diff_b/ref))")
        @test diff_b < tol * ref
    end
end

# ─── test_eigQS ──────────────────────────────────────────────────────────────
# Build a non-trivial Hermitian test input by copying the structure of q.I and
# replacing each block's RMT with a random real symmetric matrix of the same
# size.  This avoids testing only the trivial identity (eigenvalues all 1) and
# exercises the decomposition with general positive-definite blocks.
#
# Checks:
#   (a) Reconstruction  : ‖V * D * V' - A‖ / ‖A‖ < tol  (relative)
#   (b) Orthonormality  : ‖V' * V - I‖ < tol
#   (c) eig_list size   : total count == sum of RMT row block sizes
#   (d) eig_list order  : sorted ascending by eigenvalue
#   (e) eig_list space  : each entry carries its sector and in-sector index
# ─────────────────────────────────────────────────────────────────────────────
function test_eigQS(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    # Copy q.I (keeps CGRs/inds/spaces intact) then overwrite each RMT block
    # with a random real-symmetric (= Hermitian) matrix of the same block size.
    # Using A = Mᵀ * M + εI guarantees positive-definite eigenvalues so the
    # sort-descending check is unambiguous.
    A = copy(q.I)
    rng = Random.MersenneTwister(42)
    for r in A.rows
        sz = size(r.RMT.data)   # (n, n, 1, …, 1)
        n  = sz[1]
        M  = randn(rng, n, n)
        H  = M' * M + I(n) * 0.1   # symmetric, positive-definite
        r.RMT.data .= reshape(Float64.(H), sz)
    end

    result = eigQS(A; hermitian = true)
    @test isnothing(result.V_inv)
    D = result.D
    V = result.V
    eig_list = result.eig_list
    println("test_eigQS: $(length(eig_list)) eigenvalues, $(length(D.rows)) D-rows, $(length(V.rows)) V-rows")

    # ── (a) Reconstruction: V * D * V' ≈ A ──────────────────────────────────
    rec      = lock(V, 1) * (D * V')
    arr_A    = Array(to_sparse_array(A))
    arr_rec  = Array(to_sparse_array(rec))
    ref_norm = max(norm(arr_A), 1.0)
    diff_a   = norm(arr_A - arr_rec) / ref_norm
    println("  ‖V*D*V' - A‖/‖A‖ = $diff_a")
    @test diff_a < tol

    # ── (b) Orthonormality: V' * V ≈ I ──────────────────────────────────────
    VtV     = V' * lock(V, 2)
    arr_VtV = Array(to_sparse_array(VtV))
    n_bond  = size(arr_VtV, 1)
    diff_b  = norm(arr_VtV - Matrix(I, n_bond, n_bond))
    println("  ‖V'V - I‖ = $diff_b")
    @test diff_b < tol

    # ── (c) eig_list size matches total RMT dimension ────────────────────────
    total_eig_count = sum(size(r.RMT.data, 1) for r in A.rows)
    @test length(eig_list) == total_eig_count

    # ── (d) eig_list is sorted ascending ─────────────────────────────────────
    if length(eig_list) > 1
        @test all(real(eig_list[i][1]) <= real(eig_list[i+1][1]) for i in 1:length(eig_list)-1)
    end

    # ── (e) eig_list entries include sector metadata ────────────────────────
    sector_dims = Dict(
        Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(A.symm)) => size(r.RMT.data, 1)
        for r in D.rows
    )
    for entry in eig_list
        _, deg, sector, idx = entry
        @test deg >= 1
        @test idx >= 1
        if haskey(sector_dims, sector)
            @test idx <= sector_dims[sector]
        end
    end
end

# ─── test_spaces_eigQS ───────────────────────────────────────────────────────
# Verify that spaces of D and V returned by eigQS are consistent:
#
#   (a) Input assertion: q.I has equal spaces on both legs
#   (b) D.spaces[1] == D.spaces[2]  (same bond sector list on both sides)
#   (c) V.spaces[1] == V.spaces[2]
#   (d) D and V share the same bond space: D.spaces[1] == V.spaces[2]
#   (e) Bond qlabels of D (and V) are a subset of input qlabels
# ─────────────────────────────────────────────────────────────────────────────
function test_spaces_eigQS(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)
    rng = Random.MersenneTwister(0)
    for r in A.rows
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n)
        r.RMT.data .= reshape(Float64.(M' * M + I(n) * 0.1), sz)
    end

    # (a) Input spaces are equal on both legs
    @test q.I.spaces[1] == q.I.spaces[2]

    result = eigQS(A; hermitian = true)
    D = result.D
    V = result.V

    println("test_spaces_eigQS: D.spaces=$(length(D.spaces[1])) sectors, V.spaces=$(length(V.spaces[1])) sectors")

    # (b) D legs share the same space
    @test D.spaces[1] == D.spaces[2]

    # (c) V legs share the same space
    @test V.spaces[1] == V.spaces[2]

    # (d) D and V share the same bond space
    @test D.spaces[1] == V.spaces[2]

    # (e) Bond qlabels ⊆ input qlabels
    input_qls = Set(ql for (_, ql) in q.I.spaces[1])
    bond_qls  = Set(ql for (_, ql) in D.spaces[1])
    @test issubset(bond_qls, input_qls)
end

# ─── test_missing_spaces_eigQS ───────────────────────────────────────────────
# Verify that sectors present only in the space list are treated as zero blocks:
# zero eigenvalues appear in eig_list and identity rows are inserted in V.
# ─────────────────────────────────────────────────────────────────────────────
function test_missing_spaces_eigQS(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    @test !isempty(q.I.rows)

    removed_row = q.I.rows[1]
    removed_sector = Tuple(removed_row.cgrs[n].qlabels[removed_row.cgrs[n].cgp[1]] for n in 1:length(q.I.symm))
    removed_dim = size(removed_row.RMT.data, 1)

    kept_rows = copy(q.I.rows[2:end])
    A = QSpace(q.I.symm, kept_rows, q.I.inds, q.I.spaces)

    result = eigQS(A; hermitian = true)
    D = result.D
    V = result.V

    zero_entries = [entry for entry in result.eig_list if entry[3] == removed_sector]
    @test length(zero_entries) == removed_dim
    @test all(iszero(entry[1]) for entry in zero_entries)

    v_row = only([r for r in V.rows if Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(V.symm)) == removed_sector])
    v_mat = reshape(v_row.RMT.data, size(v_row.RMT.data, 1), size(v_row.RMT.data, 2))
    @test v_mat ≈ Matrix(I, removed_dim, removed_dim)

    @test any(ql == removed_sector for (_, ql) in D.spaces[1])

    VtV = V' * lock(V, 2)
    arr_VtV = Array(to_sparse_array(VtV))
    @test norm(arr_VtV - Matrix(I, size(arr_VtV, 1), size(arr_VtV, 2))) < tol
end

# ─── test_truncate_missing_zero_spaces_eigQS ────────────────────────────────
# Verify that truncation preserves sectors selected only through zero
# eigenvalues, even when the corresponding D rows are absent.
# ─────────────────────────────────────────────────────────────────────────────
function test_truncate_missing_zero_spaces_eigQS(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    @test !isempty(q.I.rows)

    removed_row = q.I.rows[1]
    removed_sector = Tuple(removed_row.cgrs[n].qlabels[removed_row.cgrs[n].cgp[1]] for n in 1:length(q.I.symm))
    removed_dim = size(removed_row.RMT.data, 1)

    kept_rows = copy(q.I.rows[2:end])
    A = QSpace(q.I.symm, kept_rows, q.I.inds, q.I.spaces)

    result = eigQS(A; hermitian = true)
    kept, discarded = discard_eigQS(result, removed_dim, "eigK", "eigD"; hermitian = true)
    eig_tag = result.D.inds[1].itags
    v_orig_leg = only(findall(i -> result.V.inds[i].itags != eig_tag, 1:2))
    v_eig_leg = only(findall(i -> result.V.inds[i].itags == eig_tag, 1:2))

    @test length(kept.eig_list) == removed_dim
    @test all(entry[3] == removed_sector for entry in kept.eig_list)
    @test all(iszero(entry[1]) for entry in kept.eig_list)

    @test isempty(kept.D.rows)
    @test kept.D.spaces[1] == [(removed_dim, removed_sector)]
    @test kept.D.spaces[2] == [(removed_dim, removed_sector)]

    @test kept.V.spaces[v_orig_leg] == result.V.spaces[v_orig_leg]
    @test discarded.V.spaces[v_orig_leg] == result.V.spaces[v_orig_leg]
    @test kept.V.spaces[v_eig_leg] == [(removed_dim, removed_sector)]
    @test all(ql != removed_sector for (_, ql) in discarded.V.spaces[v_eig_leg])

    @test any(ql == removed_sector for (_, ql) in kept.V.spaces[1])
    @test any(ql == removed_sector for (_, ql) in kept.V.spaces[2])
    @test all(ql != removed_sector for (_, ql) in discarded.D.spaces[1])

    result_full = eigQS_full(A)
    kept_full, discarded_full = discard_eigQS(result_full, removed_dim, "eigKf", "eigDf"; hermitian = false)
    eig_tag_full = result_full.D.inds[1].itags
    v_full_orig_leg = only(findall(i -> result_full.V.inds[i].itags != eig_tag_full, 1:2))
    v_full_eig_leg = only(findall(i -> result_full.V.inds[i].itags == eig_tag_full, 1:2))
    vinv_orig_leg = only(findall(i -> result_full.V_inv.inds[i].itags != eig_tag_full, 1:2))
    vinv_eig_leg = only(findall(i -> result_full.V_inv.inds[i].itags == eig_tag_full, 1:2))

    @test kept_full.V.spaces[v_full_orig_leg] == result_full.V.spaces[v_full_orig_leg]
    @test discarded_full.V.spaces[v_full_orig_leg] == result_full.V.spaces[v_full_orig_leg]
    @test kept_full.V.spaces[v_full_eig_leg] == [(removed_dim, removed_sector)]
    @test all(ql != removed_sector for (_, ql) in discarded_full.V.spaces[v_full_eig_leg])

    @test kept_full.V_inv.spaces[vinv_orig_leg] == result_full.V_inv.spaces[vinv_orig_leg]
    @test discarded_full.V_inv.spaces[vinv_orig_leg] == result_full.V_inv.spaces[vinv_orig_leg]
    @test kept_full.V_inv.spaces[vinv_eig_leg] == [(removed_dim, removed_sector)]
    @test all(ql != removed_sector for (_, ql) in discarded_full.V_inv.spaces[vinv_eig_leg])
end

# ─── test_truncate_eigQS ─────────────────────────────────────────────────────
# Verify that eigenvalue truncation keeps the Nkeep smallest eigenvalues and
# splits the eigendecomposition consistently into kept and discarded parts.
# ─────────────────────────────────────────────────────────────────────────────
function test_truncate_eigQS(option::LocalSpaceOptions; tol::Float64 = 1e-9)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)

    offset = 0.0
    for r in A.rows
        sz = size(r.RMT.data)
        n  = sz[1]
        vals = collect(offset .+ (1.0:n))
        offset += n + 1.0
        r.RMT.data .= reshape(Matrix(Diagonal(vals)), sz)
    end

    result = eigQS(A; hermitian = true)
    D = result.D
    V = result.V
    eig_list = result.eig_list
    Nkeep = min(3, length(eig_list))
    kept, discarded = discard_eigQS(result, Nkeep, "eigK", "eigD"; hermitian = true)
    Vkeep = kept.V
    Dkeep = kept.D
    eig_keep = kept.eig_list
    Vdiscard = discarded.V
    Ddiscard = discarded.D
    eig_discard = discarded.eig_list

    sorted_asc = sort(copy(eig_list); by = x -> real(x[1]))
    sorted_keep = sorted_asc[1:Nkeep]
    sorted_discard = sorted_asc[Nkeep+1:end]
    @test [(x[1], x[2], x[3]) for x in eig_keep] == [(x[1], x[2], x[3]) for x in sorted_keep]
    @test [(x[1], x[2], x[3]) for x in eig_discard] == [(x[1], x[2], x[3]) for x in sorted_discard]

    @test length(eig_keep) == Nkeep
    @test length(eig_keep) + length(eig_discard) == length(eig_list)

    for (entries, sorted_entries) in ((eig_keep, sorted_keep), (eig_discard, sorted_discard))
        isempty(entries) && continue
        sector_maps = Dict{typeof(entries[1][3]), Dict{Int, Int}}()
        sector_indices = Dict{typeof(entries[1][3]), Vector{Int}}()
        for entry in sorted_entries
            push!(get!(sector_indices, entry[3], Int[]), entry[4])
        end
        for (sector, idxs) in sector_indices
            sector_maps[sector] = Dict(old_idx => new_idx for (new_idx, old_idx) in enumerate(sort(unique(idxs))))
        end
        for (entry, entry_orig) in zip(entries, sorted_entries)
            @test entry[4] == sector_maps[entry[3]][entry_orig[4]]
        end
    end

    keep_vals = isempty(Dkeep.rows) ? eltype(D.rows[1].RMT.data)[] :
        sort(vcat([diag(reshape(r.RMT.data, size(r.RMT.data, 1), size(r.RMT.data, 2))) for r in Dkeep.rows]...))
    disc_vals = isempty(Ddiscard.rows) ? eltype(D.rows[1].RMT.data)[] :
        sort(vcat([diag(reshape(r.RMT.data, size(r.RMT.data, 1), size(r.RMT.data, 2))) for r in Ddiscard.rows]...))

    @test keep_vals ≈ sort([x[1] for x in eig_keep])
    @test disc_vals ≈ sort([x[1] for x in eig_discard])

    n_keep_cols = sum((size(r.RMT.data, 2) for r in Vkeep.rows); init = 0)
    n_disc_cols = sum((size(r.RMT.data, 2) for r in Vdiscard.rows); init = 0)
    @test n_keep_cols == length(eig_keep)
    @test n_disc_cols == length(eig_discard)

    keep_qls = Set(ql for (_, ql) in Dkeep.spaces[1])
    discard_qls = Set(ql for (_, ql) in Ddiscard.spaces[1])
    @test issubset(keep_qls, Set(ql for (_, ql) in D.spaces[1]))
    @test issubset(discard_qls, Set(ql for (_, ql) in D.spaces[1]))

    if !isempty(Dkeep.rows)
        rec_keep = lock(Vkeep, 1) * (Dkeep * Vkeep')
        arr_keep = Array(to_sparse_array(rec_keep))
        @test isfinite(norm(arr_keep))
    end

    if !isempty(Ddiscard.rows)
        rec_discard = lock(Vdiscard, 1) * (Ddiscard * Vdiscard')
        arr_discard = Array(to_sparse_array(rec_discard))
        @test isfinite(norm(arr_discard))
    end

    rec_total = nothing
    if !isempty(Dkeep.rows)
        rec_keep = lock(Vkeep, 1) * (Dkeep * Vkeep')
        rec_total = isnothing(rec_total) ? rec_keep : rec_total + rec_keep
    end
    if !isempty(Ddiscard.rows)
        rec_discard = lock(Vdiscard, 1) * (Ddiscard * Vdiscard')
        rec_total = isnothing(rec_total) ? rec_discard : rec_total + rec_discard
    end

    @test !isnothing(rec_total)
    arr_total = Array(to_sparse_array(rec_total))
    arr_A = Array(to_sparse_array(A))
    @test norm(arr_A - arr_total) / max(norm(arr_A), 1.0) < tol
end

# ─── test_eigQS_full_discard ─────────────────────────────────────────────────
# Verify that the result struct and discard path also work for the full
# non-Hermitian eigendecomposition, including V_inv slicing.
# ─────────────────────────────────────────────────────────────────────────────
function test_eigQS_full_discard(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)
    rng = Random.MersenneTwister(7)

    for r in A.rows
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n) + 0.2 * Matrix(I, n, n)
        r.RMT.data .= reshape(ComplexF64.(M), sz)
    end

    result = eigQS_full(A)
    @test !isnothing(result.V_inv)

    Nkeep = min(2, length(result.eig_list))
    kept, discarded = discard_eigQS(result, Nkeep, "eigK", "eigD"; hermitian = false)

    @test !isnothing(kept.V_inv)
    @test !isnothing(discarded.V_inv)
    @test length(kept.eig_list) <= Nkeep
    @test length(kept.eig_list) + length(discarded.eig_list) == length(result.eig_list)

    n_keep_v_cols = sum((size(r.RMT.data, 2) for r in kept.V.rows); init = 0)
    n_keep_vinv_rows = isnothing(kept.V_inv) ? 0 : sum((size(r.RMT.data, 1) for r in kept.V_inv.rows); init = 0)
    @test n_keep_v_cols == length(kept.eig_list)
    @test n_keep_vinv_rows == length(kept.eig_list)

    n_disc_v_cols = sum((size(r.RMT.data, 2) for r in discarded.V.rows); init = 0)
    n_disc_vinv_rows = isnothing(discarded.V_inv) ? 0 : sum((size(r.RMT.data, 1) for r in discarded.V_inv.rows); init = 0)
    @test n_disc_v_cols == length(discarded.eig_list)
    @test n_disc_vinv_rows == length(discarded.eig_list)
end

# ─── test_discard_eigQS_tags ─────────────────────────────────────────────────
# Verify that discard_eigQS retags the bond legs of D, V, and V_inv for kept
# and discarded eigenspaces independently.
# ─────────────────────────────────────────────────────────────────────────────
function test_discard_eigQS_tags(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    A = copy(q.I)
    rng = Random.MersenneTwister(17)

    for r in A.rows
        sz = size(r.RMT.data)
        n  = sz[1]
        M  = randn(rng, n, n) + 0.3 * Matrix(I, n, n)
        r.RMT.data .= reshape(ComplexF64.(M), sz)
    end

    result = eigQS_full(A, "origEig")
    kept_tag = "keptEig"
    discarded_tag = "discardedEig"
    kept, discarded = discard_eigQS(
        result,
        min(2, length(result.eig_list)),
        kept_tag,
        discarded_tag;
        hermitian = false,
    )

    @test kept.D.inds[1].itags == kept_tag
    @test kept.D.inds[2].itags == kept_tag
    @test kept.V.inds[2].itags == kept_tag
    @test !isnothing(kept.V_inv)
    @test kept.V_inv.inds[1].itags == kept_tag

    @test discarded.D.inds[1].itags == discarded_tag
    @test discarded.D.inds[2].itags == discarded_tag
    @test discarded.V.inds[2].itags == discarded_tag
    @test !isnothing(discarded.V_inv)
    @test discarded.V_inv.inds[1].itags == discarded_tag

    @test kept.V.inds[1] == result.V.inds[1]
    @test kept.V_inv.inds[2] == result.V_inv.inds[2]
    @test discarded.V.inds[1] == result.V.inds[1]
    @test discarded.V_inv.inds[2] == result.V_inv.inds[2]
end

# ─── test_spaces_svdQS ───────────────────────────────────────────────────────
# Verify that spaces of U, S, Vd returned by svdQS are consistent.
#
#   No-truncation case:
#   (a) U bond == S left
#   (b) S.spaces[2] is the exact dual splist of S.spaces[1]
#   (c) Vd bond == S right
#   (d) U non-bond leg spaces == corresponding input qf spaces
#   (e) Vd non-bond leg space == corresponding input qf space
#
#   Truncation case (Nkeep=1):
#   (f) Ut bond == St left
#   (g) St.spaces[2] is the exact dual splist of St.spaces[1]
#   (h) Vdt bond == St right
#   (i) St left ⊆ Ut bond  (S reduced; U/Vd bond inherits full pre-trunc space)
#   (j) Ut / Vdt non-bond leg spaces still match input qf spaces
# ─────────────────────────────────────────────────────────────────────────────
function test_spaces_svdQS(option::LocalSpaceOptions)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2); itags="lurlur")
    qf  = QSpace(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a   # rank-4: legs (lur1_in, lur2_in, lur2_out, op)

    # qlabel set from a splist
    qls(sp) = Set(ql for (_, ql) in sp)
    symm = q.I.symm; N = length(symm)

    # Compute the dual of a splist: apply get_dualq per-symmetry to every qlabel tuple.
    function dual_sp(sp)
        ET = eltype(sp)
        sort!(ET[(d, Tuple(get_dualq(symm[n], ql[n]) for n in 1:N))
                 for (d, ql) in sp]; by = x -> x[2])
    end

    # ── No-truncation case ──────────────────────────────────────────────────
    U, S, Vd = svdQS(ct, (1, 2))
    println("test_spaces_svdQS (no trunc): U bond=$(length(U.spaces[end])), " *
            "S=$(length(S.spaces[1]))/$(length(S.spaces[2])), " *
            "Vd bond=$(length(Vd.spaces[1])) sectors")

    # (a) U bond == S left
    @test qls(U.spaces[end]) == qls(S.spaces[1])

    # (b) S.spaces[2] is the exact dual of S.spaces[1]
    @test S.spaces[2] == dual_sp(S.spaces[1])

    # (c) Vd bond == S right
    @test qls(Vd.spaces[1]) == qls(S.spaces[2])

    # (d) U non-bond legs inherit spaces from input (left_legs = 1, 2)
    @test U.spaces[1] == ct.spaces[1]
    @test U.spaces[2] == ct.spaces[2]

    # (e) Vd non-bond leg inherits space from input (right_leg = 3)
    @test Vd.spaces[2] == ct.spaces[3]

    # ── Truncation case (Nkeep=1) ───────────────────────────────────────────
    Ut, St, Vdt = svdQS(ct, (1, 2); Nkeep = 1)
    println("test_spaces_svdQS (Nkeep=1): Ut bond=$(length(Ut.spaces[end])), " *
            "St=$(length(St.spaces[1]))/$(length(St.spaces[2])) sectors")

    # (f) Ut bond == St left
    @test qls(Ut.spaces[end]) == qls(St.spaces[1])

    # (g) St.spaces[2] is the exact dual of St.spaces[1]
    @test St.spaces[2] == dual_sp(St.spaces[1])

    # (h) Vdt bond == St right
    @test qls(Vdt.spaces[1]) == qls(St.spaces[2])

    # (i) St left ⊆ Ut bond (S sectors reduced; U/Vd bond keeps full pre-trunc space)
    @test issubset(qls(St.spaces[1]), qls(Ut.spaces[end]))

    # (j) Non-bond legs still match input after truncation
    @test Ut.spaces[1] == ct.spaces[1]
    @test Ut.spaces[2] == ct.spaces[2]
    @test Vdt.spaces[2] == ct.spaces[3]

    # ── Truncation case (Nkeep=2) ───────────────────────────────────────────
    Ut, St, Vdt = svdQS(ct, (1, 2); Nkeep = 2)
    println("test_spaces_svdQS (Nkeep=2): Ut bond=$(length(Ut.spaces[end])), " *
            "St=$(length(St.spaces[1]))/$(length(St.spaces[2])) sectors")

    # (f) Ut bond == St left
    @test qls(Ut.spaces[end]) == qls(St.spaces[1])

    # (g) St.spaces[2] is the exact dual of St.spaces[1]
    @test St.spaces[2] == dual_sp(St.spaces[1])

    # (h) Vdt bond == St right
    @test qls(Vdt.spaces[1]) == qls(St.spaces[2])

    # (i) St left ⊆ Ut bond (S sectors reduced; U/Vd bond keeps full pre-trunc space)
    @test issubset(qls(St.spaces[1]), qls(Ut.spaces[end]))

    # (j) Non-bond legs still match input after truncation
    @test Ut.spaces[1] == ct.spaces[1]
    @test Ut.spaces[2] == ct.spaces[2]
    @test Vdt.spaces[2] == ct.spaces[3]

end

# ─── test_truncate_svdQS ─────────────────────────────────────────────────────
# Verify that Nkeep truncation keeps the largest singular values, and that
# missing sectors are still counted as zero singular values when enough states
# are kept to include them. Original-leg spaces on U and Vd must stay intact.
# ─────────────────────────────────────────────────────────────────────────────
function test_truncate_svdQS(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))
    @test !isempty(q.I.rows)

    removed_row = q.I.rows[1]
    removed_dim = size(removed_row.RMT.data, 1)

    kept_rows = copy(q.I.rows[2:end])
    A = QSpace(q.I.symm, kept_rows, q.I.inds, q.I.spaces)

    offset = 0.0
    all_positive_vals = Float64[]
    for r in A.rows
        sz = size(r.RMT.data)
        n  = sz[1]
        vals = collect(offset .+ (1.0:n))
        offset += n + 1.0
        append!(all_positive_vals, vals)
        r.RMT.data .= reshape(Matrix(Diagonal(vals)), sz)
    end

    npositive_keep = min(2, length(all_positive_vals))
    Utop, Stop, Vdtop = svdQS(A, (1,); Nkeep = npositive_keep)

    @test Utop.spaces[1] == A.spaces[1]
    @test Vdtop.spaces[2] == A.spaces[2]
    @test Utop.spaces[2] == Stop.spaces[1]
    @test Vdtop.spaces[1] == Stop.spaces[2]

    kept_vals = isempty(Stop.rows) ? Float64[] :
        sort(vcat([diag(reshape(r.RMT.data, size(r.RMT.data, 1), size(r.RMT.data, 2))) for r in Stop.rows]...))
    expected_vals = sort(all_positive_vals; rev = true)[1:npositive_keep] |> sort
    @test kept_vals ≈ expected_vals

    total_keep = length(all_positive_vals) + removed_dim
    U, S, Vd = svdQS(A, (1,); Nkeep = total_keep)

    @test U.spaces[1] == A.spaces[1]
    @test Vd.spaces[2] == A.spaces[2]
    @test U.spaces[2] == S.spaces[1]
    @test Vd.spaces[1] == S.spaces[2]

    s_left_row_sectors = Set(
        Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(S.symm))
        for r in S.rows
    )
    s_right_row_sectors = Set(
        Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[2]] for n in 1:length(S.symm))
        for r in S.rows
    )

    missing_u_sector = only([
        ql for (dim, ql) in U.spaces[2]
        if dim == removed_dim && ql ∉ s_left_row_sectors
    ])
    missing_vd_sector = only([
        ql for (dim, ql) in Vd.spaces[1]
        if dim == removed_dim && ql ∉ s_right_row_sectors
    ])

    @test missing_u_sector in Set(ql for (_, ql) in S.spaces[1])
    @test missing_vd_sector in Set(ql for (_, ql) in S.spaces[2])

    u_row = only([r for r in U.rows if Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[end]] for n in 1:length(U.symm)) == missing_u_sector])
    u_mat = reshape(u_row.RMT.data, size(u_row.RMT.data, 1), size(u_row.RMT.data, 2))
    @test u_mat ≈ Matrix(I, removed_dim, removed_dim)

    vd_row = only([r for r in Vd.rows if Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:length(Vd.symm)) == missing_vd_sector])
    vd_mat = reshape(vd_row.RMT.data, size(vd_row.RMT.data, 1), size(vd_row.RMT.data, 2))
    @test vd_mat ≈ Matrix(I, removed_dim, removed_dim)
end

# ─── test_lock_reduce ─────────────────────────────────────────────────────────
# Verify selective lock-level reduction in contraction:
#
# Rule: a free output leg has its lock decremented by 1 only when the *other*
# tensor in the contraction has at least one leg with the same itag.
#
#   Scenario 1 — match present   : A free leg "free" lock=1 + B has "free" → lock → 0
#   Scenario 2 — no match        : A free leg "unique" lock=1, B has no "unique" → lock stays 1
#   Scenario 3 — lock already 0  : match present but lock=0 → lock stays 0
# ─────────────────────────────────────────────────────────────────────────────
function test_lock_reduce(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    # ── Scenario 1: matching leg present → lock decrements ───────────────────
    # q.I has legs ('+', '-').  A1 is built from q.I (same dirs), B1 from q.I'
    # (dirs flipped to ('-', '+')) so they can be contracted on "ct".
    # A1: ("ct" '+' lock=0,  "free" '-' lock=1)
    # B1: ("ct" '-' lock=0,  "free" '+' lock=0)
    # Contract A1*B1 on "ct"; A1's "free" lock=1 has a match in B1 → lock → 0.
    A1 = QSpace(q.I,
        (QIndex("ct",   '+', 0, 0),
         QIndex("free", '-', 0, 1)))
    B1 = QSpace(q.I',
        (QIndex("ct",   '-', 0, 0),
         QIndex("free", '+', 0, 0)))

    C1 = A1 * B1
    free_pos1 = findfirst(idx -> idx.itags == "free" && idx.dir == '-', C1.inds)
    @test C1.inds[free_pos1].lock == 0
    println("test_lock_reduce (match):    \"free\" lock=$(C1.inds[free_pos1].lock) (expected 0)")

    # ── Scenario 2: no matching leg → lock unchanged ─────────────────────────
    # A2: ("ct" '+' lock=0,  "unique" '-' lock=1)   — from q.I
    # B2: ("ct" '-' lock=0,  "other"  '+' lock=0)   — from q.I'; "unique" absent
    A2 = QSpace(q.I,
        (QIndex("ct",     '+', 0, 0),
         QIndex("unique", '-', 0, 1)))
    B2 = QSpace(q.I',
        (QIndex("ct",    '-', 0, 0),
         QIndex("other", '+', 0, 0)))

    C2 = A2 * B2
    unique_pos = findfirst(idx -> idx.itags == "unique", C2.inds)
    @test C2.inds[unique_pos].lock == 1
    println("test_lock_reduce (no match): \"unique\" lock=$(C2.inds[unique_pos].lock) (expected 1)")

    # ── Scenario 3: lock already 0 with match → stays 0 ─────────────────────
    # A3's "free" lock=0 and B1 has a matching "free" leg, but 0 cannot go lower.
    # Must use explicit contract (not *) to avoid auto-contracting both "ct" and "free".
    A3 = QSpace(q.I,
        (QIndex("ct",   '+', 0, 0),
         QIndex("free", '-', 0, 0)))
    C3 = contract(A3, (1,), B1, (1,))
    free_pos3 = findfirst(idx -> idx.itags == "free" && idx.dir == '-', C3.inds)
    @test C3.inds[free_pos3].lock == 0
    println("test_lock_reduce (lock=0):   \"free\" lock=$(C3.inds[free_pos3].lock) (expected 0)")
end

# ─── test_contract_requires_matching_spaces_in_star ─────────────────────────
# Verify that automatic contraction via `*` does not match tagged legs when
# their precomputed space metadata differs, even if the QIndex fields match.
function test_contract_requires_matching_spaces_in_star(option::LocalSpaceOptions)
    q = getLocalSpace(option, ("lur", "lur", "op"))

    A = QSpace(q.I,
        (QIndex("ct", '+', 0, 0),
         QIndex("",   '-', 0, 0)))
    B = QSpace(q.I',
        (QIndex("ct", '-', 0, 0),
         QIndex("",   '+', 0, 0)))

    first_dim, first_sector = first(B.spaces[1])
    bad_leg1 = vcat([(first_dim + 1, first_sector)], B.spaces[1][2:end])
    B_bad = QSpace(B.symm, B.rows, B.inds, (bad_leg1, B.spaces[2]))

    @test_throws AssertionError A * B_bad
end

# ─── test_contract_v2 ───────────────────────────────────────────────────────
# Compare contract_v2 (batched-GEMM version) against the original contract
# by converting both results to sparse arrays and computing the norm of the
# difference.  Several contraction patterns are exercised:
#
#   1. F × Identity   (3-leg × 4-leg, single contracted leg)
#   2. Inner product   (full contraction → scalar, via q · q')
#   3. Identity × Identity (2-leg × 2-leg, single contracted leg)
#   4. Large tensor  (ct × ct', contracting a subset of legs)
# ─────────────────────────────────────────────────────────────────────────────

# Helper: replicate the matching logic of Base.:*(::QSpace, ::QSpace) to find
# the contracted leg pairs, then call both contract and contract_v2 with
# those same explicit legs. Returns (arr_old, arr_new, diff_norm).
function _compare_contract_v1v2(q1::QSpace, q2::QSpace;
                                 tol::Float64 = 1e-10)
    QD1 = length(q1.inds)
    QD2 = length(q2.inds)
    cands1 = [(i, q1.inds[i]) for i in 1:QD1
              if !isempty(q1.inds[i].itags) && q1.inds[i].lock == 0]
    cands2 = [(j, q2.inds[j]) for j in 1:QD2
              if !isempty(q2.inds[j].itags) && q2.inds[j].lock == 0]
    legs1 = Int[];  legs2 = Int[];  matched2 = Set{Int}()
    for (i, idx1) in cands1
        hits = [(pos, j) for (pos, (j, idx2)) in enumerate(cands2)
                if idx1 == change_dir(idx2) &&
                   q1.spaces[i] == q2.spaces[j] &&
                   pos ∉ matched2]
        if length(hits) == 1
            pos, j = hits[1]
            push!(legs1, i); push!(legs2, j); push!(matched2, pos)
        end
    end
    @assert !isempty(legs1) "No matching legs found for v1-vs-v2 comparison"

    ct_v1 = contract(q1, Tuple(legs1), q2, Tuple(legs2); verify_legs=false)
    ct_v2 = contract_v2(q1, Tuple(legs1), q2, Tuple(legs2); verify_legs=false)

    # For 0-D (scalar) results, compare scalar values directly.
    if length(ct_v1.inds) == 0
        val_v1 = ct_v1[]
        val_v2 = ct_v2[]
        diff = abs(val_v1 - val_v2)
        ref  = max(abs(val_v1), 1.0)
        @test diff / ref < tol
        return diff
    end

    arr_v1 = Array(to_sparse_array(ct_v1))
    arr_v2 = Array(to_sparse_array(ct_v2))

    @assert size(arr_v1) == size(arr_v2) "Shape mismatch: $(size(arr_v1)) vs $(size(arr_v2))"
    diff = norm(arr_v1 - arr_v2)
    ref  = max(norm(arr_v1), 1.0)
    @test diff / ref < tol
    return diff
end

function test_contract_v2(option::LocalSpaceOptions; tol::Float64 = 1e-10)
    q   = getLocalSpace(option, ("lur", "lur", "op"))
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    qf  = QSpace(q.F, ("lur2", "lur2", "op"))
    a   = getIdentity((qi1, 2), (qi2, 2))

    # ── Case 1: F × Identity (3-leg × 4-leg, 1 contracted leg) ──────────────
    d1 = _compare_contract_v1v2(qf, a; tol=tol)
    println("test_contract_v2 [F*a]:   Δ = $d1")

    # ── Case 2: Inner product  (full contraction → scalar) ───────────────────
    d2 = _compare_contract_v1v2(qf, qf'; tol=tol)
    println("test_contract_v2 [F·F']:  Δ = $d2")

    # ── Case 3: Identity × Identity (2-leg × 2-leg, 1 contracted leg) ───────
    # Use explicit legs; verify_legs=false since tags may not match.
    ct3_v1 = contract(qi1, (2,), qi2, (1,); verify_legs=false)
    ct3_v2 = contract_v2(qi1, (2,), qi2, (1,); verify_legs=false)
    arr3_v1 = Array(to_sparse_array(ct3_v1))
    arr3_v2 = Array(to_sparse_array(ct3_v2))
    d3 = norm(arr3_v1 - arr3_v2)
    ref3 = max(norm(arr3_v1), 1.0)
    @test d3 / ref3 < tol
    println("test_contract_v2 [I*I]:   Δ = $d3")

    # ── Case 4: Larger tensor (ct × ct') ─────────────────────────────────────
    ct = qf * a   # 4-leg tensor
    d4 = _compare_contract_v1v2(ct, ct'; tol=tol)
    println("test_contract_v2 [ct·ct']: Δ = $d4")

    println("test_contract_v2 passed (all cases).")
end



