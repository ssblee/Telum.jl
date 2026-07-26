# Scalar multiplication and division: only the RMT arrays are scaled.
# CGT metadata (w-matrices, qlabels) are left untouched.
function _materialized_scale(qs::TLArray{T, QD, N, RD}, fac::Number) where {T, QD, N, RD}
    RT = promote_type(T, typeof(fac))
    if iszero(fac)
        RMTs = Vector{Array{RT, RD}}(undef, sector_count(qs))
        return TLArray(symm(qs), copy(qs.qlabels), qs.wmatdata, qs.wmatinfo, RMTs, qs.inds, qs.spaces)
    end
    if RT !== T
        RMTs = eltype(qs.RMTs) <: DiagRMT ? Vector{DiagRMT{RT, RD}}(undef, sector_count(qs)) :
                                            Vector{Array{RT, RD}}(undef, sector_count(qs))
        for sector_index in sector_slots(qs)
            qs.iszero[sector_index] && continue
            RMTs[sector_index] = qs.RMTs[sector_index] * fac
        end
        return TLArray(symm(qs), copy(qs.qlabels), _copy_wmat_storage(qs; deep=true)...,
                       RMTs, qs.inds, _copy_spaces_tuple(qs.spaces))
    end
    result = deepcopy(qs)
    for sector_index in sector_slots(result)
        result.iszero[sector_index] && continue
        result.RMTs[sector_index] = result.RMTs[sector_index] * fac
    end
    return result
end
Base.:*(qs::AbstractTLArray, fac::Number) = _view_scale(qs, fac)
Base.:*(fac::Number, qs::AbstractTLArray) = qs * fac
Base.:/(qs::AbstractTLArray, fac::Number) = qs * (1 / fac)
Base.:-(qs::AbstractTLArray) = qs * -1

# Return a deep copy of a TLArray (CGT metadata, RMTs, indices, spaces all copied).
Base.copy(q::TLArray) = deepcopy(q)
Base.copy(q::AbstractTLArray) = to_concrete(q)

function _identity_on_tlarray(q::AbstractTLArray{T, QD, N, RD}) where {T, QD, N, RD}
    @assert QD == 2 "Scalar add/subtract is only defined for rank-2 TLArray objects"

    in_legs  = findlegs(q; dir='+')
    out_legs = findlegs(q; dir='-')
    @assert length(in_legs) == 1 && length(out_legs) == 1 "Scalar add/subtract requires exactly one incoming and one outgoing leg"

    in_leg  = only(in_legs)
    out_leg = only(out_legs)
    legspaces = spaces(q)
    qinds = inds(q)
    @assert legspaces[in_leg] == legspaces[out_leg] "Scalar add/subtract requires matching incoming and outgoing spaces"

    id_q = getIdentity((q, out_leg); itag=qinds[out_leg].itags)
    return TLArray(id_q, (qinds[in_leg], qinds[out_leg]))
end

# ─── TLArray norm ─────────────────────────────────────────────────────────────
#
# Exploits the Wigner-Eckart decomposition to compute the Frobenius norm
# directly from the reduced matrix elements (RMTs) without building the full
# dense tensor.
#
# For all ranks:
#   Sectors from different q-label sectors are orthogonal by symmetry.  Within
#   each sector, the CGT canonical basis elements are orthonormal and the
#   w-matrix is left-orthogonal (Uᵀ·U = I), so the M columns of cgt_block
#   are also orthonormal.  Therefore all CGT cross-terms vanish and:
#       ‖A‖² = Σ_r ‖RMT_r‖²
#
# ─────────────────────────────────────────────────────────────────────────────
function LinearAlgebra.norm(q::TLArray{T, QD, N}) where {T, QD, N}
    s = zero(Float64)
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        rmt, scale = sector_rmt(q, sector_index)
        s += abs2(scale) * sum(abs2, rmt)
    end
    return sqrt(s)
end

LinearAlgebra.norm(q::TLArrayView) = abs(q.scale) * norm(q.arr)

function LinearAlgebra.norm(q::TLArrayContraction)
    compute_sectors(q, sector_slots(q))
    s = zero(Float64)
    for sector_index in sector_slots(q)
        q.iszero[sector_index] && continue
        rmt, scale = sector_rmt(q, sector_index)
        s += abs2(scale) * sum(abs2, rmt)
    end
    return sqrt(s)
end

Base.conj(q::AbstractTLArray) = _view_conj(q)
Base.adjoint(q::AbstractTLArray) = conj(q)
