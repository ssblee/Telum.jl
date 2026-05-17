# ─── RMT contraction helper ──────────────────────────────────────────────────
# Takes two pre-permuted RMTs and contracts over the contracted-leg axes.
# P1 shape: (free1..., contr..., om1_1,...,om1_N)   [nf1 + CN + N axes]
# P2 shape: (free2..., contr..., om2_1,...,om2_N)   [nf2 + CN + N axes]
# Result shape: (free1..., free2..., om1_1*om2_1, ..., om1_N*om2_N)  [nf1+nf2+N axes]
function _contract_RMTs(P1::AbstractArray, P2::AbstractArray,
                         nf1::Int, nf2::Int, N::Int, CN::Int)
    sz_f1  = [size(P1, i)      for i in 1:nf1]
    sz_c   = [size(P1, nf1+k)  for k in 1:CN]
    sz_om1 = [size(P1, nf1+CN+n) for n in 1:N]
    sz_f2  = [size(P2, i)      for i in 1:nf2]
    sz_om2 = [size(P2, nf2+CN+n) for n in 1:N]

    f1_dim = prod(sz_f1; init=1)
    f2_dim = prod(sz_f2; init=1)
    c_dim  = prod(sz_c; init=1)
    om1_dim = prod(sz_om1; init=1)
    om2_dim = prod(sz_om2; init=1)

    P1_3 = reshape(P1, f1_dim, c_dim, om1_dim)
    P2_3 = reshape(P2, f2_dim, c_dim, om2_dim)

    R1_mat = om1_dim == 1 ? reshape(P1_3, f1_dim, c_dim) :
             reshape(permutedims(P1_3, (1, 3, 2)), f1_dim * om1_dim, c_dim)
    R2_mat = om2_dim == 1 ? reshape(P2_3, f2_dim, c_dim) :
             reshape(permutedims(P2_3, (1, 3, 2)), f2_dim * om2_dim, c_dim)
    C_mat  = R1_mat * R2_mat'

    # Reshape → (f1..., om1..., f2..., om2...)
    QD_out = nf1 + nf2
    C = reshape(C_mat, sz_f1..., sz_om1..., sz_f2..., sz_om2...)

    # Permute C directly to (f1..., f2..., om1_1, om2_1, ..., om1_N, om2_N)
    # by composing the two permutations into one.
    composed_perm = vcat(1:nf1, nf1+N+1:nf1+N+nf2,
                         [[nf1+n, nf1+N+nf2+n] for n in 1:N]...)
    C_i = permutedims(C, composed_perm)

    # Merge each (om1_n, om2_n) pair → om1_n * om2_n
    sz_om_merged = [sz_om1[n] * sz_om2[n] for n in 1:N]
    return reshape(C_i, sz_f1..., sz_f2..., sz_om_merged...)
end

@inline _identity_perm(::Val{D}) where {D} = ntuple(identity, Val(D))

@inline function _maybe_permutedims(A::AbstractArray{T, D},
                                    perm::NTuple{D, Int}) where {T, D}
    return perm == _identity_perm(Val(D)) ? A : permutedims(A, perm)
end

@inline function _maybe_permutedims(A::AbstractArray{T, D}, perm) where {T, D}
    return Tuple(perm) == _identity_perm(Val(D)) ? A : permutedims(A, perm)
end

function _hptt_permutedims(A::Array{Float64, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{Float64}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:dTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float64, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Float64, Ptr{Float64}, Ptr{Int32}, Cint, Cint),
          perm0, D, 1.0, A, sizeA, C_NULL,
          0.0, out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims(A::Array{Float32, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{Float32}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:sTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, Float32, Ptr{Float32}, Ptr{Int32}, Ptr{Int32},
           Float32, Ptr{Float32}, Ptr{Int32}, Cint, Cint),
          perm0, D, Float32(1), A, sizeA, C_NULL,
          Float32(0), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims(A::Array{ComplexF64, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{ComplexF64}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:zTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF64, Cuchar, Ptr{ComplexF64}, Ptr{Int32}, Ptr{Int32},
           ComplexF64, Ptr{ComplexF64}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF64), false, A, sizeA, C_NULL,
          zero(ComplexF64), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

function _hptt_permutedims(A::Array{ComplexF32, D},
                           perm::NTuple{D, Int}) where {D}
    perm == _identity_perm(Val(D)) && return A

    out = Array{ComplexF32}(undef, ntuple(i -> size(A, perm[i]), Val(D)))
    perm0 = Int32[perm[i] - 1 for i in 1:D]
    sizeA = Int32[size(A, i) for i in 1:D]
    ccall((:cTensorTranspose, HPTT_jll.libhptt), Cvoid,
          (Ptr{Int32}, Cint, ComplexF32, Cuchar, Ptr{ComplexF32}, Ptr{Int32}, Ptr{Int32},
           ComplexF32, Ptr{ComplexF32}, Ptr{Int32}, Cint, Cint),
          perm0, D, one(ComplexF32), false, A, sizeA, C_NULL,
          zero(ComplexF32), out, C_NULL, max(1, Threads.nthreads()), 0)
    return out
end

@inline function _hptt_permutedims(A::AbstractArray{T, D},
                                   perm::NTuple{D, Int}) where {T, D}
    return Tuple(perm) == _identity_perm(Val(D)) ? A : permutedims(A, perm)
end

# ─── CGTContrInfo ────────────────────────────────────────────────────────────
# Holds the CGT space/leg data needed to call getNsave_Xsymbol for one
# non-abelian symmetry in a matched (r1, r2) sector pair.
struct CGTContrInfo{S<:NonabelianSymm, U1, D1, U2, D2, NZ, M}
    up1sp::NTuple{U1, NTuple{NZ, Int}}   # incoming qlabels of CGT1
    dn1sp::NTuple{D1, NTuple{NZ, Int}}   # outgoing qlabels of CGT1
    up2sp::NTuple{U2, NTuple{NZ, Int}}   # incoming qlabels of CGT2
    dn2sp::NTuple{D2, NTuple{NZ, Int}}   # outgoing qlabels of CGT2
    ctlegs1::NTuple{M, Int}              # contracted stored-qlabel indices in CGT1
    ctlegs2::NTuple{M, Int}              # contracted stored-qlabel indices in CGT2
end

# ─── Output CGT metadata ─────────────────────────────────────────────────────

# Core plain-data function: given two CGTs' qlabel/legdir/cgp data and the
# physical free/contracted legs, compute the resulting CGT's qlabels, cgp,
# and legdir.  This function is independent of the sector/CGT structs.
#
# Arguments (one symmetry at a time):
#   qlabels1/2  : NTuple{QD, NTuple{NZ,Int}} — stored qlabels of each CGT
#   legdir1/2   : (m, k) — number of incoming / outgoing stored qlabels
#   cgp1/2      : NTuple{QD, Int} — physical leg → stored qlabel index
#   free1/2     : physical free leg indices (1-based in each source TLArray)
#   legs1/2     : physical contracted leg indices
#
# Returns (new_qlabels, new_cgp, new_legdir).
function get_new_cgp(qlabels1::NTuple{QD1, NTuple{NZ, Int}}, legdir1,
                     cgp1::NTuple{QD1, Int},
                     free1::NTuple{NF1, Int}, legs1::NTuple{CN, Int},
                     qlabels2::NTuple{QD2, NTuple{NZ, Int}}, legdir2,
                     cgp2::NTuple{QD2, Int},
                     free2::NTuple{NF2, Int}, legs2::NTuple{CN, Int}) where {QD1, QD2, NZ, NF1, NF2, CN}
    m1, _ = legdir1;  m2, _ = legdir2
    QD_out = NF1 + NF2
    ctset1 = ntuple(k -> cgp1[legs1[k]], Val(CN))
    ctset2 = ntuple(k -> cgp2[legs2[k]], Val(CN))

    # (qlabel, output_physical_leg_index) pairs, insertion order = CGT1 then CGT2
    up3 = Vector{Tuple{NTuple{NZ, Int}, Int}}()
    dn3 = Vector{Tuple{NTuple{NZ, Int}, Int}}()
    sizehint!(up3, QD_out)
    sizehint!(dn3, QD_out)

    for (i, l_in) in enumerate(free1)
        sp = cgp1[l_in]
        sp ∈ ctset1 && continue
        if sp <= m1
            push!(up3, (qlabels1[sp], i))
        else
            push!(dn3, (qlabels1[sp], i))
        end
    end
    for (i, l_in) in enumerate(free2)
        sp = cgp2[l_in]
        sp ∈ ctset2 && continue
        if sp <= m2
            push!(up3, (qlabels2[sp], i + NF1))
        else
            push!(dn3, (qlabels2[sp], i + NF1))
        end
    end

    # Stable sort: equal qlabels keep CGT1-before-CGT2 insertion order.
    sort!(up3; by = x -> x[1], alg = MergeSort)
    sort!(dn3; by = x -> x[1], alg = MergeSort)

    m3, k3 = length(up3), length(dn3)
    new_qlabels = ntuple(Val(QD_out)) do i
        i <= m3 ? up3[i][1] : dn3[i - m3][1]
    end
    new_legdir  = (m3, k3)

    cgp3 = zeros(Int, QD_out)
    for (si, (_, l_out)) in enumerate(up3);  cgp3[l_out] = si      end
    for (si, (_, l_out)) in enumerate(dn3);  cgp3[l_out] = m3 + si end

    return (new_qlabels, ntuple(i -> cgp3[i], Val(QD_out)), new_legdir)
end

function get_new_cgp(qlabels1, legdir1, cgp1, free1, legs1,
                     qlabels2, legdir2, cgp2, free2, legs2)
    return get_new_cgp(qlabels1, legdir1, cgp1, Tuple(free1), Tuple(legs1),
                       qlabels2, legdir2, cgp2, Tuple(free2), Tuple(legs2))
end

function get_new_cgt_metadata(q1::TLArray{T1, QD1, N}, i1::Int,
                              q2::TLArray{T2, QD2, N}, i2::Int,
                              free1, free2, legs1, legs2) where {T1, QD1, T2, QD2, N}
    ntuple(Val(N)) do n
        qlabels1, cgp1, legdir1 = _sector_cgt_metadata(q1, i1, n)
        qlabels2, cgp2, legdir2 = _sector_cgt_metadata(q2, i2, n)
        get_new_cgp(qlabels1, legdir1, cgp1, free1, legs1,
                    qlabels2, legdir2, cgp2, free2, legs2)
    end
end

@inline function _physical_qlabels_from_cgt_metadata(::Type{QT},
                                                     metadata,
                                                     ::Val{QD},
                                                     ::Val{N}) where {QT, QD, N}
    return ntuple(Val(QD)) do leg
        ntuple(n -> metadata[n][1][metadata[n][2][leg]], Val(N))::QT
    end
end

# ─── merge_new_sector helpers ───────────────────────────────────────────────────

# Mode-product: replace axis `axis` of array A (size k) with the output of
# multiplying matrix M (shape r×k) along that axis.
# Equivalent to: result[..., j, ...] = Σ_l M[j,l] * A[...,l,...] at `axis`.
function _contract_om_axis_data(A::AbstractArray{T1, D}, M::AbstractMatrix{T2}, axis::Int) where {T1, T2<:Real, D}
    dims  = size(A)
    k     = dims[axis]
    r     = size(M, 1)
    @assert size(M, 2) == k "axis size $(k) != M columns $(size(M, 2))"

    findim = Base.setindex(dims, r, axis)
    if r == 1 && k == 1
        return LurTensor(reshape(A .* M[1, 1], findim))
    end

    prod_before = 1
    for i in 1:axis-1
        prod_before *= dims[i]
    end
    prod_after = 1
    for i in axis+1:D
        prod_after *= dims[i]
    end

    # Reshape A to (prod_before, k, prod_after), move k to the back,
    # multiply by M', then restore.  If prod_after == 1, k is already last.
    A_3  = reshape(A, prod_before, k, prod_after)
    if prod_after == 1
        A_mk = reshape(A_3, prod_before, k)
        R_mr = A_mk * transpose(M)
        return LurTensor(reshape(R_mr, findim))
    end

    A_mk = reshape(permutedims(A_3, (1, 3, 2)), prod_before * prod_after, k)
    R_mr = A_mk * transpose(M)
    R_3  = permutedims(reshape(R_mr, prod_before, prod_after, r), (1, 3, 2))
    return LurTensor(reshape(R_3, findim))
end

function _contract_om_axis(A::LurTensor{T1, D, A1}, M::LurTensor{T2, 2, A2}, axis::Int) where {T1, T2<:Real, D, A1, A2}
    return _contract_om_axis_data(A.data, M.data, axis)
end

function _contract_RMT_pair(rmt1::LurTensor{T1, RD1},
                            rmt2::LurTensor{T2, RD2},
                            factors::NTuple{N, LT},
                            perm1::NTuple{RD1, Int},
                            perm2::NTuple{RD2, Int},
                            nf1::Int,
                            nf2::Int,
                            cn::Int,
                            QD_out::Int) where {T1, T2, RD1, RD2, N, LT<:LurTensor{Float64, 3}}
    kept_sizes1, _, _ = _rmt_layout_sizes(rmt1, perm1, nf1, cn, N)
    kept_sizes2, _, _ = _rmt_layout_sizes(rmt2, perm2, nf2, cn, N)
    data1 = _permuted_rmt_data(rmt1, perm1, nf1, cn, N)
    data2 = _permuted_rmt_data(rmt2, perm2, nf2, cn, N)
    rank_sizes, rank_dim, _, _, _ = _combined_om_factor_info(factors)

    RT = promote_type(T1, T2, Float64)
    result_data = zeros(RT, (kept_sizes1..., kept_sizes2..., rank_sizes...))
    result_view = reshape(result_data, size(data1, 1), size(data2, 1), rank_dim)
    K, _ = _rmt_factor_array_and_rank_sizes(factors)
    temp = Vector{RT}(undef, _rmt_contract_temp_len(data1, data2, K))
    _contract_RMT_pair_into!(result_view, data1, data2, K, temp)
    return LurTensor(result_data)
end

const _RMT_CONTRACT_TULLIO_THRESHOLD = 1_000_000

function _rmt_layout_sizes(rmt::LurTensor{T, RD},
                           perm::NTuple{RD, Int},
                           nf::Int,
                           cn::Int,
                           nsyms::Int) where {T, RD}
    return _rmt_layout_sizes(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _rmt_layout_sizes(rmt::LurTensor{T, RD},
                           perm::NTuple{RD, Int},
                           ::Val{NF},
                           ::Val{CN},
                           ::Val{N}) where {T, RD, NF, CN, N}
    dims = size(rmt)
    kept_sizes = ntuple(i -> dims[perm[i]], Val(NF))
    contracted_sizes = ntuple(i -> dims[perm[NF + i]], Val(CN))
    om_sizes = ntuple(i -> dims[perm[NF + CN + i]], Val(N))
    return kept_sizes, contracted_sizes, om_sizes
end

@inline _rmt_array_data(rmt::LurTensor{T, RD}) where {T, RD} =
    rmt.data::Array{T, RD}

function _permuted_rmt_data(rmt::LurTensor{T, RD},
                            perm::NTuple{RD, Int},
                            nf::Int,
                            cn::Int,
                            nsyms::Int) where {T, RD}
    return _permuted_rmt_data(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _permuted_rmt_data(rmt::LurTensor{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    data = reshape(_hptt_permutedims(_rmt_array_data(rmt), perm),
                   prod(kept_sizes; init=1),
                   prod(contracted_sizes; init=1),
                   prod(om_sizes; init=1))
    return data
end

function _cached_permuted_rmt_data!(cache::Vector{Array{T, 3}},
                                    sectors::AbstractVector,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    nf::Int,
                                    cn::Int,
                                    nsyms::Int) where {T, RD}
    return _cached_permuted_rmt_data!(cache, sectors, idx, perm,
                                      Val(nf), Val(cn), Val(nsyms))
end

function _cached_permuted_rmt_data!(cache::Vector{Array{T, 3}},
                                    sectors::AbstractVector,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {T, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = _permuted_rmt_data(sector_rmt(sectors[idx]), perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

function _cached_permuted_rmt_data!(cache::Vector{Array{T, 3}},
                                    q::TLArray,
                                    idx::Int,
                                    perm::NTuple{RD, Int},
                                    ::Val{NF},
                                    ::Val{CN},
                                    ::Val{N}) where {T, RD, NF, CN, N}
    if !isassigned(cache, idx)
        cache[idx] = _permuted_rmt_data(sector_rmt(q, idx), perm, Val(NF), Val(CN), Val(N))
    end
    return cache[idx]
end

function _combined_om_factor_info(factors::NTuple{N, LT}) where {N, LT<:LurTensor{Float64, 3}}
    rank_sizes = ntuple(n -> size(factors[n].data, 1), Val(N))
    scale = 1.0

    rank_dim = 1
    om1_dim = 1
    om2_dim = 1
    for n in 1:N
        factor = factors[n].data
        r_n, om1_n, om2_n = size(factor)

        if r_n == 1 && om1_n == 1 && om2_n == 1
            scale *= factor[1, 1, 1]
            continue
        end

        rank_dim *= r_n
        om1_dim *= om1_n
        om2_dim *= om2_n
    end

    return rank_sizes, rank_dim, om1_dim, om2_dim, scale
end

function _combined_om_factor_array(factors::NTuple{N, LT},
                                   rank_dim_total::Int,
                                   om1_dim_total::Int,
                                   om2_dim_total::Int) where {N, LT<:LurTensor{Float64, 3}}
    K = ones(Float64, 1, 1, 1)
    rank_dim = 1
    om1_dim = 1
    om2_dim = 1

    for n in 1:N
        factor = factors[n].data
        r_n, om1_n, om2_n = size(factor)

        if r_n == 1 && om1_n == 1 && om2_n == 1
            continue
        end

        F = reshape(factor, r_n, om1_n, om2_n)
        K_new = Array{Float64}(undef, rank_dim * r_n,
                               om1_dim * om1_n,
                               om2_dim * om2_n)
        @inbounds for o2n in 1:om2_n, o2p in 1:om2_dim,
                      o1n in 1:om1_n, o1p in 1:om1_dim,
                      rn in 1:r_n, rp in 1:rank_dim
            K_new[rp + (rn - 1) * rank_dim,
                  o1p + (o1n - 1) * om1_dim,
                  o2p + (o2n - 1) * om2_dim] = K[rp, o1p, o2p] * F[rn, o1n, o2n]
        end
        K = K_new
        rank_dim *= r_n
        om1_dim *= om1_n
        om2_dim *= om2_n
    end

    @assert size(K) == (rank_dim_total, om1_dim_total, om2_dim_total)
    return K
end

function _rmt_factor_array_and_rank_sizes(factors::NTuple{N, LT}) where {N, LT<:LurTensor{Float64, 3}}
    rank_sizes, rank_dim, om1_dim, om2_dim, scale =
        _combined_om_factor_info(factors)
    K = _combined_om_factor_array(factors, rank_dim, om1_dim, om2_dim)
    scale == 1.0 || (K .*= scale)
    return K, rank_sizes
end

function _prepare_compress_sector(
    new_wmats::NTuple{N, Vector{LurTensor{Float64, 3, AW}}},
    tol::Float64 = 1e-12,
) where {N, AW<:AbstractArray{Float64, 3}}
    K = length(first(new_wmats))
    @assert K > 0 "_compress_sector requires at least one RMT contract"

    U_mats   = Vector{LurTensor{Float64, 2, Matrix{Float64}}}(undef, N)
    SV_split = Matrix{LurTensor{Float64, 3, AW}}(undef, N, K)

    for n in 1:N
        shared = _qr_shared_isometry(new_wmats[n]; tol=tol)
        isnothing(shared) && return nothing
        common_iso::LurTensor{Float64, 2, Matrix{Float64}},
        factors::Vector{LurTensor{Float64, 3, AW}} = shared

        U_mats[n] = common_iso
        for i in 1:K
            SV_split[n, i] = factors[i]
        end
    end

    factor_arrays = Vector{Array{Float64, 3}}(undef, K)
    first_rank_sizes = ntuple(_ -> 0, Val(N))
    for i in 1:K
        factor_arrays[i], rank_sizes =
            _rmt_factor_array_and_rank_sizes(ntuple(n -> SV_split[n, i], Val(N)))
        i == 1 && (first_rank_sizes = rank_sizes)
    end

    return U_mats, factor_arrays, first_rank_sizes
end

function _rmt_contract_order(fdim::Int, gdim::Int, cdim::Int,
                             o1dim::Int, o2dim::Int, rdim::Int)
    ab_cost = fdim * gdim * o1dim * o2dim * cdim +
              fdim * gdim * rdim * o1dim * o2dim
    ak_cost = fdim * cdim * rdim * o1dim * o2dim +
              fdim * gdim * rdim * cdim * o2dim
    bk_cost = gdim * cdim * rdim * o1dim * o2dim +
              fdim * gdim * rdim * cdim * o1dim

    if ab_cost <= ak_cost && ab_cost <= bk_cost
        return :AB, ab_cost
    elseif ak_cost <= bk_cost
        return :AK, ak_cost
    else
        return :BK, bk_cost
    end
end

function _rmt_contract_temp_len(fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int, rdim::Int,
                                order::Symbol)
    if order === :AB
        return fdim * o1dim * gdim * o2dim
    elseif order === :AK
        return fdim * cdim * o2dim * rdim
    else
        return gdim * cdim * o1dim * rdim
    end
end

function _rmt_contract_temp_len(fdim::Int, gdim::Int, cdim::Int,
                                o1dim::Int, o2dim::Int,
                                K::AbstractArray,
                                threshold::Int = _RMT_CONTRACT_TULLIO_THRESHOLD)
    rank_dim, om1_dim, om2_dim = size(K)
    rank_dim == 1 && om1_dim == 1 && om2_dim == 1 && return 0

    order, _ = _rmt_contract_order(fdim, gdim, cdim, o1dim, o2dim, rank_dim)
    return _rmt_contract_temp_len(fdim, gdim, cdim, o1dim, o2dim, rank_dim, order)
end

function _rmt_contract_temp_len(A::AbstractArray,
                                B::AbstractArray,
                                K::AbstractArray,
                                threshold::Int = _RMT_CONTRACT_TULLIO_THRESHOLD)
    fdim, cdim, o1dim = size(A)
    gdim, _, o2dim = size(B)
    return _rmt_contract_temp_len(fdim, gdim, cdim, o1dim, o2dim,
                                  K, threshold)
end

function _rmt_contract_dims(rmt::LurTensor{T, RD},
                            perm::NTuple{RD, Int},
                            nf::Int,
                            cn::Int,
                            nsyms::Int) where {T, RD}
    return _rmt_contract_dims(rmt, perm, Val(nf), Val(cn), Val(nsyms))
end

function _rmt_contract_dims(rmt::LurTensor{T, RD},
                            perm::NTuple{RD, Int},
                            ::Val{NF},
                            ::Val{CN},
                            ::Val{N}) where {T, RD, NF, CN, N}
    kept_sizes, contracted_sizes, om_sizes =
        _rmt_layout_sizes(rmt, perm, Val(NF), Val(CN), Val(N))
    return (prod(kept_sizes; init=1),
            prod(contracted_sizes; init=1),
            prod(om_sizes; init=1))
end

@inline function _rmt_temp_view(temp::Vector{T},
                               dims::NTuple{N, Int}) where {T, N}
    @assert length(temp) >= prod(dims; init=1)
    return unsafe_wrap(Array, pointer(temp), dims; own=false)
end

@inline function _contiguous_matrix_view(A::Array{T}, offset::Int,
                                         dims::Tuple{Int, Int}) where {T}
    return unsafe_wrap(Array, pointer(A, offset), dims; own=false)
end

function _accumulate_small!(out::AbstractArray{T, 3},
                            A::AbstractArray,
                            B::AbstractArray,
                            K::AbstractArray,
                            order::Symbol,
                            temp::Vector{T}) where {T}
    fdim, cdim, o1dim = size(A)
    gdim = size(B, 1)
    o2dim = size(B, 3)
    rdim = size(K, 1)

    if order === :AB
        tmp = _rmt_temp_view(temp, (fdim, o1dim, gdim, o2dim))
        @tullio avx=true tmp[f, o1, g, o2] = A[f, c, o1] * B[g, c, o2]
        @tullio avx=true out[f, g, r] += tmp[f, o1, g, o2] * K[r, o1, o2]
    elseif order === :AK
        tmp = _rmt_temp_view(temp, (fdim, cdim, o2dim, rdim))
        @tullio avx=true tmp[f, c, o2, r] = A[f, c, o1] * K[r, o1, o2]
        @tullio avx=true out[f, g, r] += tmp[f, c, o2, r] * B[g, c, o2]
    else
        tmp = _rmt_temp_view(temp, (gdim, cdim, o1dim, rdim))
        @tullio avx=true tmp[g, c, o1, r] = B[g, c, o2] * K[r, o1, o2]
        @tullio avx=true out[f, g, r] += A[f, c, o1] * tmp[g, c, o1, r]
    end

    return out
end

function _accumulate_mkl!(out::AbstractArray{T, 3},
                          A::AbstractArray,
                          B::AbstractArray,
                          K::AbstractArray,
                          order::Symbol,
                          temp::Vector{T}) where {T}
    fdim, cdim, o1dim = size(A)
    gdim = size(B, 1)
    o2dim = size(B, 3)
    rdim = size(K, 1)

    if order === :AB
        tmp = _rmt_temp_view(temp, (fdim, gdim, o1dim, o2dim))
        for i in 1:o1dim
            for j in 1:o2dim
                tmp_view = @view tmp[:, :, i, j]
                A_, B_ = @view(A[:, :, i]), @view B[:, :, j]
                mul!(tmp_view, A_, transpose(B_), one(T), zero(T))
            end
        end
        tmp_ = reshape(tmp, fdim * gdim, o1dim * o2dim)
        K_ = reshape(K, rdim, o1dim * o2dim)
        out_ = reshape(out, fdim * gdim, rdim)
        mul!(out_, tmp_, transpose(K_), one(T), one(T))

    elseif order === :AK
        tmp = _rmt_temp_view(temp, (fdim, cdim, o2dim, rdim))
        Kt = permutedims(K, (2, 3, 1))
        A_ = reshape(A, fdim * cdim, o1dim)
        Kt_ = reshape(Kt, o1dim, o2dim * rdim)
        tmp_r1 = reshape(tmp, fdim * cdim, o2dim * rdim)
        mul!(tmp_r1, A_, Kt_, one(T), zero(T))

        B_mat_t = transpose(reshape(B, gdim, cdim * o2dim))
        for r in 1:rdim
            tmp_r2 = reshape(@view(tmp[:, :, :, r]), (fdim, cdim * o2dim))
            out_r = @view(out[:, :, r])
            mul!(out_r, tmp_r2, B_mat_t, one(T), one(T))
        end

    else
        tmp = _rmt_temp_view(temp, (gdim, cdim, o1dim, rdim))
        Kt = permutedims(K, (3, 2, 1))
        B_ = reshape(B, gdim * cdim, o2dim)
        Kt_ = reshape(Kt, o2dim, o1dim * rdim)
        tmp_r1 = reshape(tmp, gdim * cdim, o1dim * rdim)
        mul!(tmp_r1, B_, Kt_, one(T), zero(T))

        A_mat = reshape(A, fdim, cdim * o1dim)
        for r in 1:rdim
            tmp_r2 = reshape(@view(tmp[:, :, :, r]), (gdim, cdim * o1dim))
            out_r = @view(out[:, :, r])
            mul!(out_r, A_mat, transpose(tmp_r2), one(T), one(T))
        end
    end

    return out
end

function _accumulate_scalar_om!(out::AbstractArray{T, 3},
                                A::AbstractArray,
                                B::AbstractArray,
                                scale::Float64) where {T}
    fdim, cdim, o1dim = size(A)
    gdim, cdim2, o2dim = size(B)
    @assert cdim == cdim2
    @assert o1dim == 1 && o2dim == 1 && size(out, 3) == 1

    A_mat = reshape(A, fdim, cdim)
    B_mat = reshape(B, gdim, cdim)
    out_mat = @view out[:, :, 1]
    mul!(out_mat, A_mat, transpose(B_mat), scale, one(T))
    return out
end

function _contract_RMT_pair_into!(out::AbstractArray{T, 3},
                                  A::AbstractArray{T1, 3},
                                  B::AbstractArray{T2, 3},
                                  K::AbstractArray,
                                  temp::Vector{T},
                                  threshold::Int = _RMT_CONTRACT_TULLIO_THRESHOLD) where {T, T1, T2}
    rank_dim, om1_dim, om2_dim = size(K)

    if rank_dim == 1 && om1_dim == 1 && om2_dim == 1
        return _accumulate_scalar_om!(out, A, B, K[1, 1, 1])
    end

    fdim, cdim, o1dim = size(A)
    gdim, _, o2dim = size(B)
    order, cost = _rmt_contract_order(fdim, gdim, cdim, o1dim, o2dim, rank_dim)

    if cost < threshold
        _accumulate_small!(out, A, B, K, order, temp)
    else
        _accumulate_mkl!(out, A, B, K, order, temp)
    end

    return out
end

# ─── _compress_sector ────────────────────────────────────────────────────────
# Pure-arithmetic core: given K per-pair (w-matrix, RMT) contributions for one
# output sector, compress the per-symmetry w-matrices into a shared basis and
# accumulate the result into a single (U_mats, result_RMT) pair.
#
#   new_wmats[n][i] : (OM3_n, OM12_n_i)  — w-matrix for pair i, symmetry n
#   new_RMTs[i]     : LurTensor{T} (sz_free..., OM12_1_i,...,OM12_N_i)
#
# Returns:
#   U_mats[n]  : LurTensor{Float64,2} (OM3_n, r_n)  — new CGT wmat per symmetry
#   result_RMT : LurTensor{T} (sz_free..., r_1,...,r_N)  — compressed RMT
#   nothing    : when QR reveals that any symmetry contributes no usable
#                columns, so the whole merged sector is identically zero
#
# We use QR-based shared isometries for every sector, including K == 1, so the
# resulting basis is normalized consistently with the multi-contribution case.
function _compress_sector(
    new_wmats ::NTuple{N, Vector{LurTensor{Float64, 2, AW}}},
    new_RMTs  ::Vector{LurTensor{T, RD, AR}},
    QD_out    ::Int,
    tol       ::Float64 = 1e-12,
) where {T, N, RD, AW<:AbstractArray{Float64, 2}, AR<:AbstractArray{T, RD}}
    K = length(new_RMTs)
    @assert K > 0 "_compress_sector requires at least one RMT"

    # Shared QR basis per symmetry
    U_mats   = Vector{LurTensor{Float64, 2, AW}}(undef, N)
    # SV pieces per symmetry and pair
    SV_split = Matrix{LurTensor{Float64, 2, AW}}(undef, N, K)  

    for n in 1:N
        shared = _qr_shared_isometry(new_wmats[n]; tol=tol)
        isnothing(shared) && return nothing
        common_iso::LurTensor{Float64, 2, AW}, 
        factors::Vector{LurTensor{Float64, 2, AW}} = shared

        U_mats[n] = common_iso
        for i in 1:K
            SV_split[n, i] = factors[i]
        end
    end

    result_RMT = new_RMTs[1]
    for n in 1:N
        result_RMT = _contract_om_axis(result_RMT, SV_split[n, 1], QD_out + n)
    end

    # Reuse the first contracted contribution as the output buffer.  This avoids
    # allocating a separate zero-filled result, and removes the double allocation
    # in the K == 1 case.
    for i in 2:K
        contrib = new_RMTs[i]
        for n in 1:N
            contrib = _contract_om_axis(contrib, SV_split[n, i], QD_out + n)
        end
        result_RMT .+= contrib
    end

    return U_mats, result_RMT
end

function _compress_sector(
    new_wmats    ::NTuple{N, Vector{LurTensor{Float64, 3, AW}}},
    sector_pairs    ::AbstractVector{Tuple{Int, Int}},
    work1        ::Dict{Int, Array{T1, 3}},
    work2        ::Dict{Int, Array{T2, 3}},
    kept_sizes1  ::NTuple{NF1, Int},
    kept_sizes2  ::NTuple{NF2, Int},
    tol          ::Float64 = 1e-12,
) where {N, AW<:AbstractArray{Float64, 3}, T1, T2, NF1, NF2}
    prepared = _prepare_compress_sector(new_wmats, tol)
    isnothing(prepared) && return nothing
    first_i, first_j = first(sector_pairs)
    first_data1 = work1[first_i]
    first_data2 = work2[first_j]

    RT = promote_type(eltype(first_data1), eltype(first_data2), Float64)
    max_temp_len = 0
    for i in eachindex(sector_pairs)
        idx1, idx2 = sector_pairs[i]
        max_temp_len = max(max_temp_len,
                           _rmt_contract_temp_len(work1[idx1], work2[idx2],
                                                  prepared[2][i]))
    end
    temp = Vector{RT}(undef, max_temp_len)

    return _contract_prepared_compress_sector(prepared, sector_pairs, work1, work2,
                                              kept_sizes1, kept_sizes2, temp,
                                              Val(NF1 + NF2 + N))
end

function _contract_prepared_compress_sector(
    prepared,
    sector_pairs    ::AbstractVector{Tuple{Int, Int}},
    work1,
    work2,
    kept_sizes1  ::NTuple{NF1, Int},
    kept_sizes2  ::NTuple{NF2, Int},
    temp         ::Vector{RT},
    ::Val{RD_out}
) where {NF1, NF2, RT, RD_out}
    first_i, first_j = first(sector_pairs)
    first_data1 = work1[first_i]
    first_data2 = work2[first_j]
    U_mats, factor_arrays, rank_sizes = prepared

    result_shape::NTuple{RD_out, Int} = (kept_sizes1..., kept_sizes2..., rank_sizes...)
    result_data::Array{RT, RD_out} = zeros(RT, result_shape)
    result_view = reshape(result_data,
                          size(first_data1, 1),
                          size(first_data2, 1),
                          size(first(factor_arrays), 1))

    for i in eachindex(sector_pairs)
        idx1, idx2 = sector_pairs[i]
        _contract_RMT_pair_into!(
            result_view,
            work1[idx1],
            work2[idx2],
            factor_arrays[i],
            temp,
        )
    end

    return U_mats, LurTensor(result_data)
end

# ─── * operator ──────────────────────────────────────────────────────────────
# Automatically contract two TLArray objects by matching their tagged, unlocked
# indices.  An index on q1 is "contractible" when it has a nonempty tag AND
# lock == 0; same criterion applies to q2.  Two contractible indices are matched
# when they compare equal under TLIndex == (same itags, dir, plev, green) and
# their precomputed leg spaces are equal. The collected matching pairs define
# legs1 / legs2 passed to `contract`.
function Base.:*(q1::TLArray, q2::TLArray)
    # Collect candidate indices from each TLArray.
    cands1 = [(i, q1.inds[i]) for i in 1:length(q1.inds)
              if !isempty(q1.inds[i].itags) && q1.inds[i].lock == 0]
    cands2 = [(j, q2.inds[j]) for j in 1:length(q2.inds)
              if !isempty(q2.inds[j].itags) && q2.inds[j].lock == 0]

    # Match candidates: for each index in cands1, find the unique equal index
    # in cands2.  Raise an error if a tag appears more than once on either side.
    legs1 = Int[]
    legs2 = Int[]
    matched2 = Set{Int}()   # positions in cands2 already consumed

    for (i::Int, idx1) in cands1
        hits = [(pos, j, idx2) for (pos, (j, idx2)) in enumerate(cands2)
                if idx1 == change_dir(idx2) &&
                   q1.spaces[i] == q2.spaces[j] &&
                   pos ∉ matched2]::Vector{Tuple{Int, Int, TLIndex}}
        if length(hits) > 1
            error("Ambiguous contraction: tag \"$(idx1.itags)\" matches more than one index in q2")
        end
        if length(hits) == 1
            pos::Int, j::Int, _ = hits[1]
            push!(legs1, i)::Vector{Int}
            push!(legs2, j)::Vector{Int}
            push!(matched2, pos)::Set{Int}
        end
    end

    @assert length(legs1) > 0 "No matching contractible indices found between the two TLArray objects"

    return contract(q1, Tuple(legs1), q2, Tuple(legs2); verify_legs=false)
end

# ─── contract ────────────────────────────────────────────────────────────────
# Contraction sorted by *contracted* qlabels.  RMT pairs are collected as
# pending contractions per output sector, so the actual RMT contraction happens
# after the w-matrices have been compressed in `_compress_sector`.
#
# Algorithm sketch:
#   1. Build sorted sector-info vectors for each TLArray. Each entry carries the
#      sector index, all physical-leg qlabels, and the contracted-qlabel key.
#   2. Two-pointer scan over both sorted vectors and process common sectors:
#      a) Contract w-matrices / X-symbols for each compatible sector pair.
#      b) Store the source RMT pair and fixed layout permutation without
#         materializing the pairwise contracted RMT.
#      c) Accumulate pending contributions per output free-sector.
#   3. Merge each output sector (QR compression → final RMT contraction).
#   4. Lock reduction / build result TLArray.

# ── Contracted-label helpers ─────────────────────────────────────────────────
function _free_legs(::Val{QD}, legs::NTuple{CN, Int}) where {QD, CN}
    return ntuple(Val(QD - CN)) do k
        nfree = 0
        for l in 1:QD
            if l ∉ legs
                nfree += 1
                nfree == k && return l
            end
        end
        throw(BoundsError())
    end
end

_contracted_qlabel_type(::Type{QT}, ::Val{CN}) where {QT, CN} = NTuple{CN, QT}

struct ContractSectorInfo{NF, QT, CQT}
    sector_index::Int
    free_qlabels::NTuple{NF, QT}
    contracted_qlabels::CQT
end

function _contract_sector_infos(::Type{QT}, q::TLArray{T, QD, N},
                             free_legs::NTuple{NF, Int},
                             legs::NTuple{CN, Int}) where {QT, T, QD, N, NF, CN}
    CQT = _contracted_qlabel_type(QT, Val(CN))
    Info = ContractSectorInfo{NF, QT, CQT}
    infos = Vector{Info}(undef, nsectors(q))
    for i in 1:nsectors(q)
        infos[i] = Info(
            i,
            ntuple(k -> sector_qlabel(QT, q, i, free_legs[k]), Val(NF)),
            ntuple(k -> sector_qlabel(QT, q, i, legs[k]), Val(CN))::CQT,
        )
    end
    return sort!(infos; by = info -> info.contracted_qlabels, alg=MergeSort)
end

function _contracted_qlabel_run(infos::AbstractVector{<:ContractSectorInfo{NF, QT, CQT}},
                                first_pos::Int) where {NF, QT, CQT}
    key = infos[first_pos].contracted_qlabels
    next_pos = first_pos + 1
    while next_pos <= lastindex(infos) && infos[next_pos].contracted_qlabels == key
        next_pos += 1
    end
    return key, first_pos:(next_pos - 1), next_pos
end

function _contract_xsym_wmat(wm1::AbstractMatrix{T1},
                             xarr::AbstractArray{T2, 3},
                             wm2::AbstractMatrix{T3}) where {T1, T2, T3}
    OM1, OM2, OM3 = size(xarr)
    size(wm1, 1) == OM1 ||
        throw(DimensionMismatch("wm1 first axis must match x-symbol OM1 axis"))
    size(wm2, 1) == OM2 ||
        throw(DimensionMismatch("wm2 first axis must match x-symbol OM2 axis"))

    d1 = size(wm1, 2)
    d2 = size(wm2, 2)
    WT = promote_type(T1, T2, T3)

    result = Array{WT}(undef, OM3, d1, d2)
    @tullio avx=true result[a, b, c] =
        xarr[bb, cc, a] * wm1[bb, b] * wm2[cc, c]
    return LurTensor(result)
end

@inline _symmetry_qlabels(qlabels::NTuple{NF, QT}, ::Val{n}) where {NF, QT, n} =
    ntuple(k -> qlabels[k][n], Val(NF))

@inline _symmetry_contracted_qlabel(qlabels::NTuple{CN, QT}, ::Val{n}, ::Val{CN}) where {CN, QT, n} =
    ntuple(k -> qlabels[k][n], Val(CN))

const _XSymCache = Dict{Any, Union{Nothing, Array{Float64, 3}}}
const _ABELIAN_WMAT_3D = LurTensor(reshape([1.0], 1, 1, 1))

@inline _xsym_cache_for(::Type{S}, caches, ::Val{n}) where {S<:AbelianSymm, n} = nothing
@inline _xsym_cache_for(::Type{S}, caches, ::Val{n}) where {S<:NonabelianSymm, n} =
    caches[n]::_XSymCache

function _new_xsym_caches(::Type{ProductSymm{Syms}}, ::Val{N}) where {Syms, N}
    caches = Vector{Union{Nothing, _XSymCache}}(undef, N)
    shared = Dict{DataType, _XSymCache}()
    for n in 1:N
        S = fieldtype(Syms, n)
        caches[n] = S <: AbelianSymm ? nothing : get!(shared, S) do
            _XSymCache()
        end
    end
    return caches
end

@inline function _xsym_cache_key(::Type{QT},
                                 contracted_qlabels::NTuple{CN, QT},
                                 free_qlabels1::NTuple{NF1, QT},
                                 free_qlabels2::NTuple{NF2, QT},
                                 ::Val{n}) where {QT, NF1, NF2, n, CN}
    return (_symmetry_contracted_qlabel(contracted_qlabels, Val(n), Val(CN)),
            _symmetry_qlabels(free_qlabels1, Val(n)),
            _symmetry_qlabels(free_qlabels2, Val(n)))
end

@inline _cgt_up_qlabels(qlabels::NTuple{QD, NTuple{NZ, Int}}, ::Val{M}) where {QD, NZ, M} =
    ntuple(i -> qlabels[i], Val(M))

@inline _cgt_dn_qlabels(qlabels::NTuple{QD, NTuple{NZ, Int}}, ::Val{M}) where {QD, NZ, M} =
    ntuple(i -> qlabels[M + i], Val(QD - M))

@inline _stored_contract_legs(cgp::NTuple{QD, Int}, legs::NTuple{CN, Int}) where {QD, CN} =
    ntuple(k -> cgp[legs[k]], Val(CN))

@generated function _load_nonabelian_xarr(::Type{S},
                                          qlabels1::NTuple{QD1, NTuple{NZ, Int}},
                                          cgp1::NTuple{QD1, Int},
                                          legdir1::Tuple{Int, Int},
                                          qlabels2::NTuple{QD2, NTuple{NZ, Int}},
                                          cgp2::NTuple{QD2, Int},
                                          legdir2::Tuple{Int, Int},
                                          legs1::NTuple{CN, Int},
                                          legs2::NTuple{CN, Int}) where {S<:NonabelianSymm, QD1, QD2, NZ, CN}
    branches = Expr[]
    for m1 in 0:QD1
        for m2 in 0:QD2
            push!(branches, quote
                if legdir1[1] == $m1 && legdir2[1] == $m2
                    up1sp = _cgt_up_qlabels(qlabels1, Val($m1))
                    dn1sp = _cgt_dn_qlabels(qlabels1, Val($m1))
                    up2sp = _cgt_up_qlabels(qlabels2, Val($m2))
                    dn2sp = _cgt_dn_qlabels(qlabels2, Val($m2))
                    ctlegs1 = _stored_contract_legs(cgp1, legs1)
                    ctlegs2 = _stored_contract_legs(cgp2, legs2)
                    xsym_obj = getNsave_Xsymbol(S, up1sp, dn1sp, up2sp, dn2sp, ctlegs1, ctlegs2)
                    isnothing(xsym_obj) && return nothing
                    return xsym_obj.xsym_arr::Array{Float64, 3}
                end
            end)
        end
    end
    return quote
        $(branches...)
        throw(ArgumentError("invalid CGT leg directions"))
    end
end

@inline function _contract_wmat_for_symmetry(::Type{S},
                                             q1::TLArray,
                                             i1::Int,
                                             q2::TLArray,
                                             i2::Int,
                                             contracted_qlabels::NTuple{CN, QT},
                                             free_qlabels1::NTuple{NF1, QT},
                                             free_qlabels2::NTuple{NF2, QT},
                                             ::Nothing,
                                             legs1::NTuple{CN, Int},
                                             legs2::NTuple{CN, Int},
                                             ::Val{n}) where {S<:AbelianSymm, QT, NF1, NF2, CN, n}
    return _ABELIAN_WMAT_3D
end

@inline function _contract_wmat_for_symmetry(::Type{S},
                                             q1::TLArray,
                                             i1::Int,
                                             q2::TLArray,
                                             i2::Int,
                                             contracted_qlabels::NTuple{CN, QT},
                                             free_qlabels1::NTuple{NF1, QT},
                                             free_qlabels2::NTuple{NF2, QT},
                                             xsym_cache::_XSymCache,
                                             legs1::NTuple{CN, Int},
                                             legs2::NTuple{CN, Int},
                                             ::Val{n}) where {S<:NonabelianSymm, QT, NF1, NF2, CN, n}
    xkey = _xsym_cache_key(QT, contracted_qlabels, free_qlabels1, free_qlabels2, Val(n))
    qlabels1, cgp1, legdir1 = _sector_cgt_metadata(q1, i1, n)
    qlabels2, cgp2, legdir2 = _sector_cgt_metadata(q2, i2, n)
    xarr = if haskey(xsym_cache, xkey)
        xsym_cache[xkey]
    else
        loaded = _load_nonabelian_xarr(S, qlabels1, cgp1, legdir1,
                                       qlabels2, cgp2, legdir2, legs1, legs2)
        xsym_cache[xkey] = loaded
        loaded
    end
    isnothing(xarr) && return nothing
    return _contract_xsym_wmat(sector_wmat(q1, i1, n).data, xarr, sector_wmat(q2, i2, n).data)
end

@generated function _contract_wmats(::Type{ProductSymm{Syms}},
                                    q1::TLArray{T1, QD1, N, RD1, QT, PS},
                                    i1::Int,
                                    q2::TLArray{T2, QD2, N, RD2, QT, PS},
                                    i2::Int,
                                    contracted_qlabels::NTuple{CN, QT},
                                    free_qlabels1::NTuple{NF1, QT},
                                    free_qlabels2::NTuple{NF2, QT},
                                    xsym_caches::AbstractVector,
                                    legs1::NTuple{CN, Int},
                                    legs2::NTuple{CN, Int}) where {Syms, T1, QD1, N, RD1, QT, PS,
                                                                   T2, QD2, RD2, NF1, NF2, CN}
    stmts = Expr[]
    wnames = Symbol[]
    WMatT = :(LurTensor{Float64, 3, Array{Float64, 3}})
    for n in 1:N
        S = Syms.parameters[n]
        w = Symbol(:wmat_, n)
        push!(wnames, w)
        push!(stmts, quote
            local $w = _contract_wmat_for_symmetry($S, q1, i1, q2, i2,
                                                   contracted_qlabels,
                                                   free_qlabels1,
                                                   free_qlabels2,
                                                   _xsym_cache_for($S, xsym_caches, Val($n)),
                                                   legs1,
                                                   legs2,
                                                   Val($n))
            $w === nothing && return nothing
            $w = $w::$WMatT
        end)
    end
    return quote
        $(stmts...)
        return ($(wnames...),)
    end
end

# ── Convenience overloads ─────────────────────────────────────────────────────
contract(q1, l1::Int, q2, l2::Int) = contract(q1, (l1,), q2, (l2,))

function contract(q1::TLArray, legs1::AbstractVector{<:Integer},
                  q2::TLArray, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
end

# ── Main entry point ──────────────────────────────────────────────────────────
function contract(q1::TLArray{T1, QD1, N, RD1, QT, PS},
                  legs1::NTuple{CN, Int},
                  q2::TLArray{T2, QD2, N, RD2, QT, PS},
                  legs2::NTuple{CN, Int};
                  reduce_lock::Bool=true,
                  verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, QT, PS, CN}

    symmetries = product_symms(PS)

    if verify_legs
        for i in 1:CN
            idx1::TLIndex = q1.inds[legs1[i]::Int]
            idx2::TLIndex = q2.inds[legs2[i]::Int]
            @assert idx1.dir != idx2.dir "Contracted legs must have opposite arrow directions: " *
                "q1 leg $(legs1[i]) has dir='$(idx1.dir)', q2 leg $(legs2[i]) has dir='$(idx2.dir)'"
            @assert idx1.itags == idx2.itags "Contracted legs must have matching itags: " *
                "q1 leg $(legs1[i]) has itag='$(idx1.itags)', q2 leg $(legs2[i]) has itag='$(idx2.itags)'"
            @assert idx1.dual == idx2.dual "Contracted legs must have matching dual flags: " *
                "q1 leg $(legs1[i]) has dual=$(idx1.dual), q2 leg $(legs2[i]) has dual=$(idx2.dual)"
            @assert q1.spaces[legs1[i]] == q2.spaces[legs2[i]] "Contracted legs must have matching space info: " *
                "q1 leg $(legs1[i]) spaces != q2 leg $(legs2[i]) spaces"
        end
    end

    T    = promote_type(T1, T2)

    nf1 = QD1 - CN
    nf2 = QD2 - CN
    free1::NTuple{nf1, Int}  = _free_legs(Val(QD1), legs1)
    free2::NTuple{nf2, Int}  = _free_legs(Val(QD2), legs2)
    QD_out = QD1 + QD2 - 2CN
    RD_out = QD_out + N

    inds_out = (ntuple(i -> q1.inds[free1[i]], Val(QD1 - CN))...,
                ntuple(i -> q2.inds[free2[i]], Val(QD2 - CN))...)

    # Fixed permutations: (free..., contracted..., om...).
    perm1 = (free1..., legs1..., ntuple(n -> QD1 + n, Val(N))...)
    perm2 = (free2..., legs2..., ntuple(n -> QD2 + n, Val(N))...)

    permuted_rmts1 = Vector{Array{T1, 3}}(undef, nsectors(q1))
    permuted_rmts2 = Vector{Array{T2, 3}}(undef, nsectors(q2))

    # ── 1. Build sorted sector-info vectors keyed by contracted qlabels ─────────
    sector_infos1 = _contract_sector_infos(QT, q1, free1, legs1)
    sector_infos2 = _contract_sector_infos(QT, q2, free2, legs2)

    # ── 2. Output-sector accumulator ────────────────────────────────────────
    FreeKey1  = NTuple{nf1, QT}
    FreeKey2  = NTuple{nf2, QT}
    OutKey    = Tuple{FreeKey1, FreeKey2}

    # TODO: When GPU support is added, these should be generalized
    WMatT = LurTensor{Float64, 3, Array{Float64, 3}}
    WmatVec   = Vector{WMatT}
    SectorPairVec = Vector{Tuple{Int, Int}}

    xsym_caches = _new_xsym_caches(PS, Val(N))

    sector_wmats = Dict{OutKey, NTuple{N, WmatVec}}()
    sector_reps  = Dict{OutKey, SectorPairVec}()

    # ── 3. Main loop: collect pending RMT contractions per sector ────────────
    pos1 = firstindex(sector_infos1)
    pos2 = firstindex(sector_infos2)
    while pos1 <= lastindex(sector_infos1) && pos2 <= lastindex(sector_infos2)
        ckey1 = sector_infos1[pos1].contracted_qlabels
        ckey2 = sector_infos2[pos2].contracted_qlabels

        if isless(ckey1, ckey2)
            _, _, pos1 = _contracted_qlabel_run(sector_infos1, pos1)
            continue
        elseif isless(ckey2, ckey1)
            _, _, pos2 = _contracted_qlabel_run(sector_infos2, pos2)
            continue
        end

        _, run1, next_pos1 = _contracted_qlabel_run(sector_infos1, pos1)
        _, run2, next_pos2 = _contracted_qlabel_run(sector_infos2, pos2)
        for p1 in run1
            i = sector_infos1[p1].sector_index
            fq1 = sector_infos1[p1].free_qlabels::FreeKey1

            for p2 in run2
                j = sector_infos2[p2].sector_index
                fq2 = sector_infos2[p2].free_qlabels::FreeKey2

                # ── W-matrix contraction (per symmetry) ──────────────────────
                wmats = _contract_wmats(PS, q1, i, q2, j, ckey1, fq1, fq2,
                                        xsym_caches, legs1, legs2)
                isnothing(wmats) && continue

                # ── Accumulate into output sector ────────────────────────────
                out_key = (fq1, fq2)::OutKey
                if !haskey(sector_wmats, out_key)
                    sector_wmats[out_key] = ntuple(_ -> WMatT[], Val(N))
                    sector_reps[out_key]  = Tuple{Int, Int}[]
                end
                for n in 1:N push!(sector_wmats[out_key][n], wmats[n]) end
                push!(sector_reps[out_key], (i, j))
            end
        end
        pos1 = next_pos1
        pos2 = next_pos2
    end
    res_nsectors = length(sector_wmats)

    # ── 4. Prepare QR/K data and one contraction temporary ───────────────────
    PreparedSectorT = Tuple{Vector{LurTensor{Float64, 2, Matrix{Float64}}},
                            Vector{Array{Float64, 3}},
                            NTuple{N, Int}}
    prepared_sectors = Dict{OutKey, PreparedSectorT}()
    sizehint!(prepared_sectors, res_nsectors)
    max_temp_len = 0
    for (out_key, new_wmats) in sector_wmats
        prepared = _prepare_compress_sector(new_wmats)
        isnothing(prepared) && continue
        _, factor_arrays, _ = prepared

        sector_pairs = sector_reps[out_key]
        @assert length(sector_pairs) == length(factor_arrays)
        for i in eachindex(sector_pairs)
            idx1, idx2 = sector_pairs[i]
            fdim, cdim1, o1dim =
                _rmt_contract_dims(sector_rmt(q1, idx1), perm1, Val(nf1), Val(CN), Val(N))
            gdim, cdim2, o2dim =
                _rmt_contract_dims(sector_rmt(q2, idx2), perm2, Val(nf2), Val(CN), Val(N))
            @assert cdim1 == cdim2
            max_temp_len = max(max_temp_len,
                               _rmt_contract_temp_len(fdim, gdim, cdim1,
                                                      o1dim, o2dim,
                                                      factor_arrays[i]))
        end

        prepared_sectors[out_key] = prepared
    end
    contract_temp = Vector{promote_type(T1, T2, Float64)}(undef, max_temp_len)

    # ── 5. Merge each output sector ──────────────────────────────────────────
    result_qlabel_sectors = Vector{NTuple{QD_out, QT}}()
    result_wmat_buffers = _wmat_buffers(PS)
    result_RMTs = LurTensor{promote_type(T1, T2, Float64), RD_out, Array{promote_type(T1, T2, Float64), RD_out}}[]
    sizehint!(result_qlabel_sectors, res_nsectors)
    for slot in eachindex(result_wmat_buffers)
        sizehint!(result_wmat_buffers[slot], res_nsectors)
    end
    sizehint!(result_RMTs, res_nsectors)
    for (out_key, prepared) in prepared_sectors
        sector_pairs = sector_reps[out_key]
        r1_idx, r2_idx = first(sector_pairs)
        new_qlabels_per_n = get_new_cgt_metadata(
            q1, r1_idx, q2, r2_idx, free1, free2, legs1, legs2)

        sz1::NTuple{RD1, Int}, sz2::NTuple{RD2, Int} =
        size(sector_rmt(q1, r1_idx)), size(sector_rmt(q2, r2_idx))

        kept_sizes1 = ntuple(i -> sz1[perm1[i]], Val(nf1))
        kept_sizes2 = ntuple(i -> sz2[perm2[i]], Val(nf2))
        for (idx1, idx2) in sector_pairs
            _cached_permuted_rmt_data!(permuted_rmts1, q1, idx1,
                                       perm1, Val(nf1), Val(CN), Val(N))
            _cached_permuted_rmt_data!(permuted_rmts2, q2, idx2,
                                       perm2, Val(nf2), Val(CN), Val(N))
        end

        U_mats, result_RMT = _contract_prepared_compress_sector(
            prepared, sector_pairs, permuted_rmts1, permuted_rmts2,
            kept_sizes1, kept_sizes2, contract_temp, Val(RD_out))
        push!(result_qlabel_sectors,
              _physical_qlabels_from_cgt_metadata(QT, new_qlabels_per_n, Val(QD_out), Val(N)))
        for n in 1:N
            _push_wmat!(result_wmat_buffers, PS, n, U_mats[n])
        end
        push!(result_RMTs, result_RMT)
    end

    # ── 6. Lock reduction ────────────────────────────────────────────────────
    changed_inds2 = Set(change_dir(q2.inds[l]) for l in 1:QD2)
    changed_inds1 = Set(change_dir(q1.inds[l]) for l in 1:QD1)
    final_inds = if reduce_lock
        ntuple(Val(QD_out)) do l
            idx = inds_out[l]
            has_match = l <= nf1 ? (idx ∈ changed_inds2) : (idx ∈ changed_inds1)
            (idx.lock > 0 && has_match) ?
                TLIndex(idx.itags, idx.dir, idx.plev, idx.lock - 1, idx.dual) : idx
        end
    else
        inds_out
    end

    spaces_out = (ntuple(i -> q1.spaces[free1[i]], Val(QD1 - CN))...,
                  ntuple(i -> q2.spaces[free2[i]], Val(QD2 - CN))...)

    result_qlabels = Matrix{QT}(undef, QD_out, length(result_qlabel_sectors))
    for sector_index in eachindex(result_qlabel_sectors), leg in 1:QD_out
        result_qlabels[leg, sector_index] = result_qlabel_sectors[sector_index][leg]
    end

    result_wmats = _wmat_matrix_from_buffers(PS, result_wmat_buffers, length(result_RMTs))
    return _field_tlarray(symmetries, result_qlabels, result_wmats, result_RMTs,
                          final_inds, spaces_out)::TLArray{promote_type(T1, T2, Float64), QD_out, N, RD_out, QT, PS}
end
