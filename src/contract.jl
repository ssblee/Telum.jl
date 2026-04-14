# ─── Helpers ─────────────────────────────────────────────────────────────────

# Physical qlabel of leg l in row r: an N-tuple of qlabels, one per symmetry.
function _row_qlabel(r::row{T, QD, N}, l::Int) where {T, QD, N}
    Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[l]] for n in 1:N)
end

# Sort key: (free_leg_qlabels..., contr_leg_qlabels...)
# Free legs first so rows with the same output sector are grouped together;
# contracted legs second so within a group rows are ordered by charge sector
# (enabling an efficient two-pointer sweep when pairing with the other QSpace).
function _contract_sort_key(r::row, free_legs, contr_legs)
    (Tuple(_row_qlabel(r, l) for l in free_legs),
     Tuple(_row_qlabel(r, l) for l in contr_legs))
end

function sort_rows_for_contract!(rows::Vector{<:row}, free_legs, contr_legs)
    sort!(rows; by = r -> _contract_sort_key(r, free_legs, contr_legs))
end

# ─── SectorMap ───────────────────────────────────────────────────────────────
# Maps the qlabels of the *free* (non-contracted) legs of a sorted row vector
# to the list of (contr_qlabels, row_index) pairs in that free-sector.
# Because rows are sorted by (free_qlabels, contr_qlabels), entries within
# each sector are already ordered by contracted qlabels.

struct SectorMap{FreeKey, ContrKey}
    # free_key → [(contr_key, row_index), ...]
    data::Dict{FreeKey, Vector{Tuple{ContrKey, Int}}}
end

function build_sector_map(rows::Vector{<:row{T, QD, N}},
                          free_legs, contr_legs) where {T, QD, N}
    FreeKey  = NTuple{length(free_legs),  NTuple{N, Tuple{Vararg{Int}}}}
    ContrKey = NTuple{length(contr_legs), NTuple{N, Tuple{Vararg{Int}}}}
    d = Dict{FreeKey, Vector{Tuple{ContrKey, Int}}}()
    for (i, r) in enumerate(rows)
        fkey = Tuple(_row_qlabel(r, l) for l in free_legs)
        ckey = Tuple(_row_qlabel(r, l) for l in contr_legs)
        push!(get!(d, fkey, Tuple{ContrKey, Int}[]), (ckey, i))
    end
    return SectorMap{FreeKey, ContrKey}(d)
end

# ─── RMT contraction helper ──────────────────────────────────────────────────
# Takes two pre-permuted RMTs and contracts over the contracted-leg axes.
# P1 shape: (free1..., om1_1,...,om1_N, contr...)   [nf1 + N + CN axes]
# P2 shape: (free2..., om2_1,...,om2_N, contr...)   [nf2 + N + CN axes]
# Result shape: (free1..., free2..., om1_1*om2_1, ..., om1_N*om2_N)  [nf1+nf2+N axes]
function _contract_RMTs(P1::AbstractArray, P2::AbstractArray,
                         nf1::Int, nf2::Int, N::Int, CN::Int)
    sz_f1  = [size(P1, i)      for i in 1:nf1]
    sz_om1 = [size(P1, nf1+n)  for n in 1:N]
    sz_c   = [size(P1, nf1+N+k) for k in 1:CN]
    sz_f2  = [size(P2, i)      for i in 1:nf2]
    sz_om2 = [size(P2, nf2+n)  for n in 1:N]

    R1_mat = reshape(P1, prod(sz_f1; init=1) * prod(sz_om1; init=1), prod(sz_c; init=1))
    R2_mat = reshape(P2, prod(sz_f2; init=1) * prod(sz_om2; init=1), prod(sz_c; init=1))
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

# ─── CGTContrInfo ────────────────────────────────────────────────────────────
# Holds the CGT space/leg data needed to call getNsave_Xsymbol for one
# non-abelian symmetry in a matched (r1, r2) row pair.
struct CGTContrInfo{S<:NonabelianSymm, U1, D1, U2, D2, NZ, M}
    up1sp::NTuple{U1, NTuple{NZ, Int}}   # incoming qlabels of CGT1
    dn1sp::NTuple{D1, NTuple{NZ, Int}}   # outgoing qlabels of CGT1
    up2sp::NTuple{U2, NTuple{NZ, Int}}   # incoming qlabels of CGT2
    dn2sp::NTuple{D2, NTuple{NZ, Int}}   # outgoing qlabels of CGT2
    ctlegs1::NTuple{M, Int}              # contracted stored-qlabel indices in CGT1
    ctlegs2::NTuple{M, Int}              # contracted stored-qlabel indices in CGT2
end

# Build CGTContrInfo for symmetry index n from a matched row pair.
# phys_legs1 / phys_legs2 are the physical leg indices (1-based in q1/q2)
# that are being contracted; cgp maps them to stored qlabel positions in the CGR.
@inline _symmetry_type(::Type{<:ProductSymm{Syms}}, ::Val{n}) where {Syms, n} =
    Syms.parameters[n]

get_cgt_contr_info(::Type{S}, r1::row, r2::row,
                   phys_legs1::NTuple{CN, Int}, phys_legs2::NTuple{CN, Int},
                   ::Val{n}) where {S<:AbelianSymm, CN, n} = nothing

function get_cgt_contr_info(::Type{S}, r1::row, r2::row,
                            phys_legs1::NTuple{CN, Int},
                            phys_legs2::NTuple{CN, Int},
                            ::Val{n}) where {S<:NonabelianSymm, CN, n}
    cgr1 = r1.cgrs[n];  cgr2 = r2.cgrs[n]
    m1, k1 = cgr1.legdir;  m2, k2 = cgr2.legdir
    up1sp  = Tuple(cgr1.qlabels[i] for i in 1:m1)
    dn1sp  = Tuple(cgr1.qlabels[i] for i in m1+1:m1+k1)
    up2sp  = Tuple(cgr2.qlabels[i] for i in 1:m2)
    dn2sp  = Tuple(cgr2.qlabels[i] for i in m2+1:m2+k2)
    # Map physical contracted legs → stored-qlabel positions via cgp.
    ctlegs1 = ntuple(k -> cgr1.cgp[phys_legs1[k]], Val(CN))
    ctlegs2 = ntuple(k -> cgr2.cgp[phys_legs2[k]], Val(CN))
    NZ = length(cgr1.qlabels[1])
    return CGTContrInfo{S, m1, k1, m2, k2, NZ, CN}(
        up1sp, dn1sp, up2sp, dn2sp, ctlegs1, ctlegs2)
end

function get_cgt_contr_info(r1::row, r2::row, phys_legs1, phys_legs2, n::Int, symm)
    S = symm[n]
    isabelian(S) && return nothing
    cgr1 = r1.cgrs[n];  cgr2 = r2.cgrs[n]
    m1, k1 = cgr1.legdir;  m2, k2 = cgr2.legdir
    up1sp  = Tuple(cgr1.qlabels[i] for i in 1:m1)
    dn1sp  = Tuple(cgr1.qlabels[i] for i in m1+1:m1+k1)
    up2sp  = Tuple(cgr2.qlabels[i] for i in 1:m2)
    dn2sp  = Tuple(cgr2.qlabels[i] for i in m2+1:m2+k2)
    ctlegs1 = Tuple(cgr1.cgp[l] for l in phys_legs1)
    ctlegs2 = Tuple(cgr2.cgp[l] for l in phys_legs2)
    NZ = length(cgr1.qlabels[1])
    M  = length(ctlegs1)
    return CGTContrInfo{S, m1, k1, m2, k2, NZ, M}(
        up1sp, dn1sp, up2sp, dn2sp, ctlegs1, ctlegs2)
end

# ─── Output CGR metadata ─────────────────────────────────────────────────────

# Core plain-data function: given two CGTs' qlabel/legdir/cgp data and the
# physical free/contracted legs, compute the resulting CGR's qlabels, cgp,
# and legdir.  This function is independent of the row/CGR structs.
#
# Arguments (one symmetry at a time):
#   qlabels1/2  : NTuple{QD, NTuple{NZ,Int}} — stored qlabels of each CGT
#   legdir1/2   : (m, k) — number of incoming / outgoing stored qlabels
#   cgp1/2      : NTuple{QD, Int} — physical leg → stored qlabel index
#   free1/2     : physical free leg indices (1-based in each source QSpace)
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

    for (i, l_in) in enumerate(free1)
        sp = cgp1[l_in]
        sp ∈ ctset1 && continue
        (sp <= m1 ? up3 : dn3) |> x -> push!(x, (qlabels1[sp], i))
    end
    for (i, l_in) in enumerate(free2)
        sp = cgp2[l_in]
        sp ∈ ctset2 && continue
        (sp <= m2 ? up3 : dn3) |> x -> push!(x, (qlabels2[sp], i + NF1))
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

# Wraps get_new_cgp across all N symmetries for a matched row pair.
function get_new_cgr_metadata(r1_rep::row{T1, QD1, N}, r2_rep::row{T2, QD2, N},
                               free1, free2, legs1, legs2) where {T1, QD1, T2, QD2, N}
    ntuple(Val(N)) do n
        cgr1n = r1_rep.cgrs[n]; cgr2n = r2_rep.cgrs[n]
        get_new_cgp(cgr1n.qlabels, cgr1n.legdir, cgr1n.cgp, free1, legs1,
                    cgr2n.qlabels, cgr2n.legdir, cgr2n.cgp, free2, legs2)
    end
end

# ─── merge_new_row helpers ───────────────────────────────────────────────────

# Mode-product: replace axis `axis` of array A (size k) with the output of
# multiplying matrix M (shape r×k) along that axis.
# Equivalent to: result[..., j, ...] = Σ_l M[j,l] * A[...,l,...] at `axis`.
function _contract_om_axis(A::AbstractArray{T}, M::AbstractMatrix, axis::Int) where T
    dims  = size(A)
    k     = dims[axis]
    r     = size(M, 1)
    @assert size(M, 2) == k "axis size $(k) ≠ M columns $(size(M, 2))"

    prod_before = prod(dims[1:axis-1]; init = 1)
    prod_after  = prod(dims[axis+1:end]; init = 1)

    # Reshape A to (prod_before, k, prod_after); move k to front; multiply; restore.
    A_3  = reshape(A, prod_before, k, prod_after)
    A_km = reshape(permutedims(A_3, (2, 1, 3)), k, prod_before * prod_after)
    R_km = M * A_km                                       # (r, prod_before * prod_after)
    R_3  = permutedims(reshape(R_km, r, prod_before, prod_after), (2, 1, 3))
    return reshape(R_3, dims[1:axis-1]..., r, dims[axis+1:end]...)
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
#   U_mats[n]  : LurTensor{Float64,2} (OM3_n, r_n)  — new CGR wmat per symmetry
#   result_RMT : LurTensor{T} (sz_free..., r_1,...,r_N)  — compressed RMT
#   nothing    : when any symmetry contributes only zero w-matrices, so the
#                whole merged row is identically zero and can be skipped
#
# We use QR-based shared isometries for every sector, including K == 1, so the
# resulting basis is normalized consistently with the multi-contribution case.
function _compress_sector(
    new_wmats ::NTuple{N, Vector{<:LurTensor{Float64, 2}}},
    new_RMTs  ::Vector{<:LurTensor{T, RD}},
    QD_out    ::Int,
    tol       ::Float64 = 1e-12,
) where {T, N, RD}
    K = length(new_RMTs)

    # ── Shared QR basis per symmetry ─────────────────────────────────────────
    U_mats   = Vector{LurTensor{Float64, 2}}(undef, N)
    SV_split = [Vector{Matrix{Float64}}(undef, K) for _ in 1:N]

    for n in 1:N
        mats = [w.data for w in new_wmats[n]]
        if all(mat -> all(iszero, mat), mats) return nothing end

        common_iso, factors = _qr_shared_isometry(mats; tol=tol)
        U_mats[n] = LurTensor(common_iso)
        for i in 1:K
            SV_split[n][i] = factors[i]
        end
    end

    # ── Preallocate output RMT as LurTensor (sz_free..., r_1,...,r_N) ──────────
    sz_free    = size(new_RMTs[1])[1:QD_out]
    r_sizes    = ntuple(n -> size(U_mats[n], 2), N)
    result_RMT = LurTensor{T}(sz_free..., r_sizes...)

    # ── Contract SV pieces into each RMT and accumulate ─────────────────────
    for i in 1:K
        contrib = new_RMTs[i].data          # plain Array{T, RD} for _contract_om_axis
        for n in 1:N
            contrib = _contract_om_axis(contrib, SV_split[n][i], QD_out + n)
        end
        result_RMT .+= contrib
    end

    return U_mats, result_RMT
end

# ─── merge_new_row ────────────────────────────────────────────────────────────
# Wraps _compress_sector and assembles the output row struct.
function merge_new_row(
    new_wmats        ::NTuple{N, Vector{<:LurTensor{Float64, 2}}},
    new_RMTs         ::Vector{<:LurTensor{T, RD}},
    new_qlabels_per_n,
    ::Type{PS},
    QD_out           ::Int,
    tol              ::Float64 = 1e-12,
) where {T, N, RD, PS<:ProductSymm}
    compressed = _compress_sector(new_wmats, new_RMTs, QD_out, tol)
    isnothing(compressed) && return nothing
    U_mats, result_RMT = compressed
    # U_mats[n] and result_RMT are already LurTensor; no re-wrapping needed.
    cgrs_new = ntuple(Val(N)) do n
        new_ql, new_cgp, new_ld = new_qlabels_per_n[n]
        CGR(_symmetry_type(PS, Val(n)), new_ql, U_mats[n], new_cgp, new_ld)
    end

    return row(Tuple(cgrs_new), result_RMT)
end

function merge_new_row(
    new_wmats        ::NTuple{N, Vector{<:LurTensor{Float64, 2}}},
    new_RMTs         ::Vector{<:LurTensor{T, RD}},
    new_qlabels_per_n,
    symm::NTuple{N, Any},
    QD_out           ::Int,
    tol              ::Float64 = 1e-12,
) where {T, N, RD}
    return merge_new_row(new_wmats, new_RMTs, new_qlabels_per_n,
                         productsymm(symm), QD_out, tol)
end


# ─── contract_old ────────────────────────────────────────────────────────────

contract_old(q1, l1::Int, q2, l2::Int) = contract_old(q1, (l1,), q2, (l2,))

# Vector / LegList overload: convert to tuples and delegate to the NTuple method.
function contract_old(q1::QSpace, legs1::AbstractVector{<:Integer},
                      q2::QSpace, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract_old(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
end

# ─── * operator ──────────────────────────────────────────────────────────────
# Automatically contract two QSpace objects by matching their tagged, unlocked
# indices.  An index on q1 is "contractible" when it has a nonempty tag AND
# lock == 0; same criterion applies to q2.  Two contractible indices are matched
# when they compare equal under QIndex == (same itags, dir, plev, green) and
# their precomputed leg spaces are equal. The collected matching pairs define
# legs1 / legs2 passed to `contract`.
function Base.:*(q1::QSpace, q2::QSpace)
    # Collect candidate indices from each QSpace.
    cands1 = [(i, q1.inds[i]) for i in 1:length(q1.inds)
              if !isempty(q1.inds[i].itags) && q1.inds[i].lock == 0]
    cands2 = [(j, q2.inds[j]) for j in 1:length(q2.inds)
              if !isempty(q2.inds[j].itags) && q2.inds[j].lock == 0]

    # Match candidates: for each index in cands1, find the unique equal index
    # in cands2.  Raise an error if a tag appears more than once on either side.
    legs1 = Int[]
    legs2 = Int[]
    matched2 = Set{Int}()   # positions in cands2 already consumed

    for (i, idx1) in cands1
        hits = [(pos, j, idx2) for (pos, (j, idx2)) in enumerate(cands2)
                if idx1 == change_dir(idx2) &&
                   q1.spaces[i] == q2.spaces[j] &&
                   pos ∉ matched2]
        if length(hits) > 1
            error("Ambiguous contraction: tag \"$(idx1.itags)\" matches more than one index in q2")
        end
        if length(hits) == 1
            pos, j, _ = hits[1]
            push!(legs1, i)
            push!(legs2, j)
            push!(matched2, pos)
        end
    end

    @assert length(legs1) > 0 "No matching contractible indices found between the two QSpace objects"

    return contract(q1, Tuple(legs1), q2, Tuple(legs2); verify_legs=false)
end

function contract_old(q1::QSpace{T1, QD1, N, RD1},
                      legs1::NTuple{CN, Int},
                      q2::QSpace{T2, QD2, N, RD2},
                      legs2::NTuple{CN, Int};
                      reduce_lock::Bool=true,
                      verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, CN}

    symmetries = symm(q1)
    @assert symmetries == symm(q2) "QSpace objects must share the same symmetry tuple"
    
    # Verify contracted legs have opposite arrow directions, matching itags/green, and same space info
    if verify_legs
        for i in 1:CN
            idx1 = q1.inds[legs1[i]]
            idx2 = q2.inds[legs2[i]]
            @assert idx1.dir != idx2.dir "Contracted legs must have opposite arrow directions: " *
                "q1 leg $(legs1[i]) has dir='$(idx1.dir)', q2 leg $(legs2[i]) has dir='$(idx2.dir)'"
            @assert idx1.itags == idx2.itags "Contracted legs must have matching itags: " *
                "q1 leg $(legs1[i]) has itag='$(idx1.itags)', q2 leg $(legs2[i]) has itag='$(idx2.itags)'"
            @assert idx1.green == idx2.green "Contracted legs must have matching green flags: " *
                "q1 leg $(legs1[i]) has green=$(idx1.green), q2 leg $(legs2[i]) has green=$(idx2.green)"
            @assert q1.spaces[legs1[i]] == q2.spaces[legs2[i]] "Contracted legs must have matching space info: " *
                "q1 leg $(legs1[i]) spaces != q2 leg $(legs2[i]) spaces"
        end
    end
    
    T    = promote_type(T1, T2)

    free1  = [l for l in 1:QD1 if l ∉ legs1]
    free2  = [l for l in 1:QD2 if l ∉ legs2]
    nf1, nf2 = length(free1), length(free2)
    QD_out = nf1 + nf2
    RD_out = QD_out + N    # N merged OM axes (om1_n * om2_n per symmetry)

    # Arrow directions of result legs: free legs keep their direction from the source.
    inds_out = Tuple([[q1.inds[l] for l in free1]; [q2.inds[l] for l in free2]])

    # Pre-compute fixed permutations for all RMTs (depends only on legs, not row data).
    perm1 = [free1; collect(QD1+1:QD1+N); collect(legs1)]  # → (f1..., om1..., c...)
    perm2 = [free2; collect(QD2+1:QD2+N); collect(legs2)]  # → (f2..., om2..., c...)

    # Sort both row vectors: non-contracted legs first, then contracted.
    rows1 = sort(q1.rows; by = r -> _contract_sort_key(r, free1, collect(legs1)))
    rows2 = sort(q2.rows; by = r -> _contract_sort_key(r, free2, collect(legs2)))

    # Pre-permute all RMTs before the matching loop.
    permed1 = [permutedims(r.RMT.data, perm1) for r in rows1]
    permed2 = [permutedims(r.RMT.data, perm2) for r in rows2]

    # Build SectorMaps: free_qlabels → [(contr_qlabels, row_idx), ...]
    sm1 = build_sector_map(rows1, free1, collect(legs1))
    sm2 = build_sector_map(rows2, free2, collect(legs2))

    # output_key = (fq1, fq2); accumulate ContrEntry list per output sector.
    result_rows = Vector{row{T, QD_out, N, RD_out}}()

    for (fq1, v1) in sm1.data
        for (fq2, v2) in sm2.data
            new_wmats = ntuple(_ -> Vector{LurTensor{Float64, 2}}(), N)
            new_RMTs  = Vector{LurTensor{T, RD_out}}()
            
            # Two-pointer merge on ckey (both v1, v2 sorted by ckey from the sort above).
            i, j = 1, 1
            while i <= length(v1) && j <= length(v2)
                (ckey1, idx1) = v1[i]
                (ckey2, idx2) = v2[j]

                if ckey1 < ckey2 i += 1
                elseif ckey1 > ckey2 j += 1
                else
                    # Unique match: exactly one row on each side per ckey.
                    r1, r2 = rows1[idx1], rows2[idx2]


                    # 1. For each symmetry, contract X-symbol with the two w-matrices
                    #    to get the new w-matrix: result_w[a,b,c] = sum_{bb,cc} X[bb,cc,a] * wmat1[bb,b] * wmat2[cc,c]
                    #    Efficient form: result_w[a,:,:] = wmat1' * X[:,:,a] * wmat2
                    zero_xsym = false
                    wmats = Vector{LurTensor{Float64, 2}}(undef, N)
                    for n in 1:N
                        cgr1n = r1.cgrs[n];  cgr2n = r2.cgrs[n]
                        wm1 = cgr1n.wmat.data   # (OM1, d1)
                        wm2 = cgr2n.wmat.data   # (OM2, d2)
                        info = get_cgt_contr_info(r1, r2, legs1, legs2, n, symmetries)
                        if info === nothing
                            # Abelian: effective X is [[[1.0]]], OM1=OM2=OM3=1
                            result_w = reshape(wm1' * wm2, 1, size(wm1,2), size(wm2,2))
                        else
                            xsym_obj = getNsave_Xsymbol(symmetries[n],
                                                        info.up1sp, info.dn1sp,
                                                        info.up2sp, info.dn2sp,
                                                        info.ctlegs1, info.ctlegs2)
                            if isnothing(xsym_obj) zero_xsym = true; break end
                            xarr = xsym_obj.xsym_arr   # (OM1, OM2, OM3)
                            OM3, d1, d2 = size(xarr, 3), size(wm1, 2), size(wm2, 2)
                            result_w = zeros(Float64, OM3, d1, d2)
                            for a in 1:OM3
                                result_w[a, :, :] = wm1' * xarr[:, :, a] * wm2
                            end
                        end
                        wmats[n] = LurTensor(reshape(result_w, size(result_w, 1), :))
                    end
                    
                    if !zero_xsym
                        for n in 1:N push!(new_wmats[n], wmats[n]) end

                        # 2. Contract pre-permuted RMTs; OM pairs merged into N axes.
                        contr_RMT = _contract_RMTs(permed1[idx1], permed2[idx2],
                                                nf1, nf2, N, CN)
                        push!(new_RMTs, LurTensor(contr_RMT))
                    end

                    i += 1; j += 1
                end
            end

            # If there exists at least one matched pair of rows, compute new CGR metadata.
            # Free-leg qlabels are identical for every row in this (fq1, fq2) sector,
            # so we derive qlabels / cgp / legdir once from the first row of each group.
            if !isempty(new_RMTs)
                r1_rep = rows1[v1[1][2]]; r2_rep = rows2[v2[1][2]]
                new_qlabels_per_n = get_new_cgr_metadata(
                    r1_rep, r2_rep, free1, free2, legs1, legs2)
                new_row = merge_new_row(new_wmats, new_RMTs, new_qlabels_per_n,
                                        symmetries, QD_out)
                isnothing(new_row) || push!(result_rows, new_row)
            end
        end
    end

    # Reduce lock level of a free leg by 1 (floor 0) only when reduce_lock is
    # enabled AND the other tensor has a leg that matches it under change_dir
    # (i.e. idx == change_dir(other_leg): same itags/plev/green/lock, opposite dir).
    # A leg with no such counterpart had no opportunity to be contracted and must
    # keep its current lock level unchanged.
    changed_inds2 = Set(change_dir(q2.inds[l]) for l in 1:QD2)
    changed_inds1 = Set(change_dir(q1.inds[l]) for l in 1:QD1)
    final_inds = if reduce_lock
        ntuple(length(inds_out)) do l
            idx = inds_out[l]
            has_match = l <= nf1 ? (idx ∈ changed_inds2) : (idx ∈ changed_inds1)
            (idx.lock > 0 && has_match) ?
                QIndex(idx.itags, idx.dir, idx.plev, idx.lock - 1, idx.green) : idx
        end
    else
        inds_out
    end

    # Compute spaces for result: free legs from q1 followed by free legs from q2
    spaces_out = ([q1.spaces[l] for l in free1]..., [q2.spaces[l] for l in free2]...)

    return QSpace(symmetries, result_rows, final_inds, spaces_out)
end


# ─── contract ────────────────────────────────────────────────────────────────
# Optimised contraction that sorts rows by *contracted* qlabels first and
# performs a single batched GEMM per common contracted sector, replacing many
# small matrix multiplies with one large one.
#
# Algorithm sketch:
#   1. Build sorted row-info vectors for each QSpace. Each entry carries the
#      row index, all physical-leg qlabels, and the contracted-qlabel key.
#   2. Two-pointer scan over both sorted vectors and process common sectors:
#      a) Pre-permute & reshape each row's RMT to (F·OM, C) matrix.
#      b) Vcat all q1 matrices → big_A;  vcat all q2 matrices → big_B.
#      c) big_C = big_A * big_B'            ← single BLAS call
#      d) Split big_C into per-(i, j) blocks and post-process each block
#         (reshape, permute, merge-OM, w-matrix / X-symbol contraction).
#      e) Accumulate results per output free-sector.
#   3. Merge each output sector (SVD compression → output row).
#   4. Lock reduction / build result QSpace.
# The legacy implementation remains available as `contract_old` for tests.

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

function _row_qlabel(::Type{QT}, r::row{T, QD, N}, l::Int) where {QT, T, QD, N}
    return ntuple(n -> r.cgrs[n].qlabels[r.cgrs[n].cgp[l]], Val(N))::QT
end

function _row_qlabels(::Type{QT}, r::row{T, QD, N}) where {QT, T, QD, N}
    return ntuple(l -> _row_qlabel(QT, r, l), Val(QD))::NTuple{QD, QT}
end

_contracted_qlabel_type(::Type{QT}, ::Val{1}) where {QT} = QT
_contracted_qlabel_type(::Type{QT}, ::Val{CN}) where {QT, CN} = NTuple{CN, QT}

_contracted_qlabel(::Type{QT}, r::row, legs::NTuple{1, Int}) where {QT} =
    _row_qlabel(QT, r, legs[1])

_contracted_qlabel(::Type{QT}, r::row, legs::NTuple{CN, Int}) where {QT, CN} =
    ntuple(i -> _row_qlabel(QT, r, legs[i]), Val(CN))

_contracted_qlabel(qlabels::NTuple{QD, QT}, legs::NTuple{1, Int}) where {QD, QT} =
    qlabels[legs[1]]

_contracted_qlabel(qlabels::NTuple{QD, QT}, legs::NTuple{CN, Int}) where {QD, QT, CN} =
    ntuple(i -> qlabels[legs[i]], Val(CN))

function _free_qlabels(qlabels::NTuple{QD, QT},
                       free_legs::NTuple{NF, Int}) where {QD, QT, NF}
    return ntuple(k -> qlabels[free_legs[k]], Val(NF))::NTuple{NF, QT}
end

struct ContractRowInfo{QD, QT, CQT}
    row_index::Int
    qlabels::NTuple{QD, QT}
    ckey::CQT
end

function _contract_row_infos(::Type{QT}, rows::AbstractVector{<:row{T, QD, N}},
                             legs::NTuple{CN, Int}) where {QT, T, QD, N, CN}
    CQT = _contracted_qlabel_type(QT, Val(CN))
    Info = ContractRowInfo{QD, QT, CQT}
    infos = Vector{Info}(undef, length(rows))
    for i in eachindex(rows)
        qlabels = _row_qlabels(QT, rows[i])
        infos[i] = Info(i, qlabels, _contracted_qlabel(qlabels, legs)::CQT)
    end
    return sort!(infos; by = info -> info.ckey, alg=MergeSort)
end

function _contracted_qlabel_entries(::Type{QT}, rows::AbstractVector{<:row},
                                    legs::NTuple{CN, Int}) where {QT, CN}
    infos = _contract_row_infos(QT, rows, legs)
    return [(info.row_index, info.ckey) for info in infos]
end

function _contracted_qlabel_run(infos::AbstractVector{<:ContractRowInfo{QD, QT, CQT}},
                                first_pos::Int) where {QD, QT, CQT}
    key = infos[first_pos].ckey
    next_pos = first_pos + 1
    while next_pos <= lastindex(infos) && infos[next_pos].ckey == key
        next_pos += 1
    end
    return key, first_pos:(next_pos - 1), next_pos
end

function _contract_wmat_for_symmetry(::Type{PS}, ::Val{n},
                                     r1::row, r2::row,
                                     legs1::NTuple{CN, Int},
                                     legs2::NTuple{CN, Int}) where {PS<:ProductSymm, n, CN}
    S = _symmetry_type(PS, Val(n))
    cgr1n = r1.cgrs[n];  cgr2n = r2.cgrs[n]
    wm1 = cgr1n.wmat.data
    wm2 = cgr2n.wmat.data
    info = get_cgt_contr_info(S, r1, r2, legs1, legs2, Val(n))

    if info === nothing
        @assert size(wm1) == (1, 1) "Abelian contraction expects q1 w-matrix to be 1x1, got size $(size(wm1))"
        @assert size(wm2) == (1, 1) "Abelian contraction expects q2 w-matrix to be 1x1, got size $(size(wm2))"
        @assert isapprox(wm1[1, 1], 1.0; atol=1e-12, rtol=1e-12) "Abelian contraction expects q1 w-matrix ≈ [1;;], got $(wm1)"
        @assert isapprox(wm2[1, 1], 1.0; atol=1e-12, rtol=1e-12) "Abelian contraction expects q2 w-matrix ≈ [1;;], got $(wm2)"
        return LurTensor([1.0;;])
    end

    xsym_obj = getNsave_Xsymbol(S,
                                info.up1sp, info.dn1sp,
                                info.up2sp, info.dn2sp,
                                info.ctlegs1, info.ctlegs2)
    isnothing(xsym_obj) && return nothing
    xarr = xsym_obj.xsym_arr
    OM3, d1, d2 = size(xarr, 3), size(wm1, 2), size(wm2, 2)
    result_w = zeros(Float64, OM3, d1, d2)
    for a in 1:OM3
        result_w[a, :, :] = wm1' * xarr[:, :, a] * wm2
    end
    return LurTensor(reshape(result_w, size(result_w, 1), :))
end

function _contract_wmats(::Type{PS}, r1::row{T1, QD1, N}, r2::row{T2, QD2, N},
                         legs1::NTuple{CN, Int},
                         legs2::NTuple{CN, Int}) where {PS<:ProductSymm, T1, QD1, T2, QD2, N, CN}
    maybe_wmats = ntuple(n -> _contract_wmat_for_symmetry(PS, Val(n), r1, r2,
                                                          legs1, legs2), Val(N))
    any(isnothing, maybe_wmats) && return nothing
    return ntuple(n -> maybe_wmats[n]::LurTensor{Float64, 2}, Val(N))
end

# ── Post-process helper ──────────────────────────────────────────────────────
# Reshape a (F1·OM1, F2·OM2) block from the batched matmul back to the
# standard contracted-RMT layout (f1…, f2…, om12_1, …, om12_N).
function _reshape_contract_block(block::AbstractMatrix,
                                  sz_f1::NTuple{NF1, Int},
                                  sz_om1::NTuple{N, Int},
                                  sz_f2::NTuple{NF2, Int},
                                  sz_om2::NTuple{N, Int}) where {NF1, NF2, N}
    # (F1·OM1, F2·OM2) → (f1…, om1…, f2…, om2…)
    C = reshape(block, sz_f1..., sz_om1..., sz_f2..., sz_om2...)

    # Permute to (f1…, f2…, om1_1, om2_1, …, om1_N, om2_N)
    composed_perm = ntuple(Val(NF1 + NF2 + 2N)) do p
        if p <= NF1
            p
        elseif p <= NF1 + NF2
            NF1 + N + (p - NF1)
        else
            t = p - NF1 - NF2
            isodd(t) ? NF1 + ((t + 1) ÷ 2) : NF1 + N + NF2 + (t ÷ 2)
        end
    end
    C_i = permutedims(C, composed_perm)

    # Merge each (om1_n, om2_n) pair → om1_n · om2_n
    sz_om_merged = ntuple(n -> sz_om1[n] * sz_om2[n], Val(N))
    return reshape(C_i, sz_f1..., sz_f2..., sz_om_merged...)
end

# ── Convenience overloads ─────────────────────────────────────────────────────
contract(q1, l1::Int, q2, l2::Int) = contract(q1, (l1,), q2, (l2,))

function contract(q1::QSpace, legs1::AbstractVector{<:Integer},
                  q2::QSpace, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
end

# ── Main entry point ──────────────────────────────────────────────────────────
function contract(q1::QSpace{T1, QD1, N, RD1, QT, PS1},
                  legs1::NTuple{CN, Int},
                  q2::QSpace{T2, QD2, N, RD2, QT, PS2},
                  legs2::NTuple{CN, Int};
                  reduce_lock::Bool=true,
                  verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, QT, PS1, PS2, CN}

    @assert PS1 === PS2 "QSpace objects must share the same symmetry tuple"
    symmetries = symm(q1)

    if verify_legs
        for i in 1:CN
            idx1 = q1.inds[legs1[i]]
            idx2 = q2.inds[legs2[i]]
            @assert idx1.dir != idx2.dir "Contracted legs must have opposite arrow directions: " *
                "q1 leg $(legs1[i]) has dir='$(idx1.dir)', q2 leg $(legs2[i]) has dir='$(idx2.dir)'"
            @assert idx1.itags == idx2.itags "Contracted legs must have matching itags: " *
                "q1 leg $(legs1[i]) has itag='$(idx1.itags)', q2 leg $(legs2[i]) has itag='$(idx2.itags)'"
            @assert idx1.green == idx2.green "Contracted legs must have matching green flags: " *
                "q1 leg $(legs1[i]) has green=$(idx1.green), q2 leg $(legs2[i]) has green=$(idx2.green)"
            @assert q1.spaces[legs1[i]] == q2.spaces[legs2[i]] "Contracted legs must have matching space info: " *
                "q1 leg $(legs1[i]) spaces != q2 leg $(legs2[i]) spaces"
        end
    end

    T    = promote_type(T1, T2)

    free1  = _free_legs(Val(QD1), legs1)
    free2  = _free_legs(Val(QD2), legs2)
    nf1 = QD1 - CN
    nf2 = QD2 - CN
    QD_out = QD1 + QD2 - 2CN
    RD_out = QD_out + N

    inds_out = (ntuple(i -> q1.inds[free1[i]], Val(QD1 - CN))...,
                ntuple(i -> q2.inds[free2[i]], Val(QD2 - CN))...)

    # Fixed permutations: (free…, om…, contracted…).
    perm1 = (free1..., ntuple(n -> QD1 + n, Val(N))..., legs1...)
    perm2 = (free2..., ntuple(n -> QD2 + n, Val(N))..., legs2...)

    rows1 = q1.rows
    rows2 = q2.rows

    # ── 1. Build sorted row-info vectors keyed by contracted qlabels ─────────
    row_infos1 = _contract_row_infos(QT, rows1, legs1)
    row_infos2 = _contract_row_infos(QT, rows2, legs2)

    # ── 2. Output-sector accumulator ─────────────────────────────────────────
    FreeKey1  = NTuple{nf1, QT}
    FreeKey2  = NTuple{nf2, QT}
    OutKey    = Tuple{FreeKey1, FreeKey2}
    WmatVec   = Vector{LurTensor{Float64, 2}}

    sector_wmats = Dict{OutKey, NTuple{N, WmatVec}}()
    sector_rmts  = Dict{OutKey, Vector{LurTensor{T, RD_out}}}()
    sector_reps  = Dict{OutKey, Tuple{Int, Int}}()

    # ── 3. Main loop: batched matmul per contracted sector ───────────────────
    pos1 = firstindex(row_infos1)
    pos2 = firstindex(row_infos2)
    while pos1 <= lastindex(row_infos1) && pos2 <= lastindex(row_infos2)
        ckey1 = row_infos1[pos1].ckey
        ckey2 = row_infos2[pos2].ckey

        if isless(ckey1, ckey2)
            _, _, pos1 = _contracted_qlabel_run(row_infos1, pos1)
            continue
        elseif isless(ckey2, ckey1)
            _, _, pos2 = _contracted_qlabel_run(row_infos2, pos2)
            continue
        end

        _, run1, next_pos1 = _contracted_qlabel_run(row_infos1, pos1)
        _, run2, next_pos2 = _contracted_qlabel_run(row_infos2, pos2)
        n1, n2 = length(run1), length(run2)

        # 3a. Compute per-row sizes and total dimensions for big_A, big_B.
        szinfo1 = Vector{Tuple{NTuple{QD1 - CN, Int}, NTuple{N, Int}}}(undef, n1)
        rsizes1 = Vector{Int}(undef, n1)
        ncols   = 0   # contracted dimension (same for all rows in this sector)
        for (ii, p1) in enumerate(run1)
            i = row_infos1[p1].row_index
            R     = rows1[i].RMT.data
            sz_f  = ntuple(k -> size(R, free1[k]), Val(QD1 - CN))
            sz_om = ntuple(n -> size(R, QD1 + n), Val(N))
            sz_c  = ntuple(k -> size(R, legs1[k]), Val(CN))
            nrows_i = prod(sz_f; init=1) * prod(sz_om; init=1)
            ncols   = prod(sz_c; init=1)
            rsizes1[ii] = nrows_i
            szinfo1[ii] = (sz_f, sz_om)
        end

        szinfo2 = Vector{Tuple{NTuple{QD2 - CN, Int}, NTuple{N, Int}}}(undef, n2)
        rsizes2 = Vector{Int}(undef, n2)
        for (jj, p2) in enumerate(run2)
            j = row_infos2[p2].row_index
            R     = rows2[j].RMT.data
            sz_f  = ntuple(k -> size(R, free2[k]), Val(QD2 - CN))
            sz_om = ntuple(n -> size(R, QD2 + n), Val(N))
            nrows_j = prod(sz_f; init=1) * prod(sz_om; init=1)
            rsizes2[jj] = nrows_j
            szinfo2[jj] = (sz_f, sz_om)
        end

        roff1 = cumsum([0; rsizes1])
        roff2 = cumsum([0; rsizes2])

        # 3b. Allocate big matrices and fill by permuting each RMT in-place.
        big_A = Matrix{T1}(undef, roff1[end], ncols)
        for (ii, p1) in enumerate(run1)
            i = row_infos1[p1].row_index
            P = permutedims(rows1[i].RMT.data, perm1)
            big_A[roff1[ii]+1:roff1[ii+1], :] = reshape(P, rsizes1[ii], ncols)
        end

        big_B = Matrix{T2}(undef, roff2[end], ncols)
        for (jj, p2) in enumerate(run2)
            j = row_infos2[p2].row_index
            P = permutedims(rows2[j].RMT.data, perm2)
            big_B[roff2[jj]+1:roff2[jj+1], :] = reshape(P, rsizes2[jj], ncols)
        end

        # 3c. Single BLAS GEMM.
        big_C = big_A * big_B'

        # 3d. Process each (row_i, row_j) pair.
        for (ii, p1) in enumerate(run1)
            i = row_infos1[p1].row_index
            r1  = rows1[i]
            fq1 = _free_qlabels(row_infos1[p1].qlabels, free1)::FreeKey1
            sz_f1, sz_om1 = szinfo1[ii]

            for (jj, p2) in enumerate(run2)
                j = row_infos2[p2].row_index
                r2  = rows2[j]
                fq2 = _free_qlabels(row_infos2[p2].qlabels, free2)::FreeKey2
                sz_f2, sz_om2 = szinfo2[jj]

                # ── W-matrix contraction (per symmetry) ──────────────────────
                wmats = _contract_wmats(PS1, r1, r2, legs1, legs2)
                isnothing(wmats) && continue

                # ── Extract block & post-process ─────────────────────────────
                block     = big_C[roff1[ii]+1:roff1[ii+1],
                                  roff2[jj]+1:roff2[jj+1]]
                contr_RMT = _reshape_contract_block(block, sz_f1, sz_om1,
                                                          sz_f2, sz_om2)

                # ── Accumulate into output sector ────────────────────────────
                out_key = (fq1, fq2)::OutKey
                if !haskey(sector_wmats, out_key)
                    sector_wmats[out_key] = ntuple(_ -> LurTensor{Float64, 2}[], Val(N))
                    sector_rmts[out_key]  = LurTensor{T, RD_out}[]
                    sector_reps[out_key]  = (i, j)
                end
                for n in 1:N push!(sector_wmats[out_key][n], wmats[n]) end
                push!(sector_rmts[out_key], LurTensor(contr_RMT))
            end
        end
        pos1 = next_pos1
        pos2 = next_pos2
    end

    # ── 4. Merge each output sector ──────────────────────────────────────────
    result_rows = Vector{row{T, QD_out, N, RD_out}}()
    for (out_key, new_wmats) in sector_wmats
        new_RMTs = sector_rmts[out_key]
        r1_idx, r2_idx = sector_reps[out_key]
        new_qlabels_per_n = get_new_cgr_metadata(
            rows1[r1_idx], rows2[r2_idx], free1, free2, legs1, legs2)
        new_row = merge_new_row(new_wmats, new_RMTs, new_qlabels_per_n,
                                PS1, QD_out)
        isnothing(new_row) || push!(result_rows, new_row)
    end

    # ── 5. Lock reduction ────────────────────────────────────────────────────
    changed_inds2 = Set(change_dir(q2.inds[l]) for l in 1:QD2)
    changed_inds1 = Set(change_dir(q1.inds[l]) for l in 1:QD1)
    final_inds = if reduce_lock
        ntuple(Val(QD_out)) do l
            idx = inds_out[l]
            has_match = l <= nf1 ? (idx ∈ changed_inds2) : (idx ∈ changed_inds1)
            (idx.lock > 0 && has_match) ?
                QIndex(idx.itags, idx.dir, idx.plev, idx.lock - 1, idx.green) : idx
        end
    else
        inds_out
    end

    spaces_out = (ntuple(i -> q1.spaces[free1[i]], Val(QD1 - CN))...,
                  ntuple(i -> q2.spaces[free2[i]], Val(QD2 - CN))...)

    return QSpace(symmetries, result_rows, final_inds, spaces_out)
end

