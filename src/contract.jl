# ─── Helpers ─────────────────────────────────────────────────────────────────

# Physical qlabel of leg l in row r: an N-tuple of qlabels, one per symmetry.
function _row_qlabel(r::row{T, QD, N}, l::Int) where {T, QD, N}
    Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[l]] for n in 1:N)
end

# Sort key: (free_leg_qlabels..., contr_leg_qlabels...)
# Free legs first so rows with the same output sector are grouped together;
# contracted legs second so within a group rows are ordered by charge sector
# (enabling an efficient two-pointer sweep when pairing with the other TLArray).
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
#   nothing    : when QR reveals that any symmetry contributes no usable
#                columns, so the whole merged row is identically zero
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

# ─── merge_new_row ────────────────────────────────────────────────────────────
# Wraps _compress_sector and assembles the output row struct.
function merge_new_row(
    new_wmats        ::NTuple{N, Vector{LurTensor{Float64, 2, AW}}},
    new_RMTs         ::Vector{LurTensor{T, RD, AR}},
    new_qlabels_per_n,
    ::Type{ProductSymm{Syms}},
    QD_out           ::Int,
    tol              ::Float64 = 1e-12,
) where {T, N, RD, Syms, AW<:AbstractArray{Float64, 2}, AR<:AbstractArray{T, RD}}
    compressed = _compress_sector(new_wmats, new_RMTs, QD_out, tol)
    isnothing(compressed) && return nothing
    U_mats, result_RMT = compressed
    # U_mats[n] and result_RMT are already LurTensor; no re-wrapping needed.
    cgrs_new = ntuple(Val(N)) do n
        new_ql, new_cgp, new_ld = new_qlabels_per_n[n]
        CGR(fieldtype(Syms, n), new_ql, U_mats[n], new_cgp, new_ld)
    end

    return row(Tuple(cgrs_new), result_RMT)
end

# ─── contract_old ────────────────────────────────────────────────────────────

contract_old(q1, l1::Int, q2, l2::Int) = contract_old(q1, (l1,), q2, (l2,))

# Vector / LegList overload: convert to tuples and delegate to the NTuple method.
function contract_old(q1::TLArray, legs1::AbstractVector{<:Integer},
                      q2::TLArray, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract_old(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
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

function contract_old(q1::TLArray{T1, QD1, N, RD1, QT, PS, CGR1},
                      legs1::NTuple{CN, Int},
                      q2::TLArray{T2, QD2, N, RD2, QT, PS, CGR2},
                      legs2::NTuple{CN, Int};
                      reduce_lock::Bool=true,
                      verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, QT, PS, CGR1, CGR2, CN}

    symmetries = product_symms(PS)
    @assert symmetries == symm(q2) "TLArray objects must share the same symmetry tuple"
    
    # Verify contracted legs have opposite arrow directions, matching itags/green, and same space info
    if verify_legs
        for i in 1:CN
            idx1 = q1.inds[legs1[i]]
            idx2 = q2.inds[legs2[i]]
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
    CGRS_out = cgrstype(PS, Val(QD_out))
    result_rows = Vector{row{T, QD_out, N, RD_out, CGRS_out}}()
    sample_wmat = !isempty(rows1) ? rows1[1].cgrs[1].wmat : rows2[1].cgrs[1].wmat
    sample_rmt  = !isempty(rows1) ? rows1[1].RMT : rows2[1].RMT
    WMatT = typeof(similar(sample_wmat, Float64, (1, 1)))
    RMTT  = typeof(similar(sample_rmt, T, ntuple(_ -> 1, RD_out)))

    for (fq1, v1) in sm1.data
        for (fq2, v2) in sm2.data
            new_wmats = ntuple(_ -> WMatT[], N)
            new_RMTs  = RMTT[]
            
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
                    wmats = Vector{WMatT}(undef, N)
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
                                        PS, QD_out)
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
                TLIndex(idx.itags, idx.dir, idx.plev, idx.lock - 1, idx.dual) : idx
        end
    else
        inds_out
    end

    # Compute spaces for result: free legs from q1 followed by free legs from q2
    spaces_out = ([q1.spaces[l] for l in free1]..., [q2.spaces[l] for l in free2]...)

    return TLArray(PS, result_rows, final_inds, spaces_out)
end


# ─── contract ────────────────────────────────────────────────────────────────
# Optimised contraction that sorts rows by *contracted* qlabels first and
# performs a single batched GEMM per common contracted sector, replacing many
# small matrix multiplies with one large one.
#
# Algorithm sketch:
#   1. Build sorted row-info vectors for each TLArray. Each entry carries the
#      row index, all physical-leg qlabels, and the contracted-qlabel key.
#   2. Two-pointer scan over both sorted vectors and process common sectors:
#      a) Pre-permute & reshape each row's RMT to (F·OM, C) matrix.
#      b) Vcat all q1 matrices → big_A;  vcat all q2 matrices → big_B.
#      c) big_C = big_A * big_B'            ← single BLAS call
#      d) Split big_C into per-(i, j) blocks and post-process each block
#         (reshape, permute, merge-OM, w-matrix / X-symbol contraction).
#      e) Accumulate results per output free-sector.
#   3. Merge each output sector (SVD compression → output row).
#   4. Lock reduction / build result TLArray.
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

_contracted_qlabel_type(::Type{QT}, ::Val{CN}) where {QT, CN} = NTuple{CN, QT}

_contracted_qlabel(::Type{QT}, r::row, legs::NTuple{CN, Int}) where {QT, CN} =
    ntuple(i -> _row_qlabel(QT, r, legs[i]), Val(CN))

struct ContractRowInfo{NF, QT, CQT}
    row_index::Int
    free_qlabels::NTuple{NF, QT}
    contracted_qlabels::CQT
end

function _contract_row_infos(::Type{QT}, rows::AbstractVector{<:row{T, QD, N}},
                             free_legs::NTuple{NF, Int},
                             legs::NTuple{CN, Int}) where {QT, T, QD, N, NF, CN}
    CQT = _contracted_qlabel_type(QT, Val(CN))
    Info = ContractRowInfo{NF, QT, CQT}
    infos = Vector{Info}(undef, length(rows))
    for i in eachindex(rows)
        infos[i] = Info(
            i,
            ntuple(k -> _row_qlabel(QT, rows[i], free_legs[k]), Val(NF)),
            _contracted_qlabel(QT, rows[i], legs)::CQT,
        )
    end
    return sort!(infos; by = info -> info.contracted_qlabels, alg=MergeSort)
end

function _contracted_qlabel_entries(::Type{QT}, rows::AbstractVector{<:row{T, QD, N}},
                                    legs::NTuple{CN, Int}) where {QT, T, QD, N, CN}
    infos = _contract_row_infos(QT, rows, _free_legs(Val(QD), legs), legs)
    return [(info.row_index, info.contracted_qlabels) for info in infos]
end

function _contracted_qlabel_run(infos::AbstractVector{<:ContractRowInfo{NF, QT, CQT}},
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
    d1 = size(wm1, 2)
    d2 = size(wm2, 2)
    WT = promote_type(T1, T2, T3)

    left_input = reshape(permutedims(xarr, (1, 3, 2)), OM1, OM3 * OM2)
    left = Matrix{WT}(undef, d1, OM3 * OM2)
    mul!(left, adjoint(wm1), left_input)

    right_input = reshape(permutedims(reshape(left, d1, OM3, OM2), (2, 1, 3)), OM3 * d1, OM2)
    result = Matrix{WT}(undef, OM3 * d1, d2)
    mul!(result, right_input, wm2)

    return LurTensor(reshape(result, OM3, d1 * d2))
end

@inline _symmetry_qlabels(qlabels::NTuple{NF, QT}, ::Val{n}) where {NF, QT, n} =
    ntuple(k -> qlabels[k][n], Val(NF))

@inline _symmetry_contracted_qlabel(qlabels::NTuple{CN, QT}, ::Val{n}, ::Val{CN}) where {CN, QT, n} =
    ntuple(k -> qlabels[k][n], Val(CN))

@inline function _xsym_cache_key_type(::Type{S}, ::Val{CN}, ::Val{NF1}, ::Val{NF2}) where {S<:NonabelianSymm, CN, NF1, NF2}
    SQT = NTuple{nzops(S), Int}
    return Tuple{NTuple{CN, SQT}, NTuple{NF1, SQT}, NTuple{NF2, SQT}}
end

@inline function _xsym_cache_type(::Type{S}, ::Val{CN}, ::Val{NF1}, ::Val{NF2}) where {S<:NonabelianSymm, CN, NF1, NF2}
    return Dict{_xsym_cache_key_type(S, Val(CN), Val(NF1), Val(NF2)), Array{Float64, 3}}
end

@generated function _distinct_nonabelian_symms_type(::Type{ProductSymm{Syms}}) where {Syms}
    unique_syms = Any[]
    for S in Syms.parameters
        if S <: NonabelianSymm && !(S in unique_syms)
            push!(unique_syms, S)
        end
    end
    return :($(Tuple{unique_syms...}))
end

@generated function _distinct_nonabelian_symms(::Type{ProductSymm{Syms}}) where {Syms}
    unique_syms = Any[]
    for S in Syms.parameters
        if S <: NonabelianSymm && !(S in unique_syms)
            push!(unique_syms, S)
        end
    end
    return Expr(:tuple, map(S -> :($S), unique_syms)...)
end

_ndistinct_nonabelian_symms(::Type{PS}) where {PS<:ProductSymm} = length(_distinct_nonabelian_symms(PS))

struct XSymCaches{CacheSyms<:Tuple, CacheTuple<:Tuple}
    caches::CacheTuple
end

XSymCaches{CacheSyms}(caches::CacheTuple) where {CacheSyms<:Tuple, CacheTuple<:Tuple} =
    XSymCaches{CacheSyms, CacheTuple}(caches)

@generated function _xsym_caches_type(::Type{ProductSymm{Syms}}, ::Val{CN}, ::Val{NF1}, ::Val{NF2}) where {Syms, CN, NF1, NF2}
    unique_syms = Any[]
    dict_types = Any[]
    for S in Syms.parameters
        if S <: NonabelianSymm && !(S in unique_syms)
            push!(unique_syms, S)
            push!(dict_types, _xsym_cache_type(S, Val(CN), Val(NF1), Val(NF2)))
        end
    end
    cache_syms_type = Tuple{unique_syms...}
    cache_tuple_type = Tuple{dict_types...}
    return :($(XSymCaches{cache_syms_type, cache_tuple_type}))
end

@inline _xsym_cache(::Type{<:AbelianSymm}, ::XSymCaches) = nothing

@generated function _xsym_cache(::Type{S}, xsym_caches::XSymCaches{CacheSyms, CacheTuple}) where {S<:NonabelianSymm, CacheSyms, CacheTuple}
    idx = findfirst(T -> T == S, fieldtypes(CacheSyms))
    return :(xsym_caches.caches[$idx])
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

@inline _cgr_up_qlabels(cgr::CGR{QD, NZ, S}, ::Val{M}) where {QD, NZ, S, M} =
    ntuple(i -> cgr.qlabels[i], Val(M))

@inline _cgr_dn_qlabels(cgr::CGR{QD, NZ, S}, ::Val{M}) where {QD, NZ, S, M} =
    ntuple(i -> cgr.qlabels[M + i], Val(QD - M))

@inline _stored_contract_legs(cgr::CGR{QD, NZ, S}, legs::NTuple{CN, Int}) where {QD, NZ, S, CN} =
    ntuple(k -> cgr.cgp[legs[k]], Val(CN))

@generated function _load_nonabelian_xarr(::Type{S},
                                          cgr1::CGR{QD1, NZ, S},
                                          cgr2::CGR{QD2, NZ, S},
                                          legs1::NTuple{CN, Int},
                                          legs2::NTuple{CN, Int}) where {S<:NonabelianSymm, QD1, QD2, NZ, CN}
    branches = Expr[]
    for m1 in 0:QD1
        for m2 in 0:QD2
            push!(branches, quote
                if cgr1.legdir[1] == $m1 && cgr2.legdir[1] == $m2
                    up1sp = _cgr_up_qlabels(cgr1, Val($m1))
                    dn1sp = _cgr_dn_qlabels(cgr1, Val($m1))
                    up2sp = _cgr_up_qlabels(cgr2, Val($m2))
                    dn2sp = _cgr_dn_qlabels(cgr2, Val($m2))
                    ctlegs1 = _stored_contract_legs(cgr1, legs1)
                    ctlegs2 = _stored_contract_legs(cgr2, legs2)
                    xsym_obj = getNsave_Xsymbol(S, up1sp, dn1sp, up2sp, dn2sp, ctlegs1, ctlegs2)
                    isnothing(xsym_obj) && return nothing
                    return xsym_obj.xsym_arr::Array{Float64, 3}
                end
            end)
        end
    end
    return quote
        $(branches...)
        throw(ArgumentError("invalid CGR leg directions"))
    end
end

function _merge_xsym_cache_for_symmetry!(::Type{S},
                                         cache::Dict{XKey, Array{Float64, 3}},
                                         ::Type{QT},
                                         ::Val{n},
                                         row_infos1::AbstractVector{<:ContractRowInfo{NF1, QT, CQT}},
                                         row_infos2::AbstractVector{<:ContractRowInfo{NF2, QT, CQT}},
                                         rows1::AbstractVector{<:row{T1, QD1, N, RD1, CGRS1}},
                                         rows2::AbstractVector{<:row{T2, QD2, N, RD2, CGRS2}},
                                         legs1::NTuple{CN, Int},
                                         legs2::NTuple{CN, Int}) where {S<:NonabelianSymm, XKey, QT, n, NF1, NF2, CQT,
                                                                         T1, T2, QD1, QD2, N, RD1, RD2, CGRS1, CGRS2, CN}
    pos1 = firstindex(row_infos1)
    pos2 = firstindex(row_infos2)
    while pos1 <= lastindex(row_infos1) && pos2 <= lastindex(row_infos2)
        ckey1 = row_infos1[pos1].contracted_qlabels
        ckey2 = row_infos2[pos2].contracted_qlabels

        if isless(ckey1, ckey2)
            _, _, pos1 = _contracted_qlabel_run(row_infos1, pos1)
            continue
        elseif isless(ckey2, ckey1)
            _, _, pos2 = _contracted_qlabel_run(row_infos2, pos2)
            continue
        end

        _, run1, next_pos1 = _contracted_qlabel_run(row_infos1, pos1)
        _, run2, next_pos2 = _contracted_qlabel_run(row_infos2, pos2)

        for p1 in run1
            info1 = row_infos1[p1]
            cgr1n = rows1[info1.row_index].cgrs[n]
            for p2 in run2
                info2 = row_infos2[p2]
                xkey = _xsym_cache_key(QT, info1.contracted_qlabels,
                                       info1.free_qlabels,
                                       info2.free_qlabels,
                                       Val(n))::XKey
                haskey(cache, xkey) && continue
                xarr = _load_nonabelian_xarr(S, cgr1n, rows2[info2.row_index].cgrs[n], legs1, legs2)
                isnothing(xarr) || (cache[xkey] = xarr)
            end
        end

        pos1 = next_pos1
        pos2 = next_pos2
    end

    return cache
end

@generated function _build_xsym_caches(::Type{ProductSymm{Syms}}, ::Type{QT},
                                       row_infos1::AbstractVector{<:ContractRowInfo{NF1, QT, CQT}},
                                       row_infos2::AbstractVector{<:ContractRowInfo{NF2, QT, CQT}},
                                       rows1::AbstractVector{<:row{T1, QD1, N, RD1, CGRS1}},
                                       rows2::AbstractVector{<:row{T2, QD2, N, RD2, CGRS2}},
                                       legs1::NTuple{CN, Int},
                                       legs2::NTuple{CN, Int}) where {Syms, QT, NF1, NF2, CQT,
                                                                       T1, T2, QD1, QD2, N, RD1, RD2, CGRS1, CGRS2, CN}
    unique_syms = Any[]
    positions_by_symmetry = Dict{Any, Vector{Int}}()
    for (n, S) in enumerate(Syms.parameters)
        if S <: NonabelianSymm
            if !haskey(positions_by_symmetry, S)
                positions_by_symmetry[S] = Int[]
                push!(unique_syms, S)
            end
            push!(positions_by_symmetry[S], n)
        end
    end

    XSymCachesType = _xsym_caches_type(ProductSymm{Syms}, Val(CN), Val(NF1), Val(NF2))
    cache_exprs = Any[]
    for S in unique_syms
        dict_type = _xsym_cache_type(S, Val(CN), Val(NF1), Val(NF2))
        merge_exprs = Any[]
        for n in positions_by_symmetry[S]
            push!(merge_exprs, :(_merge_xsym_cache_for_symmetry!($S, cache, QT, Val($n),
                                                                 row_infos1, row_infos2, rows1, rows2,
                                                                 legs1, legs2)))
        end
        push!(cache_exprs, quote
            cache = $dict_type()
            $(merge_exprs...)
            cache
        end)
    end

    cache_tuple = Expr(:tuple, cache_exprs...)
    return :($XSymCachesType($cache_tuple))
end

@inline function _contract_wmat_for_symmetry(::Type{S},
                                             r1::row{T1, QD1, N, RD1, CGRS1},
                                             r2::row{T2, QD2, N, RD2, CGRS2},
                                             contracted_qlabels::NTuple{CN, QT},
                                             free_qlabels1::NTuple{NF1, QT},
                                             free_qlabels2::NTuple{NF2, QT},
                                             ::Nothing,
                                             ::Val{n}) where {S<:AbelianSymm, QT, NF1, NF2, CN,
                                                              T1, QD1, T2, QD2, N, RD1, RD2, CGRS1, CGRS2, n}
    return LurTensor([1.0;;])
end

@inline function _contract_wmat_for_symmetry(::Type{S},
                                             r1::row{T1, QD1, N, RD1, CGRS1},
                                             r2::row{T2, QD2, N, RD2, CGRS2},
                                             contracted_qlabels::NTuple{CN, QT},
                                             free_qlabels1::NTuple{NF1, QT},
                                             free_qlabels2::NTuple{NF2, QT},
                                             xsym_cache::Dict{XKey, Array{Float64, 3}},
                                             ::Val{n}) where {S<:NonabelianSymm, QT, NF1, NF2, XKey, CN, n,
                                                              T1, QD1, T2, QD2, N, RD1, RD2, CGRS1, CGRS2}
    xkey = _xsym_cache_key(QT, contracted_qlabels, free_qlabels1, free_qlabels2, Val(n))::XKey
    xarr = get(xsym_cache, xkey, nothing)
    isnothing(xarr) && return nothing
    return _contract_xsym_wmat(r1.cgrs[n].wmat.data, xarr, r2.cgrs[n].wmat.data)
end

function _contract_wmats(::Type{ProductSymm{Syms}},
                         r1::row{T1, QD1, N, RD1, CGRS1},
                         r2::row{T2, QD2, N, RD2, CGRS2},
                         contracted_qlabels::NTuple{CN, QT},
                         free_qlabels1::NTuple{NF1, QT},
                         free_qlabels2::NTuple{NF2, QT},
                         xsym_caches::XSymCaches) where {Syms, QT, NF1, NF2, CN,
                                                         T1, QD1, T2, QD2, N, RD1, RD2, CGRS1, CGRS2}
    vec_tuple = ntuple(Val(N)) do n
        S = fieldtype(Syms, n)
        cache = _xsym_cache(S, xsym_caches)
        _contract_wmat_for_symmetry(S, r1, r2,
                                    contracted_qlabels,
                                    free_qlabels1,
                                    free_qlabels2,
                                    cache,
                                    Val(n))
    end
    if any(isnothing, vec_tuple)
        return nothing
    end
    return vec_tuple
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

function contract(q1::TLArray, legs1::AbstractVector{<:Integer},
                  q2::TLArray, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
end

# ── Main entry point ──────────────────────────────────────────────────────────
function contract(q1::TLArray{T1, QD1, N, RD1, QT, PS, CGR1},
                  legs1::NTuple{CN, Int},
                  q2::TLArray{T2, QD2, N, RD2, QT, PS, CGR2},
                  legs2::NTuple{CN, Int};
                  reduce_lock::Bool=true,
                  verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, QT, PS, CGR1, CGR2, CN}

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

    # Fixed permutations: (free…, om…, contracted…).
    perm1 = (free1..., ntuple(n -> QD1 + n, Val(N))..., legs1...)
    perm2 = (free2..., ntuple(n -> QD2 + n, Val(N))..., legs2...)

    rows1 = q1.rows
    rows2 = q2.rows

    # ── 1. Build sorted row-info vectors keyed by contracted qlabels ─────────
    row_infos1 = _contract_row_infos(QT, rows1, free1, legs1)
    row_infos2 = _contract_row_infos(QT, rows2, free2, legs2)

    # ?? 2. Output-sector accumulator ?????????????????????????????????????????
    FreeKey1  = NTuple{nf1, QT}
    FreeKey2  = NTuple{nf2, QT}
    OutKey    = Tuple{FreeKey1, FreeKey2}
    sample_wmat = !isempty(rows1) ? rows1[1].cgrs[1].wmat : rows2[1].cgrs[1].wmat
    sample_rmt  = !isempty(rows1) ? rows1[1].RMT : rows2[1].RMT

    # TODO: When GPU support is added, these should be generalized
    WMatT = LurTensor{Float64, 2, Array{Float64, 2}}
    RMTT  = LurTensor{T, RD_out, Array{T, RD_out}}
    WmatVec   = Vector{WMatT}

    # X-symbol cache type
    XCT = _xsym_caches_type(PS, Val(CN), Val(nf1), Val(nf2))
    xsym_caches::XCT = _build_xsym_caches(PS, QT, row_infos1, row_infos2,
                                     rows1, rows2, legs1, legs2)

    sector_wmats = Dict{OutKey, NTuple{N, WmatVec}}()
    sector_rmts  = Dict{OutKey, Vector{RMTT}}()
    sector_reps  = Dict{OutKey, Tuple{Int, Int}}()

    # ── 3. Main loop: batched matmul per contracted sector ───────────────────
    pos1 = firstindex(row_infos1)
    pos2 = firstindex(row_infos2)
    while pos1 <= lastindex(row_infos1) && pos2 <= lastindex(row_infos2)
        ckey1 = row_infos1[pos1].contracted_qlabels
        ckey2 = row_infos2[pos2].contracted_qlabels

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
        ncols::Int = 0   # contracted dimension (same for all rows in this sector)
        for (ii, p1) in enumerate(run1)
            i = row_infos1[p1].row_index
            R     = rows1[i].RMT.data
            sz_f::NTuple{QD1 - CN, Int}  = ntuple(k -> size(R, free1[k]), Val(QD1 - CN))
            sz_om::NTuple{N, Int}  = ntuple(n -> size(R, QD1 + n), Val(N))
            sz_c::NTuple{CN, Int}  = ntuple(k -> size(R, legs1[k]), Val(CN))
            nrows_i::Int = prod(sz_f; init=1) * prod(sz_om; init=1)
            ncols = prod(sz_c; init=1)
            rsizes1[ii] = nrows_i
            szinfo1[ii] = (sz_f, sz_om)
        end

        szinfo2 = Vector{Tuple{NTuple{QD2 - CN, Int}, NTuple{N, Int}}}(undef, n2)
        rsizes2 = Vector{Int}(undef, n2)
        for (jj, p2) in enumerate(run2)
            j = row_infos2[p2].row_index
            R     = rows2[j].RMT.data
            sz_f::NTuple{QD2 - CN, Int}  = ntuple(k -> size(R, free2[k]), Val(QD2 - CN))
            sz_om::NTuple{N, Int}  = ntuple(n -> size(R, QD2 + n), Val(N))
            nrows_j::Int = prod(sz_f; init=1) * prod(sz_om; init=1)
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
            fq1 = row_infos1[p1].free_qlabels::FreeKey1
            sz_f1, sz_om1 = szinfo1[ii]

            for (jj, p2) in enumerate(run2)
                j = row_infos2[p2].row_index
                r2  = rows2[j]
                fq2 = row_infos2[p2].free_qlabels::FreeKey2
                sz_f2, sz_om2 = szinfo2[jj]

                # ── W-matrix contraction (per symmetry) ──────────────────────
                wmats = _contract_wmats(PS, r1, r2, ckey1, fq1, fq2, xsym_caches)
                isnothing(wmats) && continue

                # ── Extract block & post-process ─────────────────────────────
                block     = @view big_C[roff1[ii]+1:roff1[ii+1],
                                        roff2[jj]+1:roff2[jj+1]]
                contr_RMT = _reshape_contract_block(block, sz_f1, sz_om1,
                                                          sz_f2, sz_om2)

                # ── Accumulate into output sector ────────────────────────────
                out_key = (fq1, fq2)::OutKey
                if !haskey(sector_wmats, out_key)
                    sector_wmats[out_key] = ntuple(_ -> WMatT[], Val(N))
                    sector_rmts[out_key]  = RMTT[]
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
    CGRS_out = cgrstype(PS, Val(QD_out))
    result_rows = Vector{row{T, QD_out, N, RD_out, CGRS_out}}()
    for (out_key, new_wmats) in sector_wmats
        new_RMTs = sector_rmts[out_key]
        r1_idx, r2_idx = sector_reps[out_key]
        new_qlabels_per_n = get_new_cgr_metadata(
            rows1[r1_idx], rows2[r2_idx], free1, free2, legs1, legs2)
        new_row = merge_new_row(new_wmats, new_RMTs, new_qlabels_per_n,
                                PS, QD_out)
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
                TLIndex(idx.itags, idx.dir, idx.plev, idx.lock - 1, idx.dual) : idx
        end
    else
        inds_out
    end

    spaces_out = (ntuple(i -> q1.spaces[free1[i]], Val(QD1 - CN))...,
                  ntuple(i -> q2.spaces[free2[i]], Val(QD2 - CN))...)

    return TLArray(PS, result_rows, final_inds, spaces_out)::TLArray{T, QD_out, N, RD_out, QT, PS, CGRS_out}
end








