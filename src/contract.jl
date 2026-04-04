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
function get_cgt_contr_info(r1::row, r2::row, phys_legs1, phys_legs2, n::Int, symm)
    S = symm[n]
    isabelian(S) && return nothing
    cgr1 = r1.cgrs[n];  cgr2 = r2.cgrs[n]
    m1, k1 = cgr1.legdir;  m2, k2 = cgr2.legdir
    up1sp  = Tuple(cgr1.qlabels[i] for i in 1:m1)
    dn1sp  = Tuple(cgr1.qlabels[i] for i in m1+1:m1+k1)
    up2sp  = Tuple(cgr2.qlabels[i] for i in 1:m2)
    dn2sp  = Tuple(cgr2.qlabels[i] for i in m2+1:m2+k2)
    # Map physical contracted legs → stored-qlabel positions via cgp.
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
function get_new_cgp(qlabels1, legdir1, cgp1, free1, legs1,
                     qlabels2, legdir2, cgp2, free2, legs2)
    m1, _ = legdir1;  m2, _ = legdir2
    nf1   = length(free1)
    nf2   = length(free2)
    QD_out = nf1 + nf2
    ctset1 = Set(cgp1[l] for l in legs1)
    ctset2 = Set(cgp2[l] for l in legs2)
    NZ_loc = length(qlabels1[1])

    # (qlabel, output_physical_leg_index) pairs, insertion order = CGT1 then CGT2
    up3 = Vector{Tuple{NTuple{NZ_loc, Int}, Int}}()
    dn3 = Vector{Tuple{NTuple{NZ_loc, Int}, Int}}()

    for (i, l_in) in enumerate(free1)
        sp = cgp1[l_in]
        sp ∈ ctset1 && continue
        (sp <= m1 ? up3 : dn3) |> x -> push!(x, (qlabels1[sp], i))
    end
    for (i, l_in) in enumerate(free2)
        sp = cgp2[l_in]
        sp ∈ ctset2 && continue
        (sp <= m2 ? up3 : dn3) |> x -> push!(x, (qlabels2[sp], i + nf1))
    end

    # Stable sort: equal qlabels keep CGT1-before-CGT2 insertion order.
    sort!(up3; by = x -> x[1], alg = MergeSort)
    sort!(dn3; by = x -> x[1], alg = MergeSort)

    up3sp = Tuple(x[1] for x in up3)
    dn3sp = Tuple(x[1] for x in dn3)
    m3, k3 = length(up3sp), length(dn3sp)

    new_qlabels = (up3sp..., dn3sp...)
    new_legdir  = (m3, k3)

    cgp3 = zeros(Int, QD_out)
    for (si, (_, l_out)) in enumerate(up3);  cgp3[l_out] = si      end
    for (si, (_, l_out)) in enumerate(dn3);  cgp3[l_out] = m3 + si end

    return (new_qlabels, Tuple(cgp3), new_legdir)
end

# Wraps get_new_cgp across all N symmetries for a matched row pair.
function get_new_cgr_metadata(r1_rep::row, r2_rep::row,
                               free1, free2, legs1, legs2)
    N = length(r1_rep.cgrs)
    ntuple(N) do n
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
#   new_RMTs[i]     : QTensor{T} (sz_free..., OM12_1_i,...,OM12_N_i)
#
# Returns:
#   U_mats[n]  : QTensor{Float64,2} (OM3_n, r_n)  — new CGR wmat per symmetry
#   result_RMT : QTensor{T} (sz_free..., r_1,...,r_N)  — compressed RMT
#
# We use QR-based shared isometries for every sector, including K == 1, so the
# resulting basis is normalized consistently with the multi-contribution case.
function _compress_sector(
    new_wmats ::NTuple{N, Vector{<:QTensor{Float64, 2}}},
    new_RMTs  ::Vector{<:QTensor{T, RD}},
    QD_out    ::Int,
    tol       ::Float64 = 1e-12,
) where {T, N, RD}
    K = length(new_RMTs)

    # ── Shared QR basis per symmetry ─────────────────────────────────────────
    U_mats   = Vector{QTensor{Float64, 2}}(undef, N)
    SV_split = [Vector{Matrix{Float64}}(undef, K) for _ in 1:N]

    for n in 1:N
        mats = [w.data for w in new_wmats[n]]
        if all(mat -> all(iszero, mat), mats)
            U_mats[n] = QTensor(zeros(Float64, size(mats[1], 1), 1))
            for i in 1:K
                SV_split[n][i] = zeros(Float64, 1, size(mats[i], 2))
            end
        else
            common_iso, factors = _qr_shared_isometry(mats; tol=tol)
            U_mats[n] = QTensor(common_iso)
            for i in 1:K
                SV_split[n][i] = factors[i]
            end
        end
    end

    # ── Preallocate output RMT as QTensor (sz_free..., r_1,...,r_N) ──────────
    sz_free    = size(new_RMTs[1])[1:QD_out]
    r_sizes    = ntuple(n -> size(U_mats[n], 2), N)
    result_RMT = QTensor{T}(sz_free..., r_sizes...)

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
    new_wmats        ::NTuple{N, Vector{<:QTensor{Float64, 2}}},
    new_RMTs         ::Vector{<:QTensor{T, RD}},
    new_qlabels_per_n,
    symm,
    QD_out           ::Int,
    tol              ::Float64 = 1e-12,
) where {T, N, RD}
    U_mats, result_RMT = _compress_sector(new_wmats, new_RMTs, QD_out, tol)
    # U_mats[n] and result_RMT are already QTensor; no re-wrapping needed.
    cgrs_new = ntuple(N) do n
        new_ql, new_cgp, new_ld = new_qlabels_per_n[n]
        CGR(symm[n], new_ql, U_mats[n], new_cgp, new_ld)
    end

    return row(Tuple(cgrs_new), result_RMT)
end


# ─── contract ────────────────────────────────────────────────────────────────

contract(q1, l1::Int, q2, l2::Int) = contract(q1, (l1,), q2, (l2,))

# Vector / LegList overload: convert to tuples and delegate to the NTuple method.
function contract(q1::QSpace, legs1::AbstractVector{<:Integer},
                  q2::QSpace, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
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

    return contract_v2(q1, Tuple(legs1), q2, Tuple(legs2); verify_legs=false)
end

function contract(q1::QSpace{T1, QD1, N, RD1},
                  legs1::NTuple{CN, Int},
                  q2::QSpace{T2, QD2, N, RD2},
                  legs2::NTuple{CN, Int};
                  reduce_lock::Bool=true,
                  verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, CN}

    @assert q1.symm == q2.symm "QSpace objects must share the same symmetry tuple"
    
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
    
    symm = q1.symm
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
            new_wmats = ntuple(_ -> Vector{QTensor{Float64, 2}}(), N)
            new_RMTs  = Vector{QTensor{T, RD_out}}()
            
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
                    wmats = Vector{QTensor{Float64, 2}}(undef, N)
                    for n in 1:N
                        cgr1n = r1.cgrs[n];  cgr2n = r2.cgrs[n]
                        wm1 = cgr1n.wmat.data   # (OM1, d1)
                        wm2 = cgr2n.wmat.data   # (OM2, d2)
                        info = get_cgt_contr_info(r1, r2, legs1, legs2, n, symm)
                        if info === nothing
                            # Abelian: effective X is [[[1.0]]], OM1=OM2=OM3=1
                            result_w = reshape(wm1' * wm2, 1, size(wm1,2), size(wm2,2))
                        else
                            xsym_obj = getNsave_Xsymbol(symm[n],
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
                        wmats[n] = QTensor(reshape(result_w, size(result_w, 1), :))
                    end
                    
                    if !zero_xsym
                        for n in 1:N push!(new_wmats[n], wmats[n]) end

                        # 2. Contract pre-permuted RMTs; OM pairs merged into N axes.
                        contr_RMT = _contract_RMTs(permed1[idx1], permed2[idx2],
                                                nf1, nf2, N, CN)
                        push!(new_RMTs, QTensor(contr_RMT))
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
                push!(result_rows,
                      merge_new_row(new_wmats, new_RMTs, new_qlabels_per_n,
                                    symm, QD_out))
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

    return QSpace(symm, result_rows, final_inds, spaces_out)
end


# ─── contract_v2 ─────────────────────────────────────────────────────────────
# Optimised contraction that groups rows by *contracted* qlabels first and
# performs a single batched GEMM per contracted sector, replacing many small
# matrix multiplies with one large one.
#
# Algorithm sketch:
#   1. Build contracted-qlabel maps: cq → [row indices]  (for each QSpace)
#   2. For each common contracted sector cq:
#      a) Pre-permute & reshape each row's RMT to (F·OM, C) matrix.
#      b) Vcat all q1 matrices → big_A;  vcat all q2 matrices → big_B.
#      c) big_C = big_A * big_B'            ← single BLAS call
#      d) Split big_C into per-(i, j) blocks and post-process each block
#         (reshape, permute, merge-OM, w-matrix / X-symbol contraction).
#      e) Accumulate results per output free-sector.
#   3. Merge each output sector (SVD compression → output row).
#   4. Lock reduction / build result QSpace.

# ── Post-process helper ──────────────────────────────────────────────────────
# Reshape a (F1·OM1, F2·OM2) block from the batched matmul back to the
# standard contracted-RMT layout (f1…, f2…, om12_1, …, om12_N).
function _reshape_contract_block(block::AbstractMatrix,
                                  sz_f1::Vector{Int}, sz_om1::Vector{Int},
                                  sz_f2::Vector{Int}, sz_om2::Vector{Int})
    nf1 = length(sz_f1)
    nf2 = length(sz_f2)
    N   = length(sz_om1)

    # (F1·OM1, F2·OM2) → (f1…, om1…, f2…, om2…)
    C = reshape(block, sz_f1..., sz_om1..., sz_f2..., sz_om2...)

    # Permute to (f1…, f2…, om1_1, om2_1, …, om1_N, om2_N)
    composed_perm = vcat(1:nf1, nf1+N+1:nf1+N+nf2,
                         [[nf1+n, nf1+N+nf2+n] for n in 1:N]...)
    C_i = permutedims(C, composed_perm)

    # Merge each (om1_n, om2_n) pair → om1_n · om2_n
    sz_om_merged = [sz_om1[n] * sz_om2[n] for n in 1:N]
    return reshape(C_i, sz_f1..., sz_f2..., sz_om_merged...)
end

# ── Convenience overloads ─────────────────────────────────────────────────────
contract_v2(q1, l1::Int, q2, l2::Int) = contract_v2(q1, (l1,), q2, (l2,))

function contract_v2(q1::QSpace, legs1::AbstractVector{<:Integer},
                     q2::QSpace, legs2::AbstractVector{<:Integer}; kwargs...)
    return contract_v2(q1, Tuple(legs1), q2, Tuple(legs2); kwargs...)
end

# TODO: Further test this function and benchmark against the original contract 
# If it is accurate and significantly faster for large contractions, 
# consider replacing the original contract with this version as the default.
# ── Main entry point ──────────────────────────────────────────────────────────
function contract_v2(q1::QSpace{T1, QD1, N, RD1},
                     legs1::NTuple{CN, Int},
                     q2::QSpace{T2, QD2, N, RD2},
                     legs2::NTuple{CN, Int};
                     reduce_lock::Bool=true,
                     verify_legs::Bool=true) where {T1, T2, QD1, QD2, N, RD1, RD2, CN}

    @assert q1.symm == q2.symm "QSpace objects must share the same symmetry tuple"

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

    symm = q1.symm
    T    = promote_type(T1, T2)

    free1  = [l for l in 1:QD1 if l ∉ legs1]
    free2  = [l for l in 1:QD2 if l ∉ legs2]
    nf1, nf2 = length(free1), length(free2)
    QD_out = nf1 + nf2
    RD_out = QD_out + N

    inds_out = Tuple([[q1.inds[l] for l in free1]; [q2.inds[l] for l in free2]])

    # Fixed permutations: (free…, om…, contracted…).
    perm1 = [free1; collect(QD1+1:QD1+N); collect(legs1)]
    perm2 = [free2; collect(QD2+1:QD2+N); collect(legs2)]

    rows1 = q1.rows
    rows2 = q2.rows

    # ── 1. Build contracted-qlabel maps ──────────────────────────────────────
    cmap1 = Dict{Any, Vector{Int}}()
    for (i, r) in enumerate(rows1)
        ck = Tuple(_row_qlabel(r, l) for l in legs1)
        push!(get!(cmap1, ck, Int[]), i)
    end

    cmap2 = Dict{Any, Vector{Int}}()
    for (j, r) in enumerate(rows2)
        ck = Tuple(_row_qlabel(r, l) for l in legs2)
        push!(get!(cmap2, ck, Int[]), j)
    end

    # ── 2. Output-sector accumulator ─────────────────────────────────────────
    FreeKey1  = NTuple{nf1, NTuple{N, Tuple{Vararg{Int}}}}
    FreeKey2  = NTuple{nf2, NTuple{N, Tuple{Vararg{Int}}}}
    OutKey    = Tuple{FreeKey1, FreeKey2}
    WmatVec   = Vector{QTensor{Float64, 2}}

    sector_wmats = Dict{OutKey, NTuple{N, WmatVec}}()
    sector_rmts  = Dict{OutKey, Vector{QTensor{T, RD_out}}}()
    sector_reps  = Dict{OutKey, Tuple{Int, Int}}()

    # ── 3. Main loop: batched matmul per contracted sector ───────────────────
    for (ck, idxs1) in cmap1
        haskey(cmap2, ck) || continue
        idxs2 = cmap2[ck]
        n1, n2 = length(idxs1), length(idxs2)

        # 3a. Compute per-row sizes and total dimensions for big_A, big_B.
        szinfo1 = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, n1)
        rsizes1 = Vector{Int}(undef, n1)
        ncols   = 0   # contracted dimension (same for all rows in this sector)
        for (ii, i) in enumerate(idxs1)
            R     = rows1[i].RMT.data
            sz_f  = [size(R, l) for l in free1]
            sz_om = [size(R, QD1+n) for n in 1:N]
            sz_c  = [size(R, l) for l in legs1]
            nrows_i = prod(sz_f; init=1) * prod(sz_om; init=1)
            ncols   = prod(sz_c; init=1)
            rsizes1[ii] = nrows_i
            szinfo1[ii] = (sz_f, sz_om)
        end

        szinfo2 = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, n2)
        rsizes2 = Vector{Int}(undef, n2)
        for (jj, j) in enumerate(idxs2)
            R     = rows2[j].RMT.data
            sz_f  = [size(R, l) for l in free2]
            sz_om = [size(R, QD2+n) for n in 1:N]
            nrows_j = prod(sz_f; init=1) * prod(sz_om; init=1)
            rsizes2[jj] = nrows_j
            szinfo2[jj] = (sz_f, sz_om)
        end

        roff1 = cumsum([0; rsizes1])
        roff2 = cumsum([0; rsizes2])

        # 3b. Allocate big matrices and fill by permuting each RMT in-place.
        big_A = Matrix{T1}(undef, roff1[end], ncols)
        for (ii, i) in enumerate(idxs1)
            P = permutedims(rows1[i].RMT.data, perm1)
            big_A[roff1[ii]+1:roff1[ii+1], :] = reshape(P, rsizes1[ii], ncols)
        end

        big_B = Matrix{T2}(undef, roff2[end], ncols)
        for (jj, j) in enumerate(idxs2)
            P = permutedims(rows2[j].RMT.data, perm2)
            big_B[roff2[jj]+1:roff2[jj+1], :] = reshape(P, rsizes2[jj], ncols)
        end

        # 3c. Single BLAS GEMM.
        big_C = big_A * big_B'

        # 3d. Process each (row_i, row_j) pair.
        for (ii, i) in enumerate(idxs1)
            r1  = rows1[i]
            fq1 = Tuple(_row_qlabel(r1, l) for l in free1)::FreeKey1
            sz_f1, sz_om1 = szinfo1[ii]

            for (jj, j) in enumerate(idxs2)
                r2  = rows2[j]
                fq2 = Tuple(_row_qlabel(r2, l) for l in free2)::FreeKey2
                sz_f2, sz_om2 = szinfo2[jj]

                # ── W-matrix contraction (per symmetry) ──────────────────────
                zero_xsym = false
                wmats = Vector{QTensor{Float64, 2}}(undef, N)
                for n in 1:N
                    cgr1n = r1.cgrs[n];  cgr2n = r2.cgrs[n]
                    wm1 = cgr1n.wmat.data
                    wm2 = cgr2n.wmat.data
                    info = get_cgt_contr_info(r1, r2, legs1, legs2, n, symm)
                    if info === nothing
                        # Abelian: X = [[[1.0]]], OM1=OM2=OM3=1
                        result_w = reshape(wm1' * wm2, 1, size(wm1,2), size(wm2,2))
                    else
                        xsym_obj = getNsave_Xsymbol(symm[n],
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
                    wmats[n] = QTensor(reshape(result_w, size(result_w, 1), :))
                end

                zero_xsym && continue

                # ── Extract block & post-process ─────────────────────────────
                block     = big_C[roff1[ii]+1:roff1[ii+1],
                                  roff2[jj]+1:roff2[jj+1]]
                contr_RMT = _reshape_contract_block(block, sz_f1, sz_om1,
                                                          sz_f2, sz_om2)

                # ── Accumulate into output sector ────────────────────────────
                out_key = (fq1, fq2)::OutKey
                if !haskey(sector_wmats, out_key)
                    sector_wmats[out_key] = ntuple(_ -> QTensor{Float64, 2}[], N)
                    sector_rmts[out_key]  = QTensor{T, RD_out}[]
                    sector_reps[out_key]  = (i, j)
                end
                for n in 1:N push!(sector_wmats[out_key][n], wmats[n]) end
                push!(sector_rmts[out_key], QTensor(contr_RMT))
            end
        end
    end

    # ── 4. Merge each output sector ──────────────────────────────────────────
    result_rows = Vector{row{T, QD_out, N, RD_out}}()
    for (out_key, new_wmats) in sector_wmats
        new_RMTs = sector_rmts[out_key]
        r1_idx, r2_idx = sector_reps[out_key]
        new_qlabels_per_n = get_new_cgr_metadata(
            rows1[r1_idx], rows2[r2_idx], free1, free2, legs1, legs2)
        push!(result_rows,
              merge_new_row(new_wmats, new_RMTs, new_qlabels_per_n,
                            symm, QD_out))
    end

    # ── 5. Lock reduction ────────────────────────────────────────────────────
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

    spaces_out = ([q1.spaces[l] for l in free1]..., [q2.spaces[l] for l in free2]...)

    return QSpace(symm, result_rows, final_inds, spaces_out)
end
