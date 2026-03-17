# ─── svd ───────────────────────────────────────────────────────────────
#
# Perform symmetry-adapted SVD of a QSpace object.
#
# Arguments:
#   q          : QSpace to decompose (any rank QD, N symmetries)
#   left_legs  : tuple/vector of leg indices forming the left (U) side
#   left_tag   : itag for the new bond leg on the U/S side   (default "svdL")
#   right_tag  : itag for the new bond leg on the S/Vd side  (default "svdR")
#   cutoff     : keep singular values > cutoff * σ_max  (default 1e-12)
#   Nkeep      : keep the Nkeep largest singular values globally, ignoring
#                degeneracy and counting missing sectors as zero singular values
#
# Returns (U, S, Vd) where:
#   U  : legs (original left legs...,  bond '+')
#   S  : legs (left_tag '-',  right_tag '-')  [diagonal, singular values]
#   Vd : legs (bond '-', original right legs...)
#
# Convention for bond legs:
#   U  bond =  QIndex(left_tag,  '+')   — incoming (enters U from the right)
#   S  left  = QIndex(left_tag,  '-')   — outgoing (leaves S to the left)
#   S  right = QIndex(right_tag, '-')   — outgoing (leaves S to the right)
#   Vd bond  = QIndex(right_tag, '-')   — outgoing (leaves Vd to the left)
#
# Legs of U and Vd that come from the original tensor inherit their QIndex
# properties (itags, lock, plev, green, direction).
#
# Algorithm:
#   1. Assign each leg of q a unique internal tag at lock=1.
#   2. Build fusing isometries aL / aR via getIdentity.
#   3. Contract q_work with aL/aR, yielding the rank-2 bipartition M.
#   4. Per-row SVD on M (sL × sR matrix). Truncate either by cutoff alone
#      or by the Nkeep largest singular values, where missing sectors from
#      M.spaces are treated as explicit zero singular values.
#   5. Build rank-2 U, S, Vd from kept singular values.
#   6. Split fused legs of U and Vd by contracting with conjugate of aL/aR.
#   7. Permute legs to desired order and restore original QIndex properties.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Implement a version that get left_legs by predicates or various keyword arguments
# TODO: Test svd with trunction for QSpace object (This is not rigorous yet)
_svd_sector_qlabels(r, N::Int) = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:N)

_svd_dual_sector(symm, sector) = Tuple(get_dualq(symm[n], sector[n]) for n in 1:length(symm))

function LinearAlgebra.svd(q::QSpace{T, QD, N, RD},
                           left_legs,
                           left_tag ::String = "svdL",
                           right_tag::String = "svdR";
                           cutoff   ::Float64 = 1e-12,
                           Nkeep    ::Union{Nothing, Int} = nothing,
) where {T, QD, N, RD}

    symm = q.symm
    left_legs  = collect(Int, left_legs)
    right_legs = [l for l in 1:QD if l ∉ left_legs]
    NL, NR = length(left_legs), length(right_legs)
    @assert NL > 0 "left_legs must not be empty"
    @assert NR > 0 "right_legs must not be empty"
    @assert isnothing(Nkeep) || Nkeep >= 0 "Nkeep must be non-negative"

    # ── Step 1: stamp unique internal tags on every leg (lock=1) ─────────────
    internal_tags = ntuple(l -> "__svd_leg_$(l)__", QD)
    q_work = QSpace(q.symm, q.rows,
        ntuple(l -> QIndex(internal_tags[l], q.inds[l].dir,
                           q.inds[l].plev, 1, q.inds[l].green), QD),
        q.spaces)  # reuse existing spaces since rows unchanged

    # ── Step 2: build fusing isometries ──────────────────────────────────────
    left_cur  = sort(left_legs)
    right_cur = sort(right_legs)

    # getIdentity((q, leg_idx)...; itags=tag) returns legs directly contractable
    # with the selected legs of q, plus one fused output leg.
    # We pass legs in sorted order so that their ordering is consistent with the
    # free1 ordering produced by the upcoming contractions (which also sorts by index).
    aL = getIdentity(((q_work, l) for l in left_cur)...;  itags=left_tag)
    aR = getIdentity(((q_work, l) for l in right_cur)...; itags=right_tag)

    # ── Step 3: contract q_work → rank-2 bipartition M ───────────────────────
    # Contract q_work over its left-side legs with aL's first NL legs.
    # Result legs: (remaining q_work legs in original order, fused aL leg)
    # The remaining legs from q_work are exactly right_cur, landing at positions
    # 1..NR in the result (free1 = right_cur legs), then fused aL at NR+1.
    M = contract(q_work, Tuple(left_cur), aL, Tuple(1:NL); reduce_lock=false)
    # Now contract M's first NR legs (right_cur legs) with aR's first NR legs.
    M = contract(M, Tuple(1:NR), aR, Tuple(1:NR); reduce_lock=false)
    # Result M is rank-2:
    #   M.inds[1] = aL fused leg  (left_tag,  '-')
    #   M.inds[2] = aR fused leg  (right_tag, '-')
    # M.rows[r].RMT.data has shape (sL, sR, 1,...,1).

    # ── Step 4: collect singular values ───────────────────────────────────────
    # For each row of M the RMT has shape (sL, sR, 1,...,1) since rank-2 CGTs
    # have no outer multiplicity. We SVD the (sL × sR) matrix directly.
    # By default, truncation follows the existing cutoff logic.
    # When Nkeep is provided instead, we keep the Nkeep largest singular values
    # globally, ignore degeneracy, and count sectors missing from M.rows as
    # explicit zero singular values using the space lists of M.
    sv_global_max = 0.0
    all_sv_entries = Tuple{Float64, Int, Int}[]  # (sv, row_idx, sv_idx)
    selected_entries = Tuple{Float64, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]  # (sv, row_idx, sector, sv_idx)
    row_sector_map = Dict{Int, NTuple{N, Tuple{Vararg{Int}}}}()
    left_space_dims = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()
    right_space_dims = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()

    for (dim, sector) in M.spaces[1]
        left_space_dims[sector] = dim
    end
    for (dim, sector) in M.spaces[2]
        right_space_dims[sector] = dim
    end

    for (ri, r) in enumerate(M.rows)
        rmt    = r.RMT.data                                # (sL, sR, 1,...,1)
        sL, sR = size(rmt, 1), size(rmt, 2)
        sector = _svd_sector_qlabels(r, N)
        row_sector_map[ri] = sector

        # Treat RMT as (sL × sR) matrix.
        mat = reshape(rmt, sL, sR)
        F   = svd(mat)
        isempty(F.S) && continue

        sv_global_max = max(sv_global_max, F.S[1])
        for j in eachindex(F.S)
            push!(all_sv_entries, (F.S[j], ri, j))
        end
    end

    if isnothing(Nkeep)
        # Sort descending by singular value.
        sort!(all_sv_entries; by = x -> -x[1])

        # Apply absolute cutoff.
        thresh = cutoff * sv_global_max
        filter!(x -> x[1] > thresh, all_sv_entries)

        for (sv, ri, j) in all_sv_entries
            push!(selected_entries, (sv, ri, row_sector_map[ri], j))
        end
    else
        for (sv, ri, j) in all_sv_entries
            push!(selected_entries, (sv, ri, row_sector_map[ri], j))
        end

        present_sectors = Set(values(row_sector_map))
        for (dimL, sector) in M.spaces[1]
            sector in present_sectors && continue
            dimR = get(right_space_dims, _svd_dual_sector(symm, sector), 0)
            for j in 1:min(dimL, dimR)
                push!(selected_entries, (0.0, 0, sector, j))
            end
        end

        sort!(selected_entries; by = x -> -x[1])
        resize!(selected_entries, min(Nkeep, length(selected_entries)))
    end

    # Group kept singular-value indices by source row / missing sector.
    keep_per_row = Dict{Int, Vector{Int}}()
    keep_missing = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()
    selected_counts = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()
    for (_, ri, sector, j) in selected_entries
        selected_counts[sector] = get(selected_counts, sector, 0) + 1
        if ri == 0
            keep_missing[sector] = get(keep_missing, sector, 0) + 1
        else
            push!(get!(keep_per_row, ri, Int[]), j)
        end
    end

    # ── Step 5: build rank-2 U, S, Vd QSpaces ─────────────────────────────────
    # For each kept sector (RMT shape is (sL, sR, 1,...,1)):
    #   U  RMT: (sL, chi, 1,...,1)     legs: (left_tag '-', left_tag '+')
    #   S  RMT: (chi, chi, 1,...,1)    legs: (left_tag '-', right_tag '-')
    #   Vd RMT: (chi, sR, 1,...,1)     legs: (right_tag '-', right_tag '+')
    # S inherits M's qlabels and legdir (both legs outgoing).
    # U and Vd have identity CGT parts: legdir=(1,1), same qlabel on both legs.
    # wmat = sqrt(dim) to satisfy CGT part = identity matrix
    rows_U  = row{Float64, 2, N, 2 + N}[]
    rows_S  = row{Float64, 2, N, 2 + N}[]
    rows_Vd = row{Float64, 2, N, 2 + N}[]

    for (ri, keep_idxs) in sort(collect(keep_per_row); by = x -> x[1])
        r        = M.rows[ri]
        rmt      = r.RMT.data
        sL, sR   = size(rmt, 1), size(rmt, 2)
        mat      = reshape(rmt, sL, sR)
        F        = svd(mat)
        k        = sort(keep_idxs)
        chi      = length(k)

        Umat  = F.U[:, k]                              # (sL, chi)
        Svals = F.S[k]                                 # (chi,)
        Vtmat = F.Vt[k, :]                             # (chi, sR)

        rmt_U  = QTensor(reshape(Umat,                       sL,  chi, ones(Int, N)...))
        rmt_S  = QTensor(reshape(Matrix(Diagonal(Svals)),    chi, chi, ones(Int, N)...))
        rmt_Vd = QTensor(reshape(transpose(Vtmat),           sR, chi,  ones(Int, N)...))

        # U: identity CGT on left_ql, legdir=(1,1)
        cgrs_U  = ntuple(N) do n
            cgr_M   = r.cgrs[n]
            left_ql = cgr_M.qlabels[cgr_M.cgp[1]]
            dim_n   = dimension(symm[n], left_ql)
            wmat_n  = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (left_ql, left_ql), wmat_n, (2, 1), (1, 1))
        end
        # S: same qlabels and legdir as M (both legs outgoing)
        cgrs_S  = ntuple(N) do n
            cgr_M    = r.cgrs[n]
            left_ql  = cgr_M.qlabels[cgr_M.cgp[1]]
            dim_n    = dimension(symm[n], left_ql)
            wmat_n   = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], cgr_M.qlabels, wmat_n, cgr_M.cgp, (0, 2))
        end
        # Vd: identity CGT on right_ql, legdir=(1,1)
        cgrs_Vd = ntuple(N) do n
            cgr_M    = r.cgrs[n]
            right_ql = cgr_M.qlabels[cgr_M.cgp[2]]
            dim_n    = dimension(symm[n], right_ql)
            wmat_n   = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (right_ql, right_ql), wmat_n, (2, 1), (1, 1))
        end

        push!(rows_U,  row(cgrs_U,  rmt_U))
        push!(rows_S,  row(cgrs_S,  rmt_S))
        push!(rows_Vd, row(cgrs_Vd, rmt_Vd))
    end

    for (sector, chi) in sort(collect(keep_missing); by = x -> x[1])
        sL = left_space_dims[sector]
        right_sector = _svd_dual_sector(symm, sector)
        sR = right_space_dims[right_sector]

        Umat = Matrix{Float64}(I, sL, sL)[:, 1:chi]
        Vmat = Matrix{Float64}(I, sR, sR)[:, 1:chi]

        rmt_U = QTensor(reshape(Umat, sL, chi, ones(Int, N)...))
        rmt_Vd = QTensor(reshape(Vmat, sR, chi, ones(Int, N)...))

        cgrs_U = ntuple(N) do n
            left_ql = sector[n]
            dim_n = dimension(symm[n], left_ql)
            wmat_n = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (left_ql, left_ql), wmat_n, (2, 1), (1, 1))
        end

        cgrs_Vd = ntuple(N) do n
            right_ql = right_sector[n]
            dim_n = dimension(symm[n], right_ql)
            wmat_n = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (right_ql, right_ql), wmat_n, (2, 1), (1, 1))
        end

        push!(rows_U, row(cgrs_U, rmt_U))
        push!(rows_Vd, row(cgrs_Vd, rmt_Vd))
    end

    bond_splist = eltype(M.spaces[1])[]
    for (_, qlabels) in M.spaces[1]
        count = get(selected_counts, qlabels, 0)
        count == 0 && continue
        push!(bond_splist, (count, qlabels))
    end
    dual_bond_splist = eltype(M.spaces[2])[
        (count, _svd_dual_sector(symm, qlabels)) for (count, qlabels) in bond_splist
    ]
    sort!(dual_bond_splist; by = x -> x[2])

    inds_U  = (QIndex(left_tag,  '-'), QIndex(left_tag,  '+'))
    inds_S  = (QIndex(left_tag,  '-'), QIndex(right_tag, '-'))
    inds_Vd = (QIndex(right_tag, '-'), QIndex(right_tag, '+'))
    
    spaces_U  = (M.spaces[1], bond_splist)
    spaces_S  = (bond_splist, dual_bond_splist)
    spaces_Vd = (M.spaces[2], dual_bond_splist)

    U_rank2  = QSpace(symm, rows_U,  inds_U,  spaces_U)
    S        = QSpace(symm, rows_S,  inds_S,  spaces_S)
    Vd_rank2 = QSpace(symm, rows_Vd, inds_Vd, spaces_Vd)

    # ── Step 6: split fused legs of U and Vd ──────────────────────────────────
    # Contract U_rank2's fused left leg (leg 1) with aL's fused output leg (leg NL+1)
    # to recover the NL original left legs.
    # Contract Vd_rank2's fused right leg (leg 2) with aR's fused output leg (leg NR+1)
    # to recover the NR original right legs.
    # (QIndex direction matching is not enforced by contract; legs are given explicitly.)
    U_split  = contract(U_rank2,  (1,),     aL', (NL + 1,); reduce_lock=false)
    Vd_split = contract(Vd_rank2, (1,),     aR', (NR + 1,); reduce_lock=false)
    # After splitting:
    #   U_split  legs: [bond (left_tag '+'), aL split legs 1..NL]
    #   Vd_split legs: [Vd bond (right_tag '-'), aR split legs 1..NR]

    # ── Step 7: permute legs to desired order and restore original QIndex properties ──
    # Desired order:
    #   U  : (left_legs[1], ..., left_legs[NL], bond)
    #   Vd : (bond, right_legs[1], ..., right_legs[NR])
    # After splitting, legs have internal tags like "__svd_leg_{orig}__"
    # and the bond leg has left_tag or right_tag.
    
    # Build permutation for U_split: desired order is (left_legs..., bond)
    u_perm = Int[]
    for orig in left_legs
        push!(u_perm, findleg(U_split, idx -> idx.itags == internal_tags[orig]))
    end
    push!(u_perm, findleg(U_split, idx -> idx.itags == left_tag))  # bond leg at the end

    # Build permutation for Vd_split: desired order is (bond, right_legs...)
    vd_perm = Int[]
    push!(vd_perm, findleg(Vd_split, idx -> idx.itags == right_tag))  # bond leg first
    for orig in right_legs
        push!(vd_perm, findleg(Vd_split, idx -> idx.itags == internal_tags[orig]))
    end
    
    # Apply permutations
    U_final  = permutedims(U_split, Tuple(u_perm))
    Vd_final = permutedims(Vd_split, Tuple(vd_perm))
    
    # Restore original QIndex properties (itags, lock, plev, green, dir) for non-bond legs
    # U legs 1:NL inherit from original left_legs, leg NL+1 is the bond
    u_inds_final = ntuple(NL + 1) do i
        if i <= NL
            orig = left_legs[i]
            q.inds[orig]  # inherit all properties from original
        else
            QIndex(left_tag, '+')  # bond leg: incoming into U
        end
    end
    
    # Vd leg 1 is the bond, legs 2:NR+1 inherit from original right_legs
    vd_inds_final = ntuple(NR + 1) do i
        if i == 1
            QIndex(right_tag, '+')  # bond leg: outgoing from Vd
        else
            orig = right_legs[i - 1]
            q.inds[orig]  # inherit all properties from original
        end
    end
    
    # Reconstruct QSpaces with final indices (spaces already permuted correctly)
    U  = QSpace(symm, U_final.rows,  u_inds_final,  U_final.spaces)
    Vd = QSpace(symm, Vd_final.rows, vd_inds_final, Vd_final.spaces)
    
    return U, S, Vd
end
