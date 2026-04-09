# ─── eigen ────────────────────────────────────────────────────────────────────
#
# Perform symmetry-adapted eigendecomposition of a rank-2 QSpace object.
#
# Arguments:
#   q          : QSpace{T, 2, N, RD} to decompose (rank 2, one incoming and one outgoing leg)
#   eig_tag    : itag for the new eigenvector bond leg (default "eig")
#   hermitian  : if true (default), treat each RMT block as Hermitian
#
# Returns `EigenResult(V, D, nothing, eig_list)` where:
#   D        : diagonal QSpace with eigenvalues, legs (eig_tag '-', eig_tag '+')
#   V        : eigenvector QSpace, legs (original outgoing leg, eig_tag '+')
#   eig_list : Vector{Tuple{eigenval_type, Int, sector_qlabels, sector_index}}
#              — (eigenvalue, degeneracy, sector, in-sector index) entries
#              sorted by ascending eigenvalue (real for Hermitian, abs for non-Hermitian).
#              degeneracy = ∏_n dim(irep at leg 1, symm n) for the sector.
#
# Convention:
#   A = V * D * V'   (Hermitian; eigenvectors are orthonormal columns of V)
#
# ─────────────────────────────────────────────────────────────────────────────

_eig_sector_qlabels(r, N) = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[1]] for n in 1:N)

_eig_sort_value(ev::Real, hermitian::Bool) = hermitian ? real(ev) : abs(ev)
_eig_sort_value(ev::Complex, hermitian::Bool) = hermitian ? real(ev) : abs(ev)

struct EigenResult{TV, TD, TVI, TL}
    V::TV
    D::TD
    V_inv::TVI
    eig_list::TL
end

function EigenResult(V::QSpace, D::QSpace, V_inv, eig_list)
    return EigenResult{typeof(V), typeof(D), typeof(V_inv), typeof(eig_list)}(
        V, D, V_inv, eig_list)
end

function _eig_identity_cgrs(symm::NTuple{N, Any},
                            sector_qlabels::NTuple{N, Tuple{Vararg{Int}}},
                            cgp::NTuple{2, Int}) where {N}
    return ntuple(N) do n
        ql = sector_qlabels[n]
        dim_n = dimension(symm[n], ql)
        wmat_n = QTensor([sqrt(Float64(dim_n));;])
        CGR(symm[n], (ql, ql), wmat_n, cgp, (1, 1))
    end
end

function _append_missing_eig_sectors!(symm::NTuple{N, Any},
                                      spaces,
                                      covered,
                                      rows_D,
                                      rows_V,
                                      rows_Vinv,
                                      eig_list,
                                      cgp::NTuple{2, Int},
                                      ::Type{T_out}) where {N, T_out}
    for (sector_qlabels, dim) in spaces
        sector_qlabels ∈ covered && continue

        cgt_dim = prod(dimension(symm[n], sector_qlabels[n]) for n in 1:N)
        for j in 1:dim
            push!(eig_list, (zero(T_out), cgt_dim, sector_qlabels, j))
        end

        cgrs = _eig_identity_cgrs(symm, sector_qlabels, cgp)
        rmt_eye = QTensor(reshape(Matrix{T_out}(I, dim, dim), dim, dim, ones(Int, N)...))

        push!(rows_V, row(cgrs, rmt_eye))
        if !isnothing(rows_Vinv)
            push!(rows_Vinv, row(cgrs, rmt_eye))
        end
    end
end

# TODO: This is not optimized.
function _renumber_eig_entries(eig_entries)
    isempty(eig_entries) && return copy(eig_entries)
    sector_indices = Dict{typeof(eig_entries[1][3]), Vector{Int}}()
    for entry in eig_entries
        push!(get!(sector_indices, entry[3], Int[]), entry[4])
    end

    sector_maps = Dict{typeof(eig_entries[1][3]), Dict{Int, Int}}()
    for (sector, idxs) in sector_indices
        sector_maps[sector] = Dict(old_idx => new_idx for (new_idx, old_idx) in enumerate(sort(unique(idxs))))
    end

    out = similar(eig_entries)
    for (i, entry) in enumerate(eig_entries)
        ev, deg, sector, old_idx = entry
        out[i] = (ev, deg, sector, sector_maps[sector][old_idx])
    end
    return out
end

_retag_qindex(idx::QIndex, tag::AbstractString) = QIndex(tag, idx.dir, idx.plev, idx.lock, idx.green)

function _retag_eigen_result(result::EigenResult, eig_tag::AbstractString)
    d_inds = (_retag_qindex(result.D.inds[1], eig_tag), _retag_qindex(result.D.inds[2], eig_tag))
    D = QSpace(result.D, d_inds)

    v_inds = (result.V.inds[1], _retag_qindex(result.V.inds[2], eig_tag))
    V = QSpace(result.V, v_inds)

    V_inv = if isnothing(result.V_inv)
        nothing
    else
        vinv_inds = (_retag_qindex(result.V_inv.inds[1], eig_tag), result.V_inv.inds[2])
        QSpace(result.V_inv, vinv_inds)
    end

    return EigenResult(V, D, V_inv, result.eig_list)
end

function _select_eig_rows(template::QSpace{T, 2, N, RD},
                          picks::Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}};
                          mode::Symbol) where {T, N, RD}
    rows_out = eltype(template.rows)[]
    sector_counts = Dict{NTuple{N, Tuple{Vararg{Int}}}, Int}()

    for (sector, idxs) in picks
        unique_count = length(unique(sort(idxs)))
        unique_count == 0 && continue
        sector_counts[sector] = unique_count
    end

    for r in template.rows
        sector = _eig_sector_qlabels(r, N)
        idxs = get(picks, sector, Int[])
        isempty(idxs) && continue

        idxs_sorted = sort(idxs)
        rmt = r.RMT.data
        s1, s2 = size(rmt, 1), size(rmt, 2)
        mat = reshape(rmt, s1, s2)

        rmt_new = if mode == :diag
            QTensor(reshape(mat[idxs_sorted, idxs_sorted], length(idxs_sorted), length(idxs_sorted), ones(Int, N)...))
        elseif mode == :cols
            QTensor(reshape(mat[:, idxs_sorted], s1, length(idxs_sorted), ones(Int, N)...))
        elseif mode == :rows
            QTensor(reshape(mat[idxs_sorted, :], length(idxs_sorted), s2, ones(Int, N)...))
        else
            error("Unknown eig row selection mode: $mode")
        end

        push!(rows_out, row(r.cgrs, rmt_new))
    end

    function build_selected_space(base_space, dim_fn)
        selected = eltype(base_space)[]
        for (sector, dim) in base_space
            haskey(sector_counts, sector) || continue
            push!(selected, (sector, dim_fn(dim, sector_counts[sector])))
        end
        return selected
    end

    spaces_out = if mode == :diag
        (
            build_selected_space(template.spaces[1], (_, count) -> count),
            build_selected_space(template.spaces[2], (_, count) -> count),
        )
    elseif mode == :cols
        (
            copy(template.spaces[1]),
            build_selected_space(template.spaces[2], (_, count) -> count),
        )
    elseif mode == :rows
        (
            build_selected_space(template.spaces[1], (_, count) -> count),
            copy(template.spaces[2]),
        )
    else
        error("Unknown eig row selection mode: $mode")
    end

    return QSpace(template.symm, rows_out, template.inds, spaces_out)
end

function _effective_eigen_keep_count(eig_entries,
                                     Nkeep::Integer,
                                     tol::Real;
                                     hermitian::Bool)
    nkeep_eff = min(Nkeep, length(eig_entries))
    if tol > 0 && 0 < nkeep_eff < length(eig_entries)
        extra = min(length(eig_entries) - nkeep_eff, max(1, ceil(Int, Nkeep * Float64(tol))))
        window_end = nkeep_eff + extra
        sort_vals = [_eig_sort_value(eig_entries[i][1], hermitian) for i in 1:window_end]
        if length(sort_vals) > 1
            _, nkeep_eff = findmax(diff(sort_vals))
        end
    end
    return nkeep_eff
end

function _split_eigen_result(result::EigenResult, Nkeep::Integer;
                             hermitian::Bool = isnothing(result.V_inv))
    return _split_eigen_result(result, Nkeep, 0.0; hermitian=hermitian)
end

function _split_eigen_result(result::EigenResult,
                             Nkeep::Integer,
                             tol::Real;
                             hermitian::Bool = isnothing(result.V_inv))
    @assert Nkeep >= 0 "Nkeep must be non-negative"
    @assert isfinite(tol) "tol must be finite"

    eig_entries = copy(result.eig_list)
    sort!(eig_entries; by = x -> _eig_sort_value(x[1], hermitian))

    nkeep_eff = _effective_eigen_keep_count(eig_entries, Nkeep, tol; hermitian=hermitian)
    kept_entries = eig_entries[1:nkeep_eff]
    discarded_entries = eig_entries[nkeep_eff+1:end]

    N = length(result.D.symm)
    kept_picks = Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}}()
    for entry in kept_entries
        push!(get!(kept_picks, entry[3], Int[]), entry[4])
    end

    discarded_picks = Dict{NTuple{N, Tuple{Vararg{Int}}}, Vector{Int}}()
    for entry in discarded_entries
        push!(get!(discarded_picks, entry[3], Int[]), entry[4])
    end

    Vkeep = _select_eig_rows(result.V, kept_picks; mode=:cols)
    Vdiscard = _select_eig_rows(result.V, discarded_picks; mode=:cols)
    Dkeep = _select_eig_rows(result.D, kept_picks; mode=:diag)
    Ddiscard = _select_eig_rows(result.D, discarded_picks; mode=:diag)

    Vinv_keep = isnothing(result.V_inv) ? nothing : _select_eig_rows(result.V_inv, kept_picks; mode=:rows)
    Vinv_discard = isnothing(result.V_inv) ? nothing : _select_eig_rows(result.V_inv, discarded_picks; mode=:rows)

    kept = EigenResult(Vkeep, Dkeep, Vinv_keep, _renumber_eig_entries(kept_entries))
    discarded = EigenResult(Vdiscard, Ddiscard, Vinv_discard, _renumber_eig_entries(discarded_entries))
    return kept, discarded
end

function LinearAlgebra.eigen(q::QSpace{T, 2, N, RD},
                             eig_tag::AbstractString = "eig";
                             hermitian::Bool = true) where {T, N, RD}

    symm = q.symm

    # ── Validate input ───────────────────────────────────────────────────────
    @assert length(q.inds) == 2 "eigen requires a rank-2 QSpace"
    dirs = (q.inds[1].dir, q.inds[2].dir)
    @assert (dirs == ('+', '-') || dirs == ('-', '+')) "eigen requires one incoming ('+') and one outgoing ('-') leg"
    @assert q.spaces[1] == q.spaces[2] "eigen: both legs of input QSpace must have the same space list (same sectors and dimensions)"
    out_leg = dirs[1] == '-' ? 1 : 2
    cgp = dirs == ('+', '-') ? (1, 2) : (2, 1)

    # ── Decompose each row ───────────────────────────────────────────────────
    T_out    = hermitian ? Float64 : Complex{Float64}
    rows_D   = row{T_out, 2, N, 2 + N}[]
    rows_V   = row{T_out, 2, N, 2 + N}[]
    # Eigenvalue, degeneracy, sector qlabels, in-sector index
    eig_list = Tuple{T_out, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]

    for r in q.rows
        rmt = r.RMT.data
        sL, sR = size(rmt, 1), size(rmt, 2)
        @assert sL == sR "eigen: RMT must be square for eigendecomposition, got ($sL, $sR)"

        mat = reshape(rmt, sL, sR)

        # TODO: Need to fix it. Hermitian matrix can have complex entries
        if hermitian
            F = eigen(Hermitian(mat))
            eigenvalues  = T_out.(F.values)
            eigenvectors = T_out.(F.vectors)
        else
            F = eigen(mat)
            eigenvalues  = T_out.(F.values)
            eigenvectors = T_out.(F.vectors)
        end

        chi = length(eigenvalues)
        sector_qlabels = _eig_sector_qlabels(r, N)

        # Degeneracy = product of irrep dims across all symmetries for this sector
        cgt_dim = prod(
            dimension(symm[n], r.cgrs[n].qlabels[r.cgrs[n].cgp[1]])
            for n in 1:N)
        for (j, ev) in enumerate(eigenvalues)
            push!(eig_list, (ev, cgt_dim, sector_qlabels, j))
        end

        rmt_D = QTensor(reshape(Matrix(Diagonal(eigenvalues)), chi, chi, ones(Int, N)...))
        rmt_V = QTensor(reshape(eigenvectors, sL, chi, ones(Int, N)...))

        cgrs_D = ntuple(N) do n
            cgr_orig = r.cgrs[n]
            ql = cgr_orig.qlabels[1]
            dim_n = dimension(symm[n], ql)
            wmat_n = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (ql, ql), wmat_n, cgp, (1, 1))
        end

        cgrs_V = ntuple(N) do n
            cgr_orig = r.cgrs[n]
            ql = cgr_orig.qlabels[1]
            dim_n = dimension(symm[n], ql)
            wmat_n = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (ql, ql), wmat_n, cgp, (1, 1))
        end

        push!(rows_D, row(cgrs_D, rmt_D))
        push!(rows_V, row(cgrs_V, rmt_V))
    end

    covered = Set(_eig_sector_qlabels(r, N) for r in q.rows)
    _append_missing_eig_sectors!(symm, q.spaces[1], covered, rows_D, rows_V, nothing, eig_list, cgp, T_out)

    # Sort eig_list ascending (real part for Hermitian, absolute value otherwise)
    sort!(eig_list; by = x -> _eig_sort_value(x[1], hermitian))

    # ── Inherit spaces from original q ──────────────────────────────────────
    # Both legs share the same space (asserted above).
    spaces_D = (q.spaces[1], q.spaces[1])
    spaces_V = (q.spaces[1], q.spaces[1])

    # ── Build output QIndex tuples ───────────────────────────────────────────
    inds_D = (QIndex(eig_tag, dirs[1]), QIndex(eig_tag, dirs[2]))

    orig_out_ind = q.inds[out_leg]
    inds_V = (QIndex(orig_out_ind.itags, dirs[1], orig_out_ind.plev, orig_out_ind.lock, orig_out_ind.green),
              QIndex(eig_tag, dirs[2]))

    D = QSpace(symm, rows_D, inds_D, spaces_D)
    V = QSpace(symm, rows_V, inds_V, spaces_V)

    return EigenResult(V, D, nothing, eig_list)
end

# ─── eigen_full ───────────────────────────────────────────────────────────────
#
# Non-Hermitian eigendecomposition that also returns V_inv.
#
# Returns `EigenResult(V, D, Vinv, eig_list)` where:
#   D        : diagonal QSpace of eigenvalues (complex), legs (eig_tag '-', eig_tag '+')
#   V        : right-eigenvector QSpace, legs (original outgoing leg, eig_tag '+')
#   Vinv     : inverse of V, legs (eig_tag '-', original incoming leg)
#   eig_list : Vector{Tuple{ComplexF64, Int, sector_qlabels, sector_index}}
#              — (eigenvalue, degeneracy, sector, in-sector index) entries
#              sorted ascending by |eigenvalue|.
#
# Satisfies A = V * D * Vinv row-by-row.
# ────────────────────────────────────────────────────────────────────────────
function eigen_full(q::QSpace{T, 2, N, RD},
                    eig_tag::AbstractString = "eig") where {T, N, RD}

    symm = q.symm

    @assert length(q.inds) == 2 "eigen_full requires a rank-2 QSpace"
    dirs = (q.inds[1].dir, q.inds[2].dir)
    @assert (dirs == ('+', '-') || dirs == ('-', '+')) "eigen_full requires one incoming ('+') and one outgoing ('-') leg"
    @assert q.spaces[1] == q.spaces[2] "eigen_full: both legs of input QSpace must have the same space list (same sectors and dimensions)"
    cgp = dirs == ('+', '-') ? (1, 2) : (2, 1)

    in_leg  = dirs[1] == '+' ? 1 : 2
    out_leg = dirs[1] == '-' ? 1 : 2

    T_out     = ComplexF64
    rows_D    = row{T_out, 2, N, 2 + N}[]
    rows_V    = row{T_out, 2, N, 2 + N}[]
    rows_Vinv = row{T_out, 2, N, 2 + N}[]
    eig_list  = Tuple{T_out, Int, NTuple{N, Tuple{Vararg{Int}}}, Int}[]

    for r in q.rows
        rmt = r.RMT.data
        sL, sR = size(rmt, 1), size(rmt, 2)
        @assert sL == sR "eigen_full: RMT must be square"

        mat = reshape(rmt, sL, sR)

        F = eigen(mat)
        eigenvalues      = T_out.(F.values)
        eigenvectors     = T_out.(F.vectors)
        eigenvectors_inv = T_out.(inv(F.vectors))

        chi = length(eigenvalues)
        sector_qlabels = _eig_sector_qlabels(r, N)

        cgt_dim = prod(
            dimension(symm[n], r.cgrs[n].qlabels[r.cgrs[n].cgp[1]])
            for n in 1:N)
        for (j, ev) in enumerate(eigenvalues)
            push!(eig_list, (ev, cgt_dim, sector_qlabels, j))
        end

        rmt_D    = QTensor(reshape(Matrix(Diagonal(eigenvalues)), chi, chi, ones(Int, N)...))
        rmt_V    = QTensor(reshape(eigenvectors, sL, chi, ones(Int, N)...))
        rmt_Vinv = QTensor(reshape(eigenvectors_inv, chi, sL, ones(Int, N)...))

        cgrs_D = ntuple(N) do n
            cgr_orig = r.cgrs[n]
            ql = cgr_orig.qlabels[1]
            dim_n = dimension(symm[n], ql)
            wmat_n = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (ql, ql), wmat_n, cgp, (1, 1))
        end

        cgrs_V = ntuple(N) do n
            cgr_orig = r.cgrs[n]
            ql = cgr_orig.qlabels[1]
            dim_n = dimension(symm[n], ql)
            wmat_n = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (ql, ql), wmat_n, cgp, (1, 1))
        end

        cgrs_Vinv = ntuple(N) do n
            cgr_orig = r.cgrs[n]
            ql = cgr_orig.qlabels[1]
            dim_n = dimension(symm[n], ql)
            wmat_n = QTensor([sqrt(Float64(dim_n));;])
            CGR(symm[n], (ql, ql), wmat_n, cgp, (1, 1))
        end

        push!(rows_D,    row(cgrs_D,    rmt_D))
        push!(rows_V,    row(cgrs_V,    rmt_V))
        push!(rows_Vinv, row(cgrs_Vinv, rmt_Vinv))
    end

    covered = Set(_eig_sector_qlabels(r, N) for r in q.rows)
    _append_missing_eig_sectors!(symm, q.spaces[1], covered, rows_D, rows_V, rows_Vinv, eig_list, cgp, T_out)

    # Sort by ascending |eigenvalue|
    sort!(eig_list; by = x -> _eig_sort_value(x[1], false))

    # ── Inherit spaces from original q ──────────────────────────────────────
    # Both legs have the same space (asserted above), so q.spaces[1] is used
    # throughout.
    spaces_D    = (q.spaces[1], q.spaces[1])
    spaces_V    = (q.spaces[1], q.spaces[1])
    spaces_Vinv = (q.spaces[1], q.spaces[1])

    # ── Build QIndex tuples ───────────────────────────────────────
    inds_D = (QIndex(eig_tag, dirs[1]), QIndex(eig_tag, dirs[2]))

    orig_out_ind = q.inds[out_leg]
    inds_V = (QIndex(orig_out_ind.itags, dirs[1], orig_out_ind.plev, orig_out_ind.lock, orig_out_ind.green),
              QIndex(eig_tag, dirs[2]))

    orig_in_ind = q.inds[in_leg]
    inds_Vinv = (QIndex(eig_tag, dirs[1]),
                 QIndex(orig_in_ind.itags, dirs[2], orig_in_ind.plev, orig_in_ind.lock, orig_in_ind.green))

    D    = QSpace(symm, rows_D,    inds_D,    spaces_D)
    V    = QSpace(symm, rows_V,    inds_V,    spaces_V)
    Vinv = QSpace(symm, rows_Vinv, inds_Vinv, spaces_Vinv)

    return EigenResult(V, D, Vinv, eig_list)
end

"""
    discard_eigen(result::EigenResult, Nkeep, tol, kept_tag, discarded_tag; hermitian=isnothing(result.V_inv))

Keep the `Nkeep` smallest eigenvalues, ignoring degeneracy, and return two
`EigenResult` objects: the kept part and the discarded part.

If `tol > 0`, inspect up to `ceil(Int, Nkeep * tol)` additional eigenvalues and
move the truncation to the largest adjacent gap within that enlarged window.
"""
function discard_eigen(result::EigenResult,
                       Nkeep::Integer,
                       tol::Real,
                       kept_tag::AbstractString = "eigK",
                       discarded_tag::AbstractString = "eigD";
                       hermitian::Bool = isnothing(result.V_inv))
    kept, discarded = _split_eigen_result(result, Nkeep, tol; hermitian=hermitian)
    return _retag_eigen_result(kept, kept_tag), _retag_eigen_result(discarded, discarded_tag)
end

function discard_eigen(result::EigenResult,
                       Nkeep::Integer,
                       kept_tag::AbstractString = "eigK",
                       discarded_tag::AbstractString = "eigD";
                       hermitian::Bool = isnothing(result.V_inv))
    return discard_eigen(result, Nkeep, 0.1, kept_tag, discarded_tag; hermitian=hermitian)
end

