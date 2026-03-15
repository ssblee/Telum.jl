using Printf
using LinearAlgebra
include("QTensor.jl")
include("utils.jl")
include("localspaces/localspaces.jl")

# ─── Tag string helpers ───────────────────────────────────────────────────────
#
# Tags are stored as a single comma-separated String sorted alphabetically,
# e.g. "bond,site,u1" (ITensor convention). Empty string means no tags.
# Individual tags contain no commas or whitespace.
#
# ─────────────────────────────────────────────────────────────────────────────

# Split tag string into sorted vector of non-empty individual tags.
_parse_tags(tags::String) = sort!(filter!(!isempty, strip.(split(tags, ','))))

# Canonical form: sorted, comma-joined, no spaces.
_normalize_tags(tags::String) = join(_parse_tags(tags), ',')

# True iff all comma-separated tags in `query` are present in `base`.
function _has_tags(base::String, query::String)
    bset = Set(_parse_tags(base))
    return all(t -> t ∈ bset, _parse_tags(query))
end

# Add tags from `newtags` to `base`; result is sorted and deduplicated.
_add_tags(base::String, newtags::String) =
    join(sort!(unique!(vcat(_parse_tags(base), _parse_tags(newtags)))), ',')

# Remove every tag listed in `rmtags` from `base`.
function _remove_tags(base::String, rmtags::String)
    rm = Set(_parse_tags(rmtags))
    return join(filter(t -> t ∉ rm, _parse_tags(base)), ',')
end

# Remove tags listed in `old` and add tags listed in `new` to `base`.
# (Applied regardless of whether `old` tags are present — use the
#  `itags` selector to restrict which legs are affected.)
function _replace_tags(base::String, old::String, new_tags::String)
    rm = Set(_parse_tags(old))
    remaining = filter(t -> t ∉ rm, _parse_tags(base))
    append!(remaining, _parse_tags(new_tags))
    return join(sort!(unique!(remaining)), ',')
end


struct QIndex
    # Tags associated with the leg, similar to ITensor
    itags::String
    # Direction of the leg, '+' for incoming, '-' for outgoing
    dir::Char
    # Prime level similar to ITensor
    plev::Int
    # Lock level. Cannot cntracted if lock > 0. 
    # Decrease lock level by 1 after contraction.
    lock::Int
    # If true, print the tag in green when displaying a QSpace.
    green::Bool

    QIndex(itags::String, dir::Char, plev::Int=0, lock::Int=0, green::Bool=false) = new(_normalize_tags(itags), dir, plev, lock, green)
end

QIndex(dir::Char, plev::Int=0, lock::Int=0) = QIndex("", dir, plev, lock)

# Two QIndex objects are equal if they share the same itags, dir, plev, and green
# (lock is intentionally ignored — it is a transient contraction counter).
Base.:(==)(a::QIndex, b::QIndex) = a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.green == b.green
Base.isequal(a::QIndex, b::QIndex) = (a == b)
Base.hash(a::QIndex, h::UInt) = hash((a.itags, a.dir, a.plev, a.green), h)

to_incoming(idx::QIndex) = QIndex(idx.itags, '+', idx.plev, idx.lock, idx.green)
to_outgoing(idx::QIndex) = QIndex(idx.itags, '-', idx.plev, idx.lock, idx.green)
change_dir(idx::QIndex)  = QIndex(idx.itags, idx.dir == '+' ? '-' : '+', idx.plev, idx.lock, idx.green)
green(idx::QIndex) = QIndex(idx.itags, idx.dir, idx.plev, idx.lock, true)

struct CGR{QD, NZ}
    symm::Any   # symmetry type, e.g. SU{2}, U1
    qlabels::NTuple{QD, NTuple{NZ, Int}}
    wmat::QTensor{Float64, 2}
    cgp::NTuple{QD, Int}
    # (# incoming legs, # outgoing legs); sum == QD
    legdir::Tuple{Int, Int}  
end

# Constructor for QD=0 case: infer NZ from the symmetry type
function CGR(symm::Type{S}, qlabels::Tuple{}, wmat::QTensor{Float64, 2}, 
             cgp::Tuple{}, legdir::Tuple{Int, Int}) where {S}
    NZ = nzops(S)
    CGR{0, NZ}(symm, qlabels, wmat, cgp, legdir)
end

struct row{T, QD, N, RD}
    cgrs::NTuple{N, CGR{QD}}
    RMT::QTensor{T, RD}
end

# ─── Pretty-printing helpers ────────────────────────────────────────────────

# Per-symmetry field widths: width for symmetry n = digits of max |label| in
# that symmetry, plus 1 if the symmetry is U1 (may have negative labels).
# symm[n] is the symmetry type; pass nothing to treat all as potentially negative.
function _label_widths(cgrs::Tuple{Vararg{<:CGR}},
                       symm::Union{Tuple, Nothing} = nothing)
    N = length(cgrs)
    map(1:N) do n
        cgr  = cgrs[n]
        s    = isnothing(symm) ? cgr.symm : symm[n]
        vals = (v for ql in cgr.qlabels for v in ql)
        mxabs = maximum(abs, vals, init=0)
        needs_sign = isnothing(s) ? any(<(0), vals) : (s <: U1)
        ndigits(max(mxabs, 1)) + (needs_sign ? 1 : 0)
    end
end

# ANSI colors cycled over label positions within one symmetry group (NZ > 1).
# Print q-label block directly to io:  [ leg1_sym1 leg1_sym2 ; leg2_sym1 ; ... ]
# widths[n] is the lpad width for cgrs[n].
# Each cgr's cgp permutation is applied: the l-th displayed leg uses cgrs[n].qlabels[cgp[l]].
function _print_qlabels(io::IO, cgrs::Tuple{Vararg{<:CGR{QD}}}, widths) where {QD}
    N = length(cgrs)
    print(io, "[")
    for l in 1:QD
        l > 1 && print(io, " ;")
        for n in 1:N
            print(io, " ")
            pl = cgrs[n].cgp[l]   # physical leg index after permutation
            for v in cgrs[n].qlabels[pl]
                print(io, lpad(v, widths[n]))
            end
        end
    end
    print(io, " ]")
end

# Print CGT dimensions (irrep dim per leg) for each non-abelian symmetry.
# For symmetry n, the CGT block has shape d_leg1 x d_leg2 x ... where
# d_l = dimension(symm[n], cgrs[n].qlabels[l]).
# symm = nothing → falls back to wmat first-axis size (no type info available).
function _print_cgt_dims(io::IO, cgrs::NTuple{N, CGR{QD}},
                         symm::Union{Tuple, Nothing} = nothing) where {N, QD}
    first = true
    for n in 1:N
        s = isnothing(symm) ? cgrs[n].symm : symm[n]
        !isnothing(s) && isabelian(s) && continue
        if isnothing(s)
            dim_str = string(size(cgrs[n].wmat.data, 1))
        else
            dims = [dimension(s, cgrs[n].qlabels[cgrs[n].cgp[l]]) for l in 1:QD]
            dim_str = join(dims, "x")
        end
        first && print(io, "| ")
        first = false
        print(io, dim_str, "\t")
    end
end

# Format a scalar RMT value as a string with consistent width:
# integers print without decimal point; floats use %#.7g which always
# shows 7 significant digits including trailing zeros (e.g. 3.46410 not 3.4641).
function _fmt_scalar_str(v::Real)
    return @sprintf("%#.7g", v)
end

# Total dimension of the CGT part: product of irrep dimensions across all
# legs and all non-abelian symmetries. Returns an Int.
function _cgt_size_2d(cgrs::NTuple{N, CGR{2}},
                        symm::Tuple) where N
    total = 1
    for n in 1:N
        isabelian(symm[n]) && continue
        total *= dimension(symm[n], cgrs[n].qlabels[1])
    end
    return total
end

# ─── Pretty-printing for row ─────────────────────────────────────────────────
#
#   rmt_dims \t [cgt_dims \t] [ qlabels ]  [\t= val]
#
# ─────────────────────────────────────────────────────────────────────────────
function Base.show(io::IO, r::row{T, QD, N, RD}) where {T, QD, N, RD}
    widths = _label_widths(r.cgrs)
    om_dim = prod(size(r.RMT.data)[QD+1:end]; init=1)
    phys_str = join(size(r.RMT.data)[1:QD], "x")
    om_str   = om_dim > 1 ? " @$om_dim" : ""
    print(io, phys_str, om_str, "\t")
    _print_cgt_dims(io, r.cgrs)          # no symm → show all non-1 CGT dims
    _print_qlabels(io, r.cgrs, widths)
    length(r.RMT.data) == 1 && print(io, "\t= ", _fmt_scalar_str(only(r.RMT.data)))
end

function Base.show(io::IO, ::MIME"text/plain", r::row{T, QD, N, RD}) where {T, QD, N, RD}
    widths = _label_widths(r.cgrs)

    # RMT line: dims, qlabels (cgp-permuted), scalar value if applicable
    om_dim = prod(size(r.RMT.data)[QD+1:end]; init=1)
    phys_str = join(size(r.RMT.data)[1:QD], "x")
    om_str   = om_dim > 1 ? " @$om_dim" : ""
    print(io, "  RMT: ", phys_str, om_str, "\t")
    _print_qlabels(io, r.cgrs, widths)
    length(r.RMT.data) == 1 && print(io, "\t", _fmt_scalar_str(only(r.RMT.data)))
    println(io)

    # Pre-compute column widths for aligned cgr lines.
    # Per-label width for cgr n (raw labels, sign from embedded symm):
    vws = map(r.cgrs) do cgr
        has_neg = cgr.symm <: U1
        mxabs = maximum(abs, (v for ql in cgr.qlabels for v in ql), init=0)
        ndigits(max(mxabs, 1)) + (has_neg ? 1 : 0)
    end
    # Fixed-width prefix: "  SYMNAME  wmat=NxM  "
    prefix_w = maximum(1:N) do n
        cgr = r.cgrs[n]
        sym_str = isnothing(cgr.symm) ? "cgr[$n]" : totxt(cgr.symm)
        length("  $(rpad(sym_str, 4))  wmat=$(join(size(cgr.wmat.data),"x"))  ")
    end

    # cgr lines: aligned prefix, raw qlabels, cgp, scalar wmat
    for n in 1:N
        cgr = r.cgrs[n]
        sym_str = isnothing(cgr.symm) ? "cgr[$n]" : totxt(cgr.symm)
        prefix = "  $(rpad(sym_str, 4))  wmat=$(join(size(cgr.wmat.data),"x"))  "
        print(io, rpad(prefix, prefix_w), "[")
        for (l, ql) in enumerate(cgr.qlabels)
            if l > 1 print(io, l == cgr.legdir[1] + 1 ? " |" : " ;") end
            print(io, " ", join((lpad(v, vws[n]) for v in ql), ""))
        end
        print(io, " ]  cgp=", cgr.cgp)
        length(cgr.wmat.data) == 1 && print(io, "  ", _fmt_scalar_str(only(cgr.wmat.data)))
        println(io)
    end
end

get_qd(::row{T, QD}) where {T, QD} = QD

# The final step of getLocalSpace function.
function get_rows(data::Vector{Tuple{NTuple{QD, NTuple{N, Tuple{Vararg{Int}}}}, Array{T, RD}}},
                  symm::NTuple{N, Any}) where {T, QD, N, RD}

    @assert RD == QD + N; @assert QD == 2 || QD == 3
    rows = Vector{row{T, QD, N, RD}}()
    for (qlabels, block) in data
        wmats = Vector{QTensor{Float64, 2}}()
        for i in 1:N
            wmat, block, _ = svd_leg(block, QD + i)
            push!(wmats, QTensor(wmat))
        end
        RMT = QTensor(block)
        cgrs = CGR{QD}[]
        for i in 1:N
            qforsymm = Tuple(qlabels[j][i] for j in 1:QD)
            # NZ = length(qforsymm[1])
            if QD == 2
                cgp = (1, 2); legdir = (1, 1)
            elseif QD == 3
                cgp = qforsymm[2] <= qforsymm[3] ? (1, 2, 3) : (1, 3, 2); legdir = (1, 2)
            else error("Invalid QSpace tensor dimension for getLocalSpace: $QD") end

            if QD == 3 && qforsymm[2] > qforsymm[3]
                qforsymm = (qforsymm[1], qforsymm[3], qforsymm[2])
            end
            push!(cgrs, CGR(symm[i], qforsymm, wmats[i], cgp, legdir))
        end
        push!(rows, row(Tuple(cgrs), RMT))
    end
    return rows
end

# ─── QSpace invariant checkers ─────────────────────────────────────────────

# Condition 1: for each CGR, if two stored qlabel positions i < j share the
# same qlabel AND the same arrow direction (both incoming or both outgoing),
# the corresponding physical legs must also appear in that order:
#   cgp_inv[i] < cgp_inv[j]   (i.e. the order in CGR.qlabels is preserved
#                               through the cgp mapping to QSpace legs).
function _check_cgr_qlabel_order(cgr::CGR{QD}) where QD
    m, k = cgr.legdir
    # Build cgp_inv: cgp_inv[si] = physical leg l with cgp[l] == si.
    cgp_inv = zeros(Int, QD)
    for l in 1:QD
        cgp_inv[cgr.cgp[l]] = l
    end
    # Incoming stored positions: 1:m
    for i in 1:m, j in i+1:m
        if cgr.qlabels[i] == cgr.qlabels[j]
            @assert cgp_inv[i] < cgp_inv[j] begin
                "CGR qlabel order violated (incoming): stored positions $i < $j " *
                "share qlabel $(cgr.qlabels[i]) but physical legs are " *
                "$(cgp_inv[i]) > $(cgp_inv[j])"
            end
        end
    end
    # Outgoing stored positions: m+1:m+k
    for i in m+1:m+k, j in i+1:m+k
        if cgr.qlabels[i] == cgr.qlabels[j]
            @assert cgp_inv[i] < cgp_inv[j] begin
                "CGR qlabel order violated (outgoing): stored positions $i < $j " *
                "share qlabel $(cgr.qlabels[i]) but physical legs are " *
                "$(cgp_inv[i]) > $(cgp_inv[j])"
            end
        end
    end
end

# Condition 3: an index with empty itags must have lock == 0
function _check_empty_tag_lock(inds::NTuple{QD, QIndex}) where QD
    for idx in inds
        if isempty(idx.itags)
            @assert idx.lock == 0 "QSpace: index with empty itag has nonzero lock=$(idx.lock)"
        end
    end
end

# Condition 2: no two QIndex objects with non-empty itags in the inds tuple
# may be equal (as determined by ==, which compares itags, dir, plev, green).
function _check_unique_inds(inds::NTuple{QD, QIndex}) where QD
    tagged = [idx for idx in inds if !isempty(idx.itags)]
    for i in 1:length(tagged), j in i+1:length(tagged)
        @assert tagged[i] != tagged[j] begin
            "Duplicate QIndex with non-empty itag in QSpace.inds: $(tagged[i])"
        end
    end
end

# Compute the spaces tuple from rows: for each leg, a vector of (dim, qlabels) pairs.
# This is the same information that leginfo extracts, but computed for all legs at once.
function _compute_spaces(rows::Vector{row{T, QD, N, RD}}) where {T, QD, N, RD}
    # For each leg, track seen qlabels to avoid duplicates
    spsets = ntuple(_ -> Set{NTuple{N, Tuple{Vararg{Int}}}}(), QD)
    splists = ntuple(_ -> Vector{Tuple{Int, NTuple{N, Tuple{Vararg{Int}}}}}(), QD)
    
    for r in rows
        for leg in 1:QD
            qlabel_leg = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[leg]] for n in 1:N)
            if qlabel_leg ∉ spsets[leg]
                dim = size(r.RMT.data, leg)
                push!(splists[leg], (dim, qlabel_leg))
                push!(spsets[leg], qlabel_leg)
            end
        end
    end
    
    return splists
end

# T: type of element in the RMT array, can be Float64, ComplexF64, etc.
# QD: The rank of tensor (# of legs), N: The number of symmetries
# RD: The rank of RMT array, which is equal to QD + N
struct QSpace{T, QD, N, RD}
    symm::NTuple{N, Any}
    # Data rows for QSpace object
    rows::Vector{row{T, QD, N, RD}}
    inds::NTuple{QD, QIndex}
    # Space list for each leg: vector of (RMT_dim, qlabels) pairs
    # Similar to leginfo.splist but precomputed for all legs
    spaces::NTuple{QD, Vector{Tuple{Int, NTuple{N, Tuple{Vararg{Int}}}}}}

    # Constructor with explicit spaces (for efficiency when spaces are already known)
    function QSpace(symm::NTuple{N, Any}, 
        rows::Vector{row{T, QD, N, RD}}, 
        inds::NTuple{QD, QIndex},
        spaces::NTuple{QD, Vector{Tuple{Int, NTuple{N, Tuple{Vararg{Int}}}}}}) where {T, QD, N, RD}

        q = new{T, QD, N, RD}(symm, rows, inds, spaces)
        normalize_qspace!(q)
        _drop_small_rows!(q)
        sort_rows!(q)
        for r in q.rows, cgr in r.cgrs
            _check_cgr_qlabel_order(cgr)
        end
        _check_unique_inds(q.inds)
        _check_empty_tag_lock(q.inds)
        return q
    end
end

# Drop rows whose norm² contribution is below cutoff² × total norm² (relative threshold).
# For QD == 2 the effective norm² per row is dim_r * ‖RMT_r‖² (see normalize_qspace!);
# for all other ranks it is simply ‖RMT_r‖².
const QSPACE_ROW_CUTOFF = 1e-14

function _drop_small_rows!(q::QSpace{T, QD, N}; cutoff::Float64 = QSPACE_ROW_CUTOFF) where {T, QD, N}
    rows = q.rows
    isempty(rows) && return

    row_norms_sq = [
        QD == 2 ? _cgt_size_2d(r.cgrs, q.symm) * sum(abs2, r.RMT.data) : sum(abs2, r.RMT.data)
        for r in rows
    ]
    total = sum(row_norms_sq)
    iszero(total) && return   # zero tensor: keep all

    thresh  = (cutoff^2) * total
    keepidx = findall(x -> x > thresh, row_norms_sq)
    length(keepidx) == length(rows) && return   # nothing to drop

    # Filter rows in-place.
    splice!(rows, 1:length(rows), rows[keepidx])
end

# Construct a QSpace with the same rows but with itags replaced.
# itags: NTuple{QD, String} — one tag per leg; all other QIndex fields are preserved.
function QSpace(q::QSpace{T, QD, N, RD}, itags::NTuple{QD, String}) where {T, QD, N, RD}
    new_inds = ntuple(l -> QIndex(itags[l], q.inds[l].dir, q.inds[l].plev,
                                  q.inds[l].lock, q.inds[l].green), QD)
    # spaces remain the same since rows didn't change
    return QSpace(q.symm, q.rows, new_inds, q.spaces)
end

# Construct a QSpace with the same rows but with all QIndex fields replaced.
# inds: NTuple{QD, QIndex} — one full QIndex per leg.
# Arrow directions must match the original QSpace (only itags/lock/plev/green may differ).
function QSpace(q::QSpace{T, QD, N, RD}, inds::NTuple{QD, QIndex}) where {T, QD, N, RD}
    @assert ntuple(l -> inds[l].dir, QD) == ntuple(l -> q.inds[l].dir, QD) "QSpace(q, inds): arrow directions must match the original QSpace on all legs"
    return QSpace(q.symm, q.rows, inds, q.spaces)
end

Base.getindex(q::QSpace, i::Int) = q.rows[i]

# For 0-dimensional QSpace (scalar), q[] returns the unique RMT element.
function Base.getindex(q::QSpace{T, 0, N, N}) where {T, N}
    @assert length(q.rows) == 1 "0D QSpace must have exactly one row"
    @assert length(q.rows[1].RMT.data) == 1 "0D QSpace RMT must be a scalar"
    return only(q.rows[1].RMT.data)
end

# ─── Leg selection utilities ─────────────────────────────────────────────────

"""
    findlegs(q::QSpace, pred::Function) -> Vector{Int}

Find all leg indices where `pred(qindex)` returns true.

# Examples
```julia
findlegs(q, idx -> idx.dir == '-')                    # all outgoing legs
findlegs(q, idx -> occursin("site", idx.itags))       # legs with "site" in tag
findlegs(q, idx -> idx.dir == '-' && idx.plev == 0)   # outgoing, unprimed
```
"""
findlegs(q::QSpace{T, QD}, pred::Function) where {T, QD} = [i for i in 1:QD if pred(q.inds[i])]

"""
    findleg(q::QSpace, pred::Function) -> Int

Find the first leg index where `pred(qindex)` returns true.
Throws an error if no leg matches.

# Examples
```julia
findleg(q, idx -> idx.itags == "bond")   # first leg with exact tag "bond"
findleg(q, idx -> idx.dir == '+')        # first incoming leg
```
"""
function findleg(q::QSpace{T, QD}, pred::Function) where {T, QD}
    for i in 1:QD pred(q.inds[i]) && return i end
    return nothing
end

# Internal: check if a QIndex matches all specified criteria
function _matches_criteria(idx::QIndex; dir=nothing, itags=nothing, plev=nothing, lock=nothing)
    (!isnothing(dir)   && idx.dir != dir)                        && return false
    (!isnothing(itags) && !_has_tags(idx.itags, itags))          && return false
    (!isnothing(plev)  && idx.plev != plev)                      && return false
    (!isnothing(lock)  && idx.lock != lock)                      && return false
    return true
end

"""
    findlegs(q::QSpace; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Find all leg indices matching the specified criteria. Unspecified criteria match any value.
If `rev=true`, the selection is reversed: legs that do *not* match are returned.

# Arguments
- `dir`: Match direction ('+' for incoming, '-' for outgoing)
- `itags`: Match if this string is a substring of the leg's itags
- `plev`: Match exact prime level
- `lock`: Match exact lock level
- `rev`: If `true`, return legs that do *not* satisfy the criteria (default `false`)

# Examples
```julia
findlegs(q; dir='-')                    # all outgoing legs
findlegs(q; itags="site")               # legs with "site" in their tag
findlegs(q; dir='+', plev=0)            # incoming, unprimed legs
findlegs(q; lock=0)                     # non-locked legs
findlegs(q; dir='-', rev=true)          # all legs that are NOT outgoing
```
"""
function findlegs(q::QSpace{T, QD}; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    return [i for i in 1:QD if _matches_criteria(q.inds[i]; dir=dir, itags=itags, plev=plev, lock=lock) ⊻ rev]
end

"""
    findleg(q::QSpace; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Find the first leg index matching the specified criteria.
Returns `nothing` if no leg matches.
If `rev=true`, returns the first leg that does *not* match the criteria.

# Examples
```julia
findleg(q; itags="bond")          # first leg with "bond" in tag
findleg(q; dir='+')               # first incoming leg
findleg(q; lock=0)                # first non-locked leg
findleg(q; dir='+', rev=true)     # first leg that is NOT incoming
```
"""
function findleg(q::QSpace{T, QD}; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    for i in 1:QD
        _matches_criteria(q.inds[i]; dir=dir, itags=itags, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

# ─── Lock utilities ──────────────────────────────────────────────────────────
#
# Lock level semantics:
#   lock > 0  : Cannot be contracted. Decreases by 1 after each contraction.
#   lock == 0 : Normal leg, can be contracted.
#   lock == -1: Permanently locked, never changes and cannot be contracted.
#
# ─────────────────────────────────────────────────────────────────────────────

const LegList = Union{AbstractVector{<:Integer}, Tuple{Vararg{<:Integer}}}

# Internal: apply a lock modification function to selected leg indices.
# legs can be any iterable of integers (Int, Vector, UnitRange, Tuple, etc.)
function _modify_lock(q::QSpace{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_lock = modify_fn(idx.lock)
        new_inds[i] = QIndex(idx.itags, idx.dir, idx.plev, new_lock, idx.green)
    end
    return QSpace(q.symm, q.rows, Tuple(new_inds), q.spaces)
end

# Lock increase function (respects permanent lock)
_lock_inc(current_lock, inc) = current_lock == -1 ? -1 : current_lock + inc

"""
    lock(q::QSpace, leg::Integer; inc::Int=1)

Increase lock level of a single specified leg by `inc` (default 1).
Permanently locked legs (lock=-1) are unchanged.
"""
function lock(q::QSpace, leg::Integer; inc::Int=1)
    return _modify_lock(q, (leg,), lk -> _lock_inc(lk, inc))
end

"""
    lock(q::QSpace, legs::LegList; inc::Int=1)

Increase lock level of the specified legs by `inc` (default 1).
`legs` can be any vector, range, or tuple of integers, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
Permanently locked legs (lock=-1) are unchanged.
"""
function lock(q::QSpace, legs::LegList; inc::Int=1)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::QSpace, pred::Function; inc::Int=1)

Increase lock level of legs satisfying predicate by `inc` (default 1).
"""
function lock(q::QSpace, pred::Function; inc::Int=1)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::QSpace; inc::Int=1, dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev=false)

Increase lock level of legs matching criteria by `inc`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lock(q; dir='-')                  # lock all outgoing legs by 1
lock(q; inc=2, itags="bond")      # lock legs with "bond" in tag by 2
lock(q; lock=0)                   # lock all currently-unlocked legs by 1
lock(q; dir='-', rev=true)        # lock all legs that are NOT outgoing
```
"""
function lock(q::QSpace; inc::Int=1, dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lockp(q::QSpace, leg::Integer)

Permanently lock a single specified leg (set lock=-1).
"""
function lockp(q::QSpace, leg::Integer)
    return _modify_lock(q, (leg,), _ -> -1)
end

"""
    lockp(q::QSpace, legs::LegList)

Permanently lock the specified legs (set lock=-1).
`legs` can be any vector, range, or tuple of integers.
"""
function lockp(q::QSpace, legs::LegList)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::QSpace, pred::Function)

Permanently lock legs satisfying predicate.
"""
function lockp(q::QSpace, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    lockp(q::QSpace; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev=false)

Permanently lock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lockp(q; itags="phys")         # permanently lock physical legs
lockp(q; lock=0)               # permanently lock all currently-unlocked legs
lockp(q; itags="phys", rev=true)  # permanently lock all legs except "phys"
```
"""
function lockp(q::QSpace; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    unlock(q::QSpace, leg::Integer)

Unlock a single specified leg (set lock=0). Also removes permanent lock.
"""
function unlock(q::QSpace, leg::Integer)
    return _modify_lock(q, (leg,), _ -> 0)
end

"""
    unlock(q::QSpace, legs::LegList)

Unlock the specified legs (set lock=0).
`legs` can be any vector, range, or tuple of integers.
"""
function unlock(q::QSpace, legs::LegList)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::QSpace, pred::Function)

Unlock legs satisfying predicate.
"""
function unlock(q::QSpace, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::QSpace; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev=false)

Unlock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
unlock(q; dir='-')             # unlock all outgoing legs
unlock(q; lock=1)              # unlock all legs currently at lock=1
unlock(q; dir='-', rev=true)   # unlock all legs that are NOT outgoing
```
"""
function unlock(q::QSpace; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> 0)
end

# ─── Prime level modification ────────────────────────────────────────────────
#
# ITensor-style prime level modification. Prime level is always non-negative.
# All increment forms use `inc` as an optional keyword argument (default 1).
#
#   prime(q; inc=1, kw...)         – increase plev of selected legs by inc
#   prime(q, leg; inc=1)           – increase plev of a single leg by inc
#   prime(q, legs; inc=1)          – increase plev of specified legs by inc
#   prime(q, pred; inc=1)          – increase plev of legs matching predicate
#   setprime(q, n; kw...)          – set plev of selected legs to n (≥0)
#   setprime(q, legs, n)           – set plev of specified legs to n
#   noprime(q; kw...)              – set plev of selected legs to 0
#   noprime(q, leg)                – set plev of a single leg to 0
#   noprime(q, legs)               – set plev of specified legs to 0
#   noprime(q, pred)               – set plev of legs matching predicate to 0
#
# Keyword selectors for criteria forms (all optional): dir, itags, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

# legs can be any iterable of integers
function _modify_plev(q::QSpace{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = QIndex(idx.itags, idx.dir, modify_fn(idx.plev), idx.lock, idx.green)
    end
    return QSpace(q.symm, q.rows, Tuple(new_inds), q.spaces)
end

"""
    prime(q::QSpace; inc::Int=1, dir, itags, plev, lock, rev=false)

Increase the prime level of matching legs by `inc` (default 1).
Prime level is clamped to 0 from below.
With no keyword arguments, all legs are affected.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
prime(q)                        # all legs: plev += 1
prime(q; inc=2)                 # all legs: plev += 2
prime(q; inc=-1)                # all legs: plev -= 1, clamped to 0
prime(q; dir='+')               # incoming legs only
prime(q; itags="site")          # legs whose tag contains "site"
prime(q; plev=0)                # only currently unprimed legs
prime(q; dir='+', rev=true)     # all legs that are NOT incoming
```
"""
function prime(q::QSpace{T, QD}; inc::Int=1, dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::QSpace, leg::Integer; inc::Int=1)

Increase the prime level of a single specified leg by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, 2)                     # leg 2: plev += 1
prime(q, 2; inc=3)              # leg 2: plev += 3
```
"""
function prime(q::QSpace{T, QD}, leg::Integer; inc::Int=1) where {T, QD}
    return _modify_plev(q, (leg,), p -> max(0, p + inc))
end

"""
    prime(q::QSpace, legs::LegList; inc::Int=1)

Increase the prime level of the specified legs by `inc` (default 1).
Prime level is clamped to 0 from below.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.

# Examples
```julia
prime(q, [1, 3])                # legs 1 and 3: plev += 1
prime(q, 1:2; inc=2)            # legs 1–2: plev += 2
prime(q, (1, 3))                # same as prime(q, [1, 3])
```
"""
function prime(q::QSpace{T, QD}, legs::LegList; inc::Int=1) where {T, QD}
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    prime(q::QSpace, pred::Function; inc::Int=1)

Increase the prime level of legs satisfying predicate by `inc` (default 1).
Prime level is clamped to 0 from below.

# Examples
```julia
prime(q, idx -> idx.dir == '+')          # incoming legs: plev += 1
prime(q, idx -> idx.plev == 0; inc=2)   # unprimed legs: plev += 2
```
"""
function prime(q::QSpace{T, QD}, pred::Function; inc::Int=1) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, p -> max(0, p + inc))
end

"""
    setprime(q::QSpace, n::Int; dir, itags, plev, lock, rev=false)

Set the prime level of matching legs to `n`. `n` must be non-negative.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setprime(q, 2)                  # set all legs to plev=2
setprime(q, 1; dir='-')         # set outgoing legs to plev=1
setprime(q, 0; itags="bond")    # same as noprime(q; itags="bond")
setprime(q, 0; dir='-', rev=true)  # clear prime on all non-outgoing legs
```
"""
function setprime(q::QSpace{T, QD}, n::Int; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> n)
end

"""
    setprime(q::QSpace, legs, n::Int)

Set the prime level of the specified legs to `n`. `n` must be non-negative.
`legs` can be any iterable of leg indices.

# Examples
```julia
setprime(q, [1, 3], 2)          # set legs 1 and 3 to plev=2
setprime(q, 1:2, 0)             # same as noprime(q, 1:2)
```
"""
function setprime(q::QSpace{T, QD}, legs, n::Int) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    return _modify_plev(q, legs, _ -> n)
end

"""
    noprime(q::QSpace; dir, itags, plev, lock, rev=false)

Set the prime level of matching legs to 0.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
noprime(q)                      # clear prime on all legs
noprime(q; dir='+')             # clear prime on incoming legs only
noprime(q; plev=1)              # clear all legs currently at plev=1
noprime(q; dir='+', rev=true)   # clear prime on all non-incoming legs
```
"""
function noprime(q::QSpace{T, QD}; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::QSpace, leg::Integer)

Set the prime level of a single specified leg to 0.
"""
function noprime(q::QSpace{T, QD}, leg::Integer) where {T, QD}
    return _modify_plev(q, (leg,), _ -> 0)
end

"""
    noprime(q::QSpace, legs::LegList)

Set the prime level of the specified legs to 0.
`legs` can be any vector, range, or tuple, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
"""
function noprime(q::QSpace{T, QD}, legs::LegList) where {T, QD}
    return _modify_plev(q, legs, _ -> 0)
end

"""
    noprime(q::QSpace, pred::Function)

Set the prime level of legs satisfying predicate to 0.
"""
function noprime(q::QSpace{T, QD}, pred::Function) where {T, QD}
    legs = findlegs(q, pred)
    return _modify_plev(q, legs, _ -> 0)
end

# ─── Tag modification ────────────────────────────────────────────────────────
#
# ITensor-style tag manipulation on QSpace legs.
# Tags are stored as sorted, comma-separated strings (e.g. "bond,site").
#
#   addtags(q, newtags; kw...)     – add tag(s) to matching legs
#   removetags(q, tags; kw...)     – remove tag(s) from matching legs
#   replacetags(q, old, new; kw…)  – swap tag(s) old → new on matching legs
#   settags(q, tags; kw...)        – replace entire tag string of matching legs
#
# Keyword selectors (all optional): dir, itags, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

function _modify_itags(q::QSpace{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = QIndex(modify_fn(idx.itags), idx.dir, idx.plev, idx.lock, idx.green)
    end
    return QSpace(q.symm, q.rows, Tuple(new_inds), q.spaces)
end

"""
    addtags(q::QSpace, newtags; dir, itags, plev, lock, rev=false)

Add one or more tags to matching legs. `newtags` may be a comma-separated
string of tags (e.g. `"bond,u1"`). The result is always sorted.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
addtags(q, "site")              # add "site" to all legs
addtags(q, "phys"; dir='+')     # add "phys" to incoming legs only
addtags(q, "u1"; itags="bond")  # add "u1" to legs that already have "bond"
addtags(q, "aux"; dir='+', rev=true)  # add "aux" to all non-incoming legs
```
"""
function addtags(q::QSpace{T, QD}, newtags::String; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_itags(q, legs, base -> _add_tags(base, newtags))
end

"""    addtags(q::QSpace, leg::Integer, newtags)

Add tags to a single specified leg.
"""
function addtags(q::QSpace{T, QD}, leg::Integer, newtags::String) where {T, QD}
    return _modify_itags(q, (leg,), base -> _add_tags(base, newtags))
end

"""    addtags(q::QSpace, legs::LegList, newtags)

Add tags to the specified legs. `legs` can be any vector, range, or tuple.
"""
function addtags(q::QSpace{T, QD}, legs::LegList, newtags::String) where {T, QD}
    return _modify_itags(q, legs, base -> _add_tags(base, newtags))
end

"""    addtags(q::QSpace, pred::Function, newtags)

Add tags to legs satisfying predicate.
"""
function addtags(q::QSpace{T, QD}, pred::Function, newtags::String) where {T, QD}
    return _modify_itags(q, findlegs(q, pred), base -> _add_tags(base, newtags))
end

"""
    removetags(q::QSpace, tags; dir, itags, plev, lock, rev=false)

Remove one or more tags from matching legs.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
removetags(q, "site")             # remove "site" from all legs
removetags(q, "phys"; dir='-')    # remove "phys" from outgoing legs
removetags(q, "aux"; dir='-', rev=true)  # remove "aux" from all non-outgoing legs
```
"""
function removetags(q::QSpace{T, QD}, tags::String; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_itags(q, legs, base -> _remove_tags(base, tags))
end

"""    removetags(q::QSpace, leg::Integer, tags)

Remove tags from a single specified leg.
"""
function removetags(q::QSpace{T, QD}, leg::Integer, tags::String) where {T, QD}
    return _modify_itags(q, (leg,), base -> _remove_tags(base, tags))
end

"""    removetags(q::QSpace, legs::LegList, tags)

Remove tags from the specified legs. `legs` can be any vector, range, or tuple.
"""
function removetags(q::QSpace{T, QD}, legs::LegList, tags::String) where {T, QD}
    return _modify_itags(q, legs, base -> _remove_tags(base, tags))
end

"""    removetags(q::QSpace, pred::Function, tags)

Remove tags from legs satisfying predicate.
"""
function removetags(q::QSpace{T, QD}, pred::Function, tags::String) where {T, QD}
    return _modify_itags(q, findlegs(q, pred), base -> _remove_tags(base, tags))
end

# TODO: Define the behavior of replacetags function more precisely,
# then implement and test
"""
    replacetags(q::QSpace, old, new; dir, itags, plev, lock, rev=false)

On matching legs, remove the tag(s) in `old` and add the tag(s) in `new`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
replacetags(q, "bond", "link")           # rename "bond" → "link" on all legs
replacetags(q, "a", "b"; dir='+')        # only on incoming legs
replacetags(q, "phys", "site"; itags="phys")  # only if leg already has "phys"
replacetags(q, "a", "b"; dir='+', rev=true)  # on all non-incoming legs
```
"""
function replacetags(q::QSpace{T, QD}, old::String, new_tags::String; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    return _modify_itags(q, legs, base -> _replace_tags(base, old, new_tags))
end

"""    replacetags(q::QSpace, leg::Integer, old, new)

Replace tags on a single specified leg.
"""
function replacetags(q::QSpace{T, QD}, leg::Integer, old::String, new_tags::String) where {T, QD}
    return _modify_itags(q, (leg,), base -> _replace_tags(base, old, new_tags))
end

"""    replacetags(q::QSpace, legs::LegList, old, new)

Replace tags on the specified legs. `legs` can be any vector, range, or tuple.
"""
function replacetags(q::QSpace{T, QD}, legs::LegList, old::String, new_tags::String) where {T, QD}
    return _modify_itags(q, legs, base -> _replace_tags(base, old, new_tags))
end

"""    replacetags(q::QSpace, pred::Function, old, new)

Replace tags on legs satisfying predicate.
"""
function replacetags(q::QSpace{T, QD}, pred::Function, old::String, new_tags::String) where {T, QD}
    return _modify_itags(q, findlegs(q, pred), base -> _replace_tags(base, old, new_tags))
end

"""
    settags(q::QSpace, tags; dir, itags, plev, lock, rev=false)

Replace the entire tag string of matching legs with `tags`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
settags(q, "bond")              # set all legs to tag "bond"
settags(q, ""; dir='+')         # clear tags on incoming legs
settags(q, "phys"; itags="site") # rename "site" → "phys" (full replacement)
settags(q, "aux"; dir='+', rev=true)  # set tag on all non-incoming legs
```
"""
function settags(q::QSpace{T, QD}, tags::String; dir=nothing, itags=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itags=itags, plev=plev, lock=lock, rev=rev)
    norm = _normalize_tags(tags)
    return _modify_itags(q, legs, _ -> norm)
end

"""    settags(q::QSpace, leg::Integer, tags)

Set the entire tag string of a single specified leg.
"""
function settags(q::QSpace{T, QD}, leg::Integer, tags::String) where {T, QD}
    norm = _normalize_tags(tags)
    return _modify_itags(q, (leg,), _ -> norm)
end

"""    settags(q::QSpace, legs::LegList, tags)

Set the entire tag string of the specified legs. `legs` can be any vector, range, or tuple.
"""
function settags(q::QSpace{T, QD}, legs::LegList, tags::String) where {T, QD}
    norm = _normalize_tags(tags)
    return _modify_itags(q, legs, _ -> norm)
end

"""    settags(q::QSpace, pred::Function, tags)

Set the entire tag string of legs satisfying predicate.
"""
function settags(q::QSpace{T, QD}, pred::Function, tags::String) where {T, QD}
    norm = _normalize_tags(tags)
    return _modify_itags(q, findlegs(q, pred), _ -> norm)
end

# ─── Pretty-printing for QSpace ──────────────────────────────────────────────
#
# Format:
#
#   QSpace{T}  [Symm1, Symm2, ...]
#     leg 1:  dir  'tag'  (plev=k)
#     leg 2:  dir  'tag'
#     ...
#     1.  <row inline>
#     2.  <row inline>
#     ...
#
# When the number of rows exceeds QSPACE_DISPLAY_HEAD + QSPACE_DISPLAY_TAIL,
# only the first QSPACE_DISPLAY_HEAD and last QSPACE_DISPLAY_TAIL rows are shown.
# Set these globals to control the truncation behaviour.
# ─────────────────────────────────────────────────────────────────────────────

const QSPACE_DISPLAY_HEAD = Ref(5)   # number of first rows to show
const QSPACE_DISPLAY_TAIL = Ref(5)   # number of last rows to show

Base.show(io::IO, qs::QSpace) = show(io, MIME"text/plain"(), qs)

# Special pretty-printing for 0-dimensional QSpace (scalar result of full contraction).
function Base.show(io::IO, ::MIME"text/plain", qs::QSpace{T, 0, N, N}) where {T, N}
    symm_names = join((totxt(s) for s in qs.symm), ", ")
    print(io, "0D QSpace{$T}, $N symmetries [$symm_names]: ", _fmt_scalar_str(qs[]))
end

function Base.show(io::IO, ::MIME"text/plain", qs::QSpace{T, QD, N, RD}) where {T, QD, N, RD}
    # --- Header: symmetries and leg dirs/tags on one line ---
    # Format:  QSpace{...}  [Sym1, Sym2]  ["tag1+", "tag2-", ...]
    symm_names = join((totxt(s) for s in qs.symm), ", ")
    print(io, "$(QD)D QSpace, $N symmetries [$symm_names]")
    leg_strs = map(qs.inds) do idx
        tag  = isempty(idx.itags) ? "" : idx.itags
        plev = idx.plev != 0 ? "[$(idx.plev)]" : ""
        raw  = "\"$(tag)$(idx.dir)$plev\""
        idx.green ? "\e[32m$(raw)\e[0m" : raw
    end
    println(io, "  [", join(leg_strs, ", "), "]")

    # --- Rows: one per line with global label width for cross-row alignment ---
    nrows = length(qs.rows)
    if nrows == 0
        print(io, "  (empty)")
        return
    end

    # Determine which row indices to display.
    head = QSPACE_DISPLAY_HEAD[]
    tail = QSPACE_DISPLAY_TAIL[]
    truncate = nrows > head + tail
    display_indices = if truncate
        vcat(1:head, nrows-tail+1:nrows)
    else
        1:nrows
    end

    # Compute per-symmetry widths globally across displayed rows only.
    widths = map(1:N) do n
        maximum(display_indices) do i
            _label_widths(qs.rows[i].cgrs, qs.symm)[n]
        end
    end
    # Pre-compute scalar width for alignment (only rows with scalar RMT).
    scalar_width = 0
    for i in display_indices
        r = qs.rows[i]
        length(r.RMT.data) == 1 || continue
        scalar_width = max(scalar_width, length(_fmt_scalar_str(only(r.RMT.data))))
    end

    for (k, i) in enumerate(display_indices)
        # Print ellipsis between head and tail blocks.
        if truncate && k == head + 1
            println(io)
            println(io, "  ⋮  ($(nrows - head - tail) rows omitted)")
        end
        r = qs.rows[i]
        om_dim = prod(size(r.RMT.data)[QD+1:end]; init=1)
        phys_str = join(size(r.RMT.data)[1:QD], "x")
        om_str   = om_dim > 1 ? " @$om_dim" : ""
        print(io, "  $i.\t", phys_str, om_str, "\t")
        _print_cgt_dims(io, r.cgrs, qs.symm)
        _print_qlabels(io, r.cgrs, widths)
        length(r.RMT.data) == 1 && print(io, "\t", lpad(_fmt_scalar_str(only(r.RMT.data)), scalar_width))
        QD == 2 && print(io, "\t√", _cgt_size_2d(r.cgrs, qs.symm))
        k < length(display_indices) && println(io)
    end
end

# Normalize QSpace w-matrices in-place.
# - For 2D: set each wmat[1] = sqrt(dim) so the CGT contributes identity scaling.
# - For 0D: set all wmat elements to 1.0, absorbing factors into RMT.
# - For other dimensions: no-op.
function normalize_qspace!(q::QSpace{T, QD, N}) where {T, QD, N}
    if QD == 2
        for r in q.rows
            for i in 1:N
                S, cgr = q.symm[i], r.cgrs[i]
                q1, q2 = cgr.qlabels
                @assert q2 == q1 || q2 == get_dualq(S, q1)
                dim = dimension(S, q1); @assert dim == dimension(S, q2)
                @assert size(cgr.wmat) == (1, 1) # No outer multiplicity
                w_factor = sqrt(dim) / cgr.wmat[1] 
                cgr.wmat[:] *= w_factor
                r.RMT[:] /= w_factor
            end
        end
    elseif QD == 0
        for r in q.rows
            for i in 1:N
                cgr = r.cgrs[i]
                @assert size(cgr.wmat) == (1, 1) "0D QSpace must have 1x1 w-matrices"
                w_val = cgr.wmat[1]
                if w_val != 1.0
                    cgr.wmat[:] .= 1.0
                    r.RMT[:] .*= w_val
                end
            end
        end
    end
    # For other dimensions (QD != 0 and QD != 2), do nothing.
end

# Sort rows of a QSpace in-place in dictionary order by physical leg qlabels.
# For each row the sort key is built leg by leg (1 → QD): at leg l, the key
# is the tuple of qlabels across all symmetries, i.e.
#   (cgrs[1].qlabels[cgp₁[l]], cgrs[2].qlabels[cgp₂[l]], ...)
# Comparison is then lexicographic over (leg 1 key, leg 2 key, ..., leg QD key).
function sort_rows!(q::QSpace{T, QD, N}) where {T, QD, N}
    sort!(q.rows; by = r -> Tuple(
            Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[l]] for n in 1:N)
        for l in QD:-1:1)
    )
end

# Scalar multiplication and division: only the RMT arrays are scaled.
# CGRs (w-matrices, qlabels) are left untouched.
function Base.:*(qs::QSpace, fac::Number)
    result = deepcopy(qs)
    for r in result.rows
        r.RMT.data .*= fac
    end
    return result
end
Base.:*(fac::Number, qs::QSpace) = qs * fac
Base.:/(qs::QSpace, fac::Number) = qs * (1 / fac)
Base.:-(qs::QSpace) = qs * -1

# Return a deep copy of a QSpace (CGRs, RMTs, indices, spaces all copied).
Base.copy(q::QSpace) = deepcopy(q)

function _identity_on_qspace(q::QSpace{T, QD, N, RD}) where {T, QD, N, RD}
    @assert QD == 2 "Scalar add/subtract is only defined for rank-2 QSpace objects"

    in_legs  = findlegs(q; dir='+')
    out_legs = findlegs(q; dir='-')
    @assert length(in_legs) == 1 && length(out_legs) == 1 "Scalar add/subtract requires exactly one incoming and one outgoing leg"

    in_leg  = only(in_legs)
    out_leg = only(out_legs)
    @assert q.spaces[in_leg] == q.spaces[out_leg] "Scalar add/subtract requires matching incoming and outgoing spaces"

    id_q = getIdentity((q, out_leg); itags=q.inds[out_leg].itags)
    return QSpace(id_q, (q.inds[in_leg], q.inds[out_leg]))
end

# ─── QSpace norm ─────────────────────────────────────────────────────────────
#
# Exploits the Wigner-Eckart decomposition to compute the Frobenius norm
# directly from the reduced matrix elements (RMTs) without building the full
# dense tensor.
#
# For QD ≠ 2 (including QD = 0 and QD ≥ 3):
#   Rows from different q-label sectors are orthogonal by symmetry.  Within
#   each row, the CGT canonical basis elements are orthonormal and the
#   w-matrix is left-orthogonal (Uᵀ·U = I), so the M columns of cgt_block
#   are also orthonormal.  Therefore all CGT cross-terms vanish and:
#       ‖A‖² = Σ_r ‖RMT_r‖²
#
# For QD = 2 (2-leg operators / matrices):
#   normalize_qspace! sets wmat[1,1] = √dim for each row, turning the 2D CGT
#   block into a (dim × dim) identity matrix  (‖Id_dim‖² = dim).
#   Consequently:
#       ‖A‖² = Σ_r dim_r · ‖RMT_r‖²
#   where dim_r = _cgt_size_2d(r.cgrs, q.symm) = ∏_{non-abelian n} d_leg1^(n).
#
# ─────────────────────────────────────────────────────────────────────────────
function LinearAlgebra.norm(q::QSpace{T, QD, N}) where {T, QD, N}
    s = zero(Float64)
    if QD == 2
        for r in q.rows
            d = _cgt_size_2d(r.cgrs, q.symm)
            s += d * sum(abs2, r.RMT.data)
        end
    else
        for r in q.rows
            s += sum(abs2, r.RMT.data)
        end
    end
    return sqrt(s)
end

# ─── QSpace addition ─────────────────────────────────────────────────────────
#
# Each row is a QSpace record uniquely identified by its physical q-labels
# (cgrs[n].qlabels with cgp applied) across all symmetries.
#
# Per the Wigner-Eckart decomposition (paper Eq. 22):
#   X = ⊕_q [ ‖X‖_{qμ} ⊗ (w_{μμ'} C_{qμ'}) ]
#
# For a sector present in both operands, both rows reference the same underlying
# sorted CGT.  The sum pools the two (w, RMT) contributions and finds a minimal
# new w-matrix via SVD using _compress_sector / merge_new_row (same routine as
# used for contractions).  A sector present in only one operand is copied as-is.
#
# If the two QSpaces share the same indices but in different order, the second
# operand is permuted to match the first before addition.
# ─────────────────────────────────────────────────────────────────────────────

# Find the unique permutation perm such that qs2.inds[perm[i]] == inds1[i]
# and qs2.spaces[perm[i]] == spaces1[i] for all i.  Raises an error if no
# such bijection exists or if it is not unique.
function _find_leg_permutation(inds1::NTuple{QD, QIndex}, spaces1,
                               inds2::NTuple{QD, QIndex}, spaces2) where {QD}
    candidates = Vector{Vector{Int}}(undef, QD)
    for i in 1:QD
        cs = Int[]
        for j in 1:QD
            if inds2[j] == inds1[i] && spaces2[j] == spaces1[i]
                push!(cs, j)
            end
        end
        isempty(cs) && error(
            "No leg in second QSpace matches leg $i of first QSpace " *
            "(itags=\"$(inds1[i].itags)\", dir='$(inds1[i].dir)')")
        candidates[i] = cs
    end
    results = Vector{NTuple{QD, Int}}()
    _enum_leg_perms!(results, candidates, Int[], Set{Int}(), QD)
    isempty(results) && error(
        "No valid bijective permutation found to match QSpace leg indices")
    length(results) > 1 && error(
        "Ambiguous permutation: $(length(results)) ways to match QSpace leg indices")
    return results[1]
end

function _enum_leg_perms!(results, candidates, current, used, QD)
    i = length(current) + 1
    if i > QD
        push!(results, Tuple(current))
        return
    end
    for j in candidates[i]
        j ∈ used && continue
        push!(current, j)
        push!(used, j)
        _enum_leg_perms!(results, candidates, current, used, QD)
        pop!(current)
        delete!(used, j)
    end
end

function Base.:+(qs1::QSpace{T1, QD, N, RD},
                 qs2::QSpace{T2, QD, N, RD}) where {T1, T2, QD, N, RD}
    @assert qs1.symm == qs2.symm "QSpace objects must share the same symmetry tuple"

    if qs1.inds != qs2.inds || qs1.spaces != qs2.spaces
        perm = _find_leg_permutation(qs1.inds, qs1.spaces, qs2.inds, qs2.spaces)
        qs2  = permuteQS(qs2, perm)
    end

    T    = promote_type(T1, T2)
    symm = qs1.symm

    # Physical q-label key for a row: the cgp-permuted qlabels per symmetry.
    # Two rows belong to the same sector iff these match for every symmetry.
    row_key(r) = ntuple(N) do n
        cgr = r.cgrs[n]
        ntuple(l -> cgr.qlabels[cgr.cgp[l]], QD)   # q-label on physical leg l
    end

    dict1 = Dict(row_key(r) => r for r in qs1.rows)
    dict2 = Dict(row_key(r) => r for r in qs2.rows)

    new_rows = Vector{row{T, QD, N, RD}}()

    for key in union(keys(dict1), keys(dict2))
        in1 = haskey(dict1, key)
        in2 = haskey(dict2, key)

        if in1 && !in2
            # Sector exists only in qs1 — copy with promoted element type.
            r = dict1[key]
            push!(new_rows, row(r.cgrs, QTensor(T.(r.RMT.data))))

        elseif in2 && !in1
            # Sector exists only in qs2 — copy with promoted element type.
            r = dict2[key]
            push!(new_rows, row(r.cgrs, QTensor(T.(r.RMT.data))))

        else
            # Sector exists in both — merge the two (w, RMT) contributions.
            # Both rows reference the same underlying sorted CGT, so the
            # stored qlabels and cgp (which encode the permutation to that
            # sorted CGT) must agree.
            r1, r2 = dict1[key], dict2[key]
            for n in 1:N
                @assert r1.cgrs[n].qlabels == r2.cgrs[n].qlabels "qlabels mismatch at symmetry $n for sector $key"
                @assert r1.cgrs[n].cgp     == r2.cgrs[n].cgp     "cgp mismatch at symmetry $n for sector $key"
            end

            # Pool w-matrices and RMTs as two contributions, then compress.
            new_wmats = ntuple(n -> [r1.cgrs[n].wmat, r2.cgrs[n].wmat], N)
            new_RMTs  = QTensor{T, RD}[QTensor(T.(r1.RMT.data)),
                                        QTensor(T.(r2.RMT.data))]
            new_qlabs = ntuple(n -> (r1.cgrs[n].qlabels,
                                     r1.cgrs[n].cgp,
                                     r1.cgrs[n].legdir), N)

            push!(new_rows, merge_new_row(new_wmats, new_RMTs, new_qlabs, symm, QD))
        end
    end

    return QSpace(symm, new_rows, qs1.inds, qs1.spaces)
end

Base.:-(qs1::QSpace, qs2::QSpace) = qs1 + (-1 * qs2)
Base.:+(q::QSpace, x::Number) = q + x * _identity_on_qspace(q)
Base.:+(x::Number, q::QSpace) = q + x
Base.:-(q::QSpace, x::Number) = q + (-x) * _identity_on_qspace(q)
Base.:-(x::Number, q::QSpace) = x * _identity_on_qspace(q) + (-q)



# ─── conj / adjoint ──────────────────────────────────────────────────────────
#
# conj(q): conjugate a QSpace object.
#   1. Every leg direction is reversed ('+' ↔ '-').
#   2. Every RMT entry is complex-conjugated.
#   3. For each CGR, the arrow directions of the underlying CGT are inverted:
#        legdir (m, k)  →  (k, m)
#        stored qlabels: [in_1..in_m, out_1..out_k] → [out_1..out_k, in_1..in_m]
#        cgp updated accordingly (stored positions renumbered after the swap).
#      Additionally, when the old incoming qlabels equal the old outgoing qlabels
#      (self-conjugate CGT), the canonical OM ordering of the new CGT differs from
#      the old one by a transposition within each central-space block — this is
#      applied as a row permutation on the first dimension of the w-matrix.
#
# adjoint(q): for QSpace tensors, defined as conj(q).
# ─────────────────────────────────────────────────────────────────────────────
function Base.conj(q::QSpace{T, QD, N, RD}) where {T, QD, N, RD}
    new_inds = ntuple(l -> change_dir(q.inds[l]), QD)

    new_rows = map(q.rows) do r
        # 1. Complex-conjugate the RMT.
        new_RMT = QTensor(conj.(r.RMT.data))

        # 2. Rebuild CGRs with flipped arrow directions.
        new_cgrs = ntuple(N) do n
            cgr   = r.cgrs[n]
            m, k  = cgr.legdir
            QD_n  = m + k

            # New qlabels: old outgoing block first, then old incoming block.
            new_qlabels = (cgr.qlabels[m+1:m+k]..., cgr.qlabels[1:m]...)

            # New cgp: stored-position map after the qlabel block swap.
            #   Old position s ∈ 1:m   (incoming) → new position s + k  (outgoing)
            #   Old position s ∈ m+1:m+k (outgoing) → new position s - m (incoming)
            new_cgp = ntuple(l -> begin
                s = cgr.cgp[l]
                s <= m ? s + k : s - m
            end, QD_n)

            new_legdir = (k, m)

            # 3. Permute the OM (first) axis of the w-matrix when the old
            #    incoming and outgoing qlabels are identical tuples.  In that
            #    case the new CGT references the same canonical CGTom but with
            #    the flat OM ordering transposed within every central-space block:
            #      old flat = start + (upidx-1) + (dnidx-1)*om_up   [upidx fast]
            #      new flat = start + (dnidx-1) + (upidx-1)*om_dn   [dnidx fast]
            S = cgr.symm
            ins, outs = cgr.qlabels[1:m], cgr.qlabels[m+1:m+k]
            ins_, _ = remove_zeros(S, ins)
            outs_, _ = remove_zeros(S, outs)
            new_wmat =
                if !isabelian(S) && ins_ == outs_
                    is1j = detect_1j(S, ins_, outs_)
                    cgt_oms  = get_CGTom(S, ins_, outs_, is1j)
                    perm_vec = get_conj_perm(cgt_oms)
                    QTensor(cgr.wmat.data[perm_vec, :])
                else
                    deepcopy(cgr.wmat)
                end

            CGR(cgr.symm, Tuple(new_qlabels), new_wmat, new_cgp, new_legdir)
        end

        row(Tuple(new_cgrs), new_RMT)
    end

    # spaces remain the same: physical qlabels at each leg don't change in conj,
    # only the CGR internal structure (incoming/outgoing) changes
    return QSpace(q.symm, collect(new_rows), new_inds, q.spaces)
end

Base.adjoint(q::QSpace) = conj(q)

getsub(q::QSpace, i::Int) = getsub(q, [i])
getsub(q::QSpace, inds::Vector{Int}) = QSpace(q.symm, q.rows[inds], q.inds, q.spaces)

"""
    empty_qspace(symm::NTuple{N, Any}, inds::NTuple{QD, QIndex}; T::Type=Float64) where {N, QD}

Create an empty rank-`QD` (zero-row) QSpace over the given symmetries.

`symm` is an `N`-tuple of symmetry types (e.g. `(SU{2}, U1)`); `inds` is a
`QD`-tuple of `QIndex` objects describing the leg directions, tags, and prime
levels.  All `QIndex` entries with non-empty tags must be pairwise distinct.

The element type of future row data defaults to `Float64`; pass `T=ComplexF64`
(or another concrete `<:Number` type) to use a different element type.
"""
function empty_qspace(symm::NTuple{N, Any}, inds::NTuple{QD, QIndex};
                      T::Type=Float64) where {N, QD}
    RD = QD + N
    rows   = Vector{row{T, QD, N, RD}}()
    spaces = ntuple(_ -> Vector{Tuple{Int, NTuple{N, Tuple{Vararg{Int}}}}}(), QD)
    return QSpace(symm, rows, inds, spaces)
end

function empty_qspace(q::QSpace; T::Type=Float64)
    return empty_qspace(q.symm, q.inds; T=T)
end

"""
    getvac(q::QSpace, itags::NTuple{2, String}=("", "")) -> QSpace

Build the rank-2 vacuum QSpace associated with `q`.

The result keeps the same symmetry tuple as `q`, has one incoming leg and one
outgoing leg, and contains exactly one trivial sector with RMT dimension 1 on
each leg. If `itags` is provided, it is used as the tags of the two legs.
"""
function getvac(q::QSpace{T, QD, N, RD},
                itags::NTuple{2, String}=("", "")) where {T, QD, N, RD}
    trivial_qlabels = ntuple(n -> Tuple(0 for _ in 1:nzops(q.symm[n])), N)
    space_entry = (1, trivial_qlabels)

    cgrs = ntuple(n -> begin
        trivial_q = trivial_qlabels[n]
        CGR(q.symm[n], (trivial_q, trivial_q), QTensor([1.0;;]), (1, 2), (1, 1))
    end, N)

    rmt_data = fill(one(T), ntuple(_ -> 1, N + 2))
    rows = row{T, 2, N, N + 2}[row(cgrs, QTensor(rmt_data))]
    inds = (QIndex(itags[1], '+'), QIndex(itags[2], '-'))
    space_template = Vector{Tuple{Int, NTuple{N, Tuple{Vararg{Int}}}}}([space_entry])
    spaces = (copy(space_template), copy(space_template))

    return QSpace(q.symm, rows, inds, spaces)
end

include("getLocalSpace.jl")
include("getIdentity.jl")
include("contract.jl")
include("get1jpair.jl")
include("svd.jl")
include("eig.jl")
include("permute.jl")
