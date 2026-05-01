using Printf
using LinearAlgebra
include("LurTensor.jl")
include("utils.jl")
include("localspaces/localspaces.jl")

# A compile-time tag for a direct product of symmetry groups. QSpaces records
# symmetry identities in type parameters; use `symm(q)` to retrieve them
# without going through dynamic property access.
abstract type ProductSymm{Syms<:Tuple{Vararg{Symmetry}}} <: Symmetry end

ProductSymm(syms::Type{<:Symmetry}...) = ProductSymm{Tuple{syms...}}

product_symms(::Type{<:ProductSymm{Syms}}) where {Syms} = Tuple(Syms.parameters)
nsymms(::Type{<:ProductSymm{Syms}}) where {Syms} = length(Syms.parameters)

function productsymm(symm::Tuple)
    all(s -> s isa Type{<:Symmetry}, symm) || throw(ArgumentError(
        "all entries of a product symmetry must be symmetry types, got $symm"))
    return ProductSymm(symm...)
end

productsymm(::Type{PS}) where {PS<:ProductSymm} = PS
product_symms(symm::Tuple) = product_symms(productsymm(symm))
nsymms(symm::Tuple) = nsymms(productsymm(symm))

# ─── Tag helpers ──────────────────────────────────────────────────────────────
#
# Tags are stored internally as an Itag wrapper around a canonical
# comma-separated String sorted alphabetically, e.g. "bond,site,u1"
# (ITensor convention). Empty string means no tags. Individual tags contain no
# commas or whitespace.
#
# ─────────────────────────────────────────────────────────────────────────────

_itag_text(tags::AbstractString) = String(tags)

# Split tag text into sorted vector of non-empty individual tags.
_parse_itag(tags::AbstractString) = sort!(filter!(!isempty, strip.(split(_itag_text(tags), ','))))

# Canonical form: sorted, comma-joined, no spaces.
_normalize_itag_text(tags::AbstractString) = join(_parse_itag(tags), ',')

struct Itag <: AbstractString
    value::String

    Itag(tags::AbstractString) = new(_normalize_itag_text(tags))
end

Itag(tags::Itag) = tags
_itag_text(tags::Itag) = tags.value
_normalize_itag(tags::AbstractString) = Itag(tags)

Base.String(tags::Itag) = tags.value
Base.convert(::Type{String}, tags::Itag) = tags.value
Base.ncodeunits(tags::Itag) = ncodeunits(tags.value)
Base.codeunit(::Type{Itag}) = codeunit(String)
Base.codeunit(tags::Itag, i::Int) = codeunit(tags.value, i)
Base.isvalid(tags::Itag, i::Int) = isvalid(tags.value, i)
Base.iterate(tags::Itag) = iterate(tags.value)
Base.iterate(tags::Itag, i::Int) = iterate(tags.value, i)
Base.length(tags::Itag) = length(tags.value)
Base.lastindex(tags::Itag) = lastindex(tags.value)
Base.thisind(tags::Itag, i::Int) = thisind(tags.value, i)
Base.nextind(tags::Itag, i::Int) = nextind(tags.value, i)
Base.nextind(tags::Itag, i::Int, n::Int) = nextind(tags.value, i, n)
Base.prevind(tags::Itag, i::Int) = prevind(tags.value, i)
Base.prevind(tags::Itag, i::Int, n::Int) = prevind(tags.value, i, n)
Base.show(io::IO, tags::Itag) = print(io, tags.value)
Base.isempty(tags::Itag) = isempty(tags.value)
Base.hash(tags::Itag, h::UInt) = hash(tags.value, h)
Base.:(==)(a::Itag, b::Itag) = a.value == b.value
Base.:(==)(a::Itag, b::AbstractString) = a.value == _normalize_itag_text(b)
Base.:(==)(a::AbstractString, b::Itag) = _normalize_itag_text(a) == b.value
Base.isequal(a::Itag, b::Itag) = (a == b)
Base.isequal(a::Itag, b::AbstractString) = (a == b)
Base.isequal(a::AbstractString, b::Itag) = (a == b)

# True iff all comma-separated tags in `query` are present in `base`.
function _has_itag(base::AbstractString, query::AbstractString)
    bset = Set(_parse_itag(base))
    return all(t -> t ∈ bset, _parse_itag(query))
end

_matches_itag_selector(base::AbstractString, query::AbstractString) = _has_itag(base, query)
_matches_itag_selector(base::AbstractString, queries::Tuple{Vararg{AbstractString}}) =
    any(query -> _has_itag(base, query), queries)
_matches_itag_selector(base::AbstractString, queries::AbstractVector{<:AbstractString}) =
    any(query -> _has_itag(base, query), queries)

# Add tags from `newtags` to `base`; result is sorted and deduplicated.
_add_itag(base::AbstractString, newtags::AbstractString) =
    Itag(join(sort!(unique!(vcat(_parse_itag(base), _parse_itag(newtags)))), ','))

# Remove every tag listed in `rmtags` from `base`.
function _remove_itag(base::AbstractString, rmtags::AbstractString)
    rm = Set(_parse_itag(rmtags))
    return Itag(join(filter(t -> t ∉ rm, _parse_itag(base)), ','))
end

# Remove tags from `base` using one or more ITensor-style tag queries.
# Each query is applied independently, and its tags are removed only if all
# tags in that query are present in `base`.
function _remove_itag(base::AbstractString,
                      rmtags::Union{Tuple{Vararg{AbstractString}},
                                    AbstractVector{<:AbstractString}})
    base_tags = _parse_itag(base)
    base_set = Set(base_tags)
    rm = Set{String}()
    for query in rmtags
        parsed = _parse_itag(query)
        all(tag -> tag ∈ base_set, parsed) || continue
        union!(rm, parsed)
    end
    return Itag(join(filter(tag -> tag ∉ rm, base_tags), ','))
end

struct QIndex
    # Tags associated with the leg, similar to ITensor
    itags::Itag
    # Direction of the leg, '+' for incoming, '-' for outgoing
    dir::Char
    # Prime level similar to ITensor
    plev::Int
    # Lock level. Cannot cntracted if lock > 0. 
    # Decrease lock level by 1 after contraction.
    lock::Int
    # If true, this leg represents the dual space.
    dual::Bool

    QIndex(itags::AbstractString, dir::Char, plev::Int=0, lock::Int=0, dual::Bool=false) = new(_normalize_itag(itags), dir, plev, lock, dual)
end

QIndex(dir::Char, plev::Int=0, lock::Int=0) = QIndex("", dir, plev, lock)

# Two QIndex objects are equal if they share the same itags, dir, plev, and dual
# (lock is intentionally ignored — it is a transient contraction counter).
Base.:(==)(a::QIndex, b::QIndex) = a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.dual == b.dual
Base.isequal(a::QIndex, b::QIndex) = (a == b)
Base.hash(a::QIndex, h::UInt) = hash((a.itags, a.dir, a.plev, a.dual), h)

to_incoming(idx::QIndex) = QIndex(idx.itags, '+', idx.plev, idx.lock, idx.dual)
to_outgoing(idx::QIndex) = QIndex(idx.itags, '-', idx.plev, idx.lock, idx.dual)
change_dir(idx::QIndex)  = QIndex(idx.itags, idx.dir == '+' ? '-' : '+', idx.plev, idx.lock, idx.dual)
dual(idx::QIndex) = QIndex(idx.itags, idx.dir, idx.plev, idx.lock, true)
change_dual(idx::QIndex) = QIndex(idx.itags, idx.dir, idx.plev, idx.lock, !idx.dual)
green(idx::QIndex) = dual(idx)
change_green(idx::QIndex) = change_dual(idx)

struct CGR{QD, NZ, S<:Symmetry}
    qlabels::NTuple{QD, NTuple{NZ, Int}}
    wmat::LurTensor{Float64, 2}
    cgp::NTuple{QD, Int}
    # (# incoming legs, # outgoing legs); sum == QD
    legdir::Tuple{Int, Int}  
end

function CGR(symm::Type{S},
             qlabels::NTuple{QD, NTuple{NZ, Int}},
             wmat::LurTensor{Float64, 2},
             cgp::NTuple{QD, Int},
             legdir::Tuple{Int, Int}) where {S<:Symmetry, QD, NZ}
    return CGR{QD, NZ, S}(qlabels, wmat, cgp, legdir)
end

# Constructor for QD=0 case: infer NZ from the symmetry type
function CGR(symm::Type{S}, qlabels::Tuple{}, wmat::LurTensor{Float64, 2}, 
             cgp::Tuple{}, legdir::Tuple{Int, Int}) where {S<:Symmetry}
    NZ = nzops(S)
    CGR{0, NZ, S}(qlabels, wmat, cgp, legdir)
end

cgrsymm(::Type{<:CGR{QD, NZ, S}}) where {QD, NZ, S} = S
cgrsymm(::CGR{QD, NZ, S}) where {QD, NZ, S} = S
@inline symm(cgr::CGR) = cgrsymm(cgr)

@inline function Base.getproperty(cgr::CGR, name::Symbol)
    name === :symm && return cgrsymm(cgr)
    return getfield(cgr, name)
end

Base.propertynames(::CGR, private::Bool=false) = (:symm, :qlabels, :wmat, :cgp, :legdir)

_cgr_qd(::CGR{QD}) where {QD} = QD

cgrstype(::Type{PS}, ::Val{QD}) where {PS<:ProductSymm, QD} =
    Tuple{(CGR{QD, nzops(S), S} for S in product_symms(PS))...}

struct row{T, QD, N, RD, CGRS<:NTuple{N, CGR{QD}}}
    cgrs::CGRS
    RMT::LurTensor{T, RD}
end

function row(cgrs::CGRS, RMT::LurTensor{T, RD}) where {CGRS<:Tuple, T, RD}
    N = length(cgrs)
    N > 0 || throw(ArgumentError("row requires at least one CGR"))
    QD = _cgr_qd(first(cgrs))
    all(cgr -> _cgr_qd(cgr) == QD, cgrs) ||
        throw(ArgumentError("all CGRs in a row must have the same tensor rank"))
    return row{T, QD, N, RD, CGRS}(cgrs, RMT)
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
        s    = isnothing(symm) ? cgrsymm(cgr) : symm[n]
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
        s = isnothing(symm) ? cgrsymm(cgrs[n]) : symm[n]
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
        has_neg = cgrsymm(cgr) <: U1
        mxabs = maximum(abs, (v for ql in cgr.qlabels for v in ql), init=0)
        ndigits(max(mxabs, 1)) + (has_neg ? 1 : 0)
    end
    # Fixed-width prefix: "  SYMNAME  wmat=NxM  "
    prefix_w = maximum(1:N) do n
        cgr = r.cgrs[n]
        S = cgrsymm(cgr)
        sym_str = isnothing(S) ? "cgr[$n]" : totxt(S)
        length("  $(rpad(sym_str, 4))  wmat=$(join(size(cgr.wmat.data),"x"))  ")
    end

    # cgr lines: aligned prefix, raw qlabels, cgp, scalar wmat
    for n in 1:N
        cgr = r.cgrs[n]
        S = cgrsymm(cgr)
        sym_str = isnothing(S) ? "cgr[$n]" : totxt(S)
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
        wmats = Vector{LurTensor{Float64, 2}}()
        for i in 1:N
            wmat, block, _ = svd_leg(block, QD + i)
            push!(wmats, LurTensor(wmat))
        end
        RMT = LurTensor(block)
        cgrs = ntuple(N) do i
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
            CGR(symm[i], qforsymm, wmats[i], cgp, legdir)
        end
        push!(rows, row(cgrs, RMT))
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
    cgp_inv = invperm(cgr.cgp)

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
# may be equal (as determined by ==, which compares itags, dir, plev, dual).
function _check_unique_inds(inds::NTuple{QD, QIndex}) where QD
    tagged = [idx for idx in inds if !isempty(idx.itags)]
    for i in 1:length(tagged), j in i+1:length(tagged)
        @assert tagged[i] != tagged[j] begin
            "Duplicate QIndex with non-empty itag in QSpace.inds: $(tagged[i])"
        end
    end
end

# Compute the spaces tuple from rows: for each leg, a vector of (qlabels, dim) pairs.
# This is the same information that leginfo extracts, but computed for all legs at once.
function _compute_spaces(rows::Vector{row{T, QD, N, RD}}) where {T, QD, N, RD}
    # For each leg, track seen qlabels to avoid duplicates
    spsets = ntuple(_ -> Set{NTuple{N, Tuple{Vararg{Int}}}}(), QD)
    splists = ntuple(_ -> Vector{Tuple{NTuple{N, Tuple{Vararg{Int}}}, Int}}(), QD)
    
    for r in rows
        for leg in 1:QD
            qlabel_leg = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[leg]] for n in 1:N)
            if qlabel_leg ∉ spsets[leg]
                dim = size(r.RMT.data, leg)
                push!(splists[leg], (qlabel_leg, dim))
                push!(spsets[leg], qlabel_leg)
            end
        end
    end
    
    return splists
end

# T: type of element in the RMT array, can be Float64, ComplexF64, etc.
# QD: The rank of tensor (# of legs), N: The number of symmetries
# RD: The rank of RMT array, which is equal to QD + N
# QT: The qlabel type for one leg sector, inferred from the symmetries
struct QSpace{T, QD, N, RD, QT, PS<:ProductSymm, CGRS<:NTuple{N, CGR{QD}}}
    # Data rows for QSpace object
    rows::Vector{row{T, QD, N, RD, CGRS}}
    inds::NTuple{QD, QIndex}
    # Space list for each leg: vector of (qlabels, RMT_dim) pairs
    # Similar to leginfo.splist but precomputed for all legs
    spaces::NTuple{QD, Vector{Tuple{QT, Int}}}

    # Constructor with explicit spaces (for efficiency when spaces are already known)
    function QSpace(symm::NTuple{N, Any}, 
        rows::AbstractVector{<:row{T, QD, N, RD}},
        inds::NTuple{QD, QIndex},
        spaces::Tuple{Vararg{<:AbstractVector, QD}}) where {T, QD, N, RD}

        QT = qlabeltype(symm)
        PS = productsymm(symm)
        CGRS = cgrstype(PS, Val(QD))
        typed_rows = Vector{row{T, QD, N, RD, CGRS}}(rows)
        typed_spaces = ntuple(l -> convert(Vector{Tuple{QT, Int}}, spaces[l]), QD)
        q = new{T, QD, N, RD, QT, PS, CGRS}(typed_rows, inds, typed_spaces)
        normalize_qspace!(q)
        _orient_wmats!(q)
        _drop_small_rows!(q)
        #sort_rows!(q)
        for r in q.rows, cgr in r.cgrs
            _check_cgr_qlabel_order(cgr)
        end
        _check_unique_inds(q.inds)
        _check_empty_tag_lock(q.inds)
        return q
    end
end

function QSpace(::Type{PS},
    rows::AbstractVector{<:row{T, QD, N, RD}},
    inds::NTuple{QD, QIndex},
    spaces::Tuple{Vararg{<:AbstractVector, QD}}) where {T, QD, N, RD, PS<:ProductSymm}
    QT = qlabeltype(PS)
    CGRS = cgrstype(PS, Val(QD))
    return QSpace(product_symms(PS), rows, inds, spaces)::QSpace{T, QD, N, RD, QT, PS, CGRS}
end

productsymm(::QSpace{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} = PS
product_symms(q::QSpace) = product_symms(productsymm(q))
@inline symm(::QSpace{T, QD, N, RD, QT, PS}) where {T, QD, N, RD, QT, PS} =
    product_symms(PS)
nsymms(q::QSpace) = nsymms(productsymm(q))

# Drop rows whose norm² contribution is below cutoff² × total norm² (relative threshold).
# For QD == 2 the effective norm² per row is dim_r * ‖RMT_r‖² (see normalize_qspace!);
# for all other ranks it is simply ‖RMT_r‖².
const QSPACE_ROW_CUTOFF = 1e-14

Base.ndims(q::QSpace{T, QD}) where {T, QD} = QD

function _orient_wmats!(q::QSpace{T, QD, N}) where {T, QD, N}
    for r in q.rows
        for cgr in r.cgrs
            if length(cgr.wmat.data) == 1 && cgr.wmat.data[1] < 0
                cgr.wmat[:] .*= -1
                r.RMT[:] .*= -one(T)
            end
        end
    end
end

function _drop_small_rows!(q::QSpace{T, QD, N}; cutoff::Float64 = QSPACE_ROW_CUTOFF) where {T, QD, N}
    rows = q.rows
    isempty(rows) && return

    row_norms_sq = [
        QD == 2 ? _cgt_size_2d(r.cgrs, symm(q)) * sum(abs2, r.RMT.data) : sum(abs2, r.RMT.data)
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
# itags: Tuple{Vararg{AbstractString, QD}} — one tag per leg; all other QIndex
# fields are preserved.
function QSpace(q::QSpace{T, QD, N, RD}, itags::Tuple{Vararg{AbstractString, QD}}) where {T, QD, N, RD}
    new_inds = ntuple(l -> QIndex(itags[l], q.inds[l].dir, q.inds[l].plev,
                                  q.inds[l].lock, q.inds[l].dual), QD)
    # spaces remain the same since rows didn't change
    return QSpace(symm(q), q.rows, new_inds, q.spaces)
end

# Construct a QSpace with the same rows but with all QIndex fields replaced.
# inds: NTuple{QD, QIndex} — one full QIndex per leg.
# Arrow directions must match the original QSpace (only itags/lock/plev/dual may differ).
function QSpace(q::QSpace{T, QD, N, RD}, inds::NTuple{QD, QIndex}) where {T, QD, N, RD}
    @assert ntuple(l -> inds[l].dir, QD) == ntuple(l -> q.inds[l].dir, QD) "QSpace(q, inds): arrow directions must match the original QSpace on all legs"
    return QSpace(symm(q), q.rows, inds, q.spaces)
end

Base.getindex(q::QSpace, i::Int) = q.rows[i]
Base.length(q::QSpace) = length(q.rows)
Base.firstindex(q::QSpace) = firstindex(q.rows)
Base.lastindex(q::QSpace) = lastindex(q.rows)

function _normalize_qspace_row_index(i::Int, nrows::Int)
    i == 0 && throw(BoundsError(1:nrows, i))
    idx = i < 0 ? nrows + i + 1 : i
    1 <= idx <= nrows || throw(BoundsError(1:nrows, i))
    return idx
end

function _normalize_qspace_row_selector(selector, nrows::Int)
    if selector isa Colon
        return collect(1:nrows)
    elseif selector isa Integer
        return Int[_normalize_qspace_row_index(Int(selector), nrows)]
    elseif selector isa AbstractVector{Bool}
        length(selector) == nrows || throw(DimensionMismatch(
            "row selector of length $(length(selector)) does not match number of rows $nrows"))
        return findall(selector)
    elseif selector isa AbstractRange{<:Integer}
        return _normalize_qspace_row_selector(collect(selector), nrows)
    elseif selector isa AbstractVector{<:Integer}
        inds = Int[_normalize_qspace_row_index(Int(i), nrows) for i in selector]
        length(unique(inds)) == length(inds) || throw(ArgumentError(
            "row selector must not contain duplicate indices"))
        return inds
    else
        throw(ArgumentError(
            "row selector must be :, Int, AbstractRange{<:Integer}, AbstractVector{<:Integer}, or AbstractVector{Bool}"))
    end
end

"""
    QSpace(q::QSpace, selector) -> QSpace

Create a new `QSpace` from a subset of `q.rows`, preserving the original
symmetry tuple, leg indices, and cached leg-space metadata in `q.spaces`.

`selector` may be `:`, an `Int`, an integer range, an integer vector, or a
boolean mask. Negative integer indices count from the end.
"""
function QSpace(q::QSpace{T, QD, N, RD}, selector) where {T, QD, N, RD}
    inds = _normalize_qspace_row_selector(selector, length(q.rows))
    return QSpace(symm(q), deepcopy(q.rows[inds]), q.inds, _copy_spaces_tuple(q.spaces))
end

Base.getindex(q::QSpace,
              selector::Union{Colon, AbstractRange{<:Integer},
                              AbstractVector{<:Integer}, AbstractVector{Bool}}) = QSpace(q, selector)

# For 0-dimensional QSpace (scalar), q[] returns the unique RMT element.
function Base.getindex(q::QSpace{T, 0, N, N}) where {T, N}
    @assert length(q.rows) <= 1 "0D QSpace must have zero or one row"
    if length(q.rows) == 1 
        @assert length(q.rows[1].RMT.data) == 1 "0D QSpace RMT must be a scalar"
        return only(q.rows[1].RMT.data)
    else return zero(T) end
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
function _matches_criteria(idx::QIndex; dir=nothing, itag=nothing, plev=nothing, lock=nothing)
    (!isnothing(dir)  && idx.dir != dir)                                 && return false
    (!isnothing(itag) && !_matches_itag_selector(idx.itags, itag))       && return false
    (!isnothing(plev) && idx.plev != plev)                               && return false
    (!isnothing(lock) && idx.lock != lock)                               && return false
    return true
end

"""
    findlegs(q::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Find all leg indices matching the specified criteria. Unspecified criteria match any value.
If `rev=true`, the selection is reversed: legs that do *not* match are returned.

# Arguments
- `dir`: Match direction ('+' for incoming, '-' for outgoing)
- `itag`: Match exact tags. A single string like `"a,b"` means both `a` and `b`
  must be present; a tuple/vector of strings means any one of those tag-sets may match
- `plev`: Match exact prime level
- `lock`: Match exact lock level
- `rev`: If `true`, return legs that do *not* satisfy the criteria (default `false`)

# Examples
```julia
findlegs(q; dir='-')                    # all outgoing legs
findlegs(q; itag="site")                # legs carrying tag "site"
findlegs(q; itag=("a,b", "a,c"))        # legs with tags a+b, or a+c
findlegs(q; dir='+', plev=0)            # incoming, unprimed legs
findlegs(q; lock=0)                     # non-locked legs
findlegs(q; dir='-', rev=true)          # all legs that are NOT outgoing
```
"""
function findlegs(q::QSpace{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    return [i for i in 1:QD if _matches_criteria(q.inds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev]
end

"""
    findleg(q::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Find the first leg index matching the specified criteria.
Returns `nothing` if no leg matches.
If `rev=true`, returns the first leg that does *not* match the criteria.

# Examples
```julia
findleg(q; itag="bond")           # first leg with "bond" in tag
findleg(q; dir='+')               # first incoming leg
findleg(q; lock=0)                # first non-locked leg
findleg(q; dir='+', rev=true)     # first leg that is NOT incoming
```
"""
function findleg(q::QSpace{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    for i in 1:QD
        _matches_criteria(q.inds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

function _matching_targets(q::QSpace; require_unlocked::Bool=false)
    return Set(change_dir(idx) for idx in q.inds
               if !require_unlocked || idx.lock == 0)
end

_has_target_match(idx::QIndex, targets; require_unlocked::Bool=false) =
    (!require_unlocked || idx.lock == 0) && (idx in targets)

function _find_matching_legs(a::QSpace{T, QD}, b::QSpace;
                             dir=nothing, itag=nothing, plev=nothing,
                             lock=nothing, rev::Bool=false,
                             matched::Bool=true,
                             require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    return [i for i in 1:QD
            if (_has_target_match(a.inds[i], targets; require_unlocked=require_unlocked) == matched) &&
               (_matches_criteria(a.inds[i]; dir=dir, itag=itag,
                                  plev=plev, lock=lock) ⊻ rev)]
end

function _find_matching_leg(a::QSpace{T, QD}, b::QSpace;
                            dir=nothing, itag=nothing, plev=nothing,
                            lock=nothing, rev::Bool=false,
                            matched::Bool=true,
                            require_unlocked::Bool=false) where {T, QD}
    targets = _matching_targets(b; require_unlocked=require_unlocked)
    for i in 1:QD
        (_has_target_match(a.inds[i], targets; require_unlocked=require_unlocked) == matched) || continue
        _matches_criteria(a.inds[i]; dir=dir, itag=itag, plev=plev, lock=lock) ⊻ rev && return i
    end
    return nothing
end

"""
    matchings(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one matching leg in `b`.

A leg matches when it has the same `itags`, `plev`, and `dual` flag as a leg
of `b`, but with opposite direction. Lock is ignored for the cross-tensor
match test. Keyword arguments filter the returned legs of `a` using the same
rules as `findlegs`.
"""
function matchings(a::QSpace{T, QD}, b::QSpace;
                   dir=nothing, itag=nothing, plev=nothing,
                   lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true)
end

"""
    matching(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a matching leg in `b`, or `nothing`
if no such leg exists.

Matching ignores lock between tensors; keyword arguments filter the returned
leg of `a` using the same rules as `findleg`.
"""
function matching(a::QSpace{T, QD}, b::QSpace;
                  dir=nothing, itag=nothing, plev=nothing,
                  lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true)
end

"""
    unmatchings(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any matching leg in `b`.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned legs of `a` using the same rules as `findlegs`.
"""
function unmatchings(a::QSpace{T, QD}, b::QSpace;
                     dir=nothing, itag=nothing, plev=nothing,
                     lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false)
end

"""
    unmatching(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any matching leg in `b`,
or `nothing` if every leg is matched.

The match test uses the same rule as `matchings`: same `itags`, `plev`, and
`dual`, opposite direction, and lock ignored between tensors. Keyword
arguments filter the returned leg of `a` using the same rules as `findleg`.
"""
function unmatching(a::QSpace{T, QD}, b::QSpace;
                    dir=nothing, itag=nothing, plev=nothing,
                    lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false)
end

"""
    contractables(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that have at least one contractable leg in `b`.

Two legs are contractable when they satisfy the same cross-tensor match rule as
`matchings` and both legs have `lock == 0`. Keyword arguments filter the
returned legs of `a` using the same rules as `findlegs`.
"""
function contractables(a::QSpace{T, QD}, b::QSpace;
                       dir=nothing, itag=nothing, plev=nothing,
                       lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=true, require_unlocked=true)
end

"""
    contractable(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that has a contractable leg in `b`, or
`nothing` if no such leg exists.

Contractable legs satisfy the same cross-tensor match rule as `matchings`, but
both tensors must have `lock == 0` on the matched legs.
"""
function contractable(a::QSpace{T, QD}, b::QSpace;
                      dir=nothing, itag=nothing, plev=nothing,
                      lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=true, require_unlocked=true)
end

"""
    uncontractables(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Vector{Int}

Return all leg indices of `a` that do not have any contractable leg in `b`.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs. Keyword arguments filter the returned legs of
`a` using the same rules as `findlegs`.
"""
function uncontractables(a::QSpace{T, QD}, b::QSpace;
                         dir=nothing, itag=nothing, plev=nothing,
                         lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_legs(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                               rev=rev, matched=false, require_unlocked=true)
end

"""
    uncontractable(a::QSpace, b::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false) -> Union{Int, Nothing}

Return the first leg index of `a` that does not have any contractable leg in
`b`, or `nothing` if every eligible leg is contractable.

Contractability requires the same cross-tensor match rule as `matchings`, plus
`lock == 0` on both matched legs.
"""
function uncontractable(a::QSpace{T, QD}, b::QSpace;
                        dir=nothing, itag=nothing, plev=nothing,
                        lock=nothing, rev::Bool=false) where {T, QD}
    return _find_matching_leg(a, b; dir=dir, itag=itag, plev=plev, lock=lock,
                              rev=rev, matched=false, require_unlocked=true)
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
        new_inds[i] = QIndex(idx.itags, idx.dir, idx.plev, new_lock, idx.dual)
    end
    return QSpace(symm(q), q.rows, Tuple(new_inds), q.spaces)
end

# Lock increase function (respects permanent lock)
_lock_inc(current_lock, inc) = current_lock == -1 ? -1 : current_lock + inc

"""
    lock(q::QSpace, leg::Integer; inc::Int=1)

Increase lock level of a single specified leg by `inc` (default 1).
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::QSpace, leg::Integer; inc::Int=1)
    return _modify_lock(q, (leg,), lk -> _lock_inc(lk, inc))
end

"""
    lock(q::QSpace, legs::LegList; inc::Int=1)

Increase lock level of the specified legs by `inc` (default 1).
`legs` can be any vector, range, or tuple of integers, e.g. `[1, 3]`, `1:3`, or `(1, 3)`.
Permanently locked legs (lock=-1) are unchanged.
"""
function Base.lock(q::QSpace, legs::LegList; inc::Int=1)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::QSpace, pred::Function; inc::Int=1)

Increase lock level of legs satisfying predicate by `inc` (default 1).
"""
function Base.lock(q::QSpace, pred::Function; inc::Int=1)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, lk -> _lock_inc(lk, inc))
end

"""
    lock(q::QSpace; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Increase lock level of legs matching criteria by `inc`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lock(q; dir='-')                  # lock all outgoing legs by 1
lock(q; inc=2, itag="bond")       # lock legs with "bond" in tag by 2
lock(q; lock=0)                   # lock all currently-unlocked legs by 1
lock(q; dir='-', rev=true)        # lock all legs that are NOT outgoing
```
"""
function Base.lock(q::QSpace; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
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
    lockp(q::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Permanently lock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
lockp(q; itag="phys")          # permanently lock physical legs
lockp(q; lock=0)               # permanently lock all currently-unlocked legs
lockp(q; itag="phys", rev=true)   # permanently lock all legs except "phys"
```
"""
function lockp(q::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_lock(q, legs, _ -> -1)
end

"""
    unlock(q::QSpace, leg::Integer)

Unlock a single specified leg (set lock=0). Also removes permanent lock.
"""
function Base.unlock(q::QSpace, leg::Integer)
    return _modify_lock(q, (leg,), _ -> 0)
end

"""
    unlock(q::QSpace, legs::LegList)

Unlock the specified legs (set lock=0).
`legs` can be any vector, range, or tuple of integers.
"""
function Base.unlock(q::QSpace, legs::LegList)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::QSpace, pred::Function)

Unlock legs satisfying predicate.
"""
function Base.unlock(q::QSpace, pred::Function)
    legs = findlegs(q, pred)
    return _modify_lock(q, legs, _ -> 0)
end

"""
    unlock(q::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev=false)

Unlock legs matching criteria.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
unlock(q; dir='-')             # unlock all outgoing legs
unlock(q; lock=1)              # unlock all legs currently at lock=1
unlock(q; dir='-', rev=true)   # unlock all legs that are NOT outgoing
```
"""
function Base.unlock(q::QSpace; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false)
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
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
# Keyword selectors for criteria forms (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

# legs can be any iterable of integers
function _modify_plev(q::QSpace{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = QIndex(idx.itags, idx.dir, modify_fn(idx.plev), idx.lock, idx.dual)
    end
    return QSpace(symm(q), q.rows, Tuple(new_inds), q.spaces)
end

"""
    prime(q::QSpace; inc::Int=1, dir, itag, plev, lock, rev=false)

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
prime(q; itag="site")           # legs whose tag contains "site"
prime(q; plev=0)                # only currently unprimed legs
prime(q; dir='+', rev=true)     # all legs that are NOT incoming
```
"""
function prime(q::QSpace{T, QD}; inc::Int=1, dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
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
    setprime(q::QSpace, n::Int; dir, itag, plev, lock, rev=false)

Set the prime level of matching legs to `n`. `n` must be non-negative.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setprime(q, 2)                  # set all legs to plev=2
setprime(q, 1; dir='-')         # set outgoing legs to plev=1
setprime(q, 0; itag="bond")     # same as noprime(q; itag="bond")
setprime(q, 0; dir='-', rev=true)  # clear prime on all non-outgoing legs
```
"""
function setprime(q::QSpace{T, QD}, n::Int; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    n >= 0 || throw(ArgumentError("prime level must be non-negative, got $n"))
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
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
    noprime(q::QSpace; dir, itag, plev, lock, rev=false)

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
function noprime(q::QSpace{T, QD}; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
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
#   additag(q, newtags; kw...)     – add tag(s) to matching legs
#   removeitag(q, tags; kw...)     – remove tag(s) from matching legs
#   setitag(q, tags; kw...)        – replace entire tag string of matching legs
#
# Keyword selectors (all optional): dir, itag, plev, lock
# ─────────────────────────────────────────────────────────────────────────────

const ITagQuerySpec = Union{AbstractString,
                            Tuple{Vararg{AbstractString}},
                            AbstractVector{<:AbstractString}}

function _modify_itag(q::QSpace{T, QD, N, RD}, legs, modify_fn::Function) where {T, QD, N, RD}
    new_inds = collect(q.inds)
    for i in legs
        idx = new_inds[i]
        new_inds[i] = QIndex(modify_fn(idx.itags), idx.dir, idx.plev, idx.lock, idx.dual)
    end
    return QSpace(symm(q), q.rows, Tuple(new_inds), q.spaces)
end

"""
    additag(q::QSpace, newtags; dir, itag, plev, lock, rev=false)

Add one or more tags to matching legs. `newtags` may be a comma-separated
string of tags (e.g. `"bond,u1"`). The result is always sorted.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
additag(q, "site")              # add "site" to all legs
additag(q, "phys"; dir='+')     # add "phys" to incoming legs only
additag(q, "u1"; itag="bond")   # add "u1" to legs that already have "bond"
additag(q, "aux"; dir='+', rev=true)  # add "aux" to all non-incoming legs
```
"""
function additag(q::QSpace{T, QD}, newtags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::QSpace, leg::Integer, newtags)

Add tags to a single specified leg.
"""
function additag(q::QSpace{T, QD}, leg::Integer, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, (leg,), base -> _add_itag(base, newtags))
end

"""    additag(q::QSpace, legs::LegList, newtags)

Add tags to the specified legs. `legs` can be any vector, range, or tuple.
"""
function additag(q::QSpace{T, QD}, legs::LegList, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, legs, base -> _add_itag(base, newtags))
end

"""    additag(q::QSpace, pred::Function, newtags)

Add tags to legs satisfying predicate.
"""
function additag(q::QSpace{T, QD}, pred::Function, newtags::AbstractString) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _add_itag(base, newtags))
end

"""
    removeitag(q::QSpace, tags; dir, itag, plev, lock, rev=false)

Remove one or more tags from matching legs.

`tags` may be either a single comma-separated tag string or a tuple/vector of
such strings. For tuple/vector input, each query is applied independently and
its tags are removed only from legs that contain all tags in that query.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
removeitag(q, "site")                     # remove "site" from all legs
removeitag(q, ("aaa,bbb", "ccc"))       # remove aaa+bbb together, and ccc independently
removeitag(q, "phys"; dir='-')           # remove "phys" from outgoing legs
removeitag(q, "aux"; dir='-', rev=true)  # remove "aux" from all non-outgoing legs
```
"""
function removeitag(q::QSpace{T, QD}, tags::ITagQuerySpec; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::QSpace, leg::Integer, tags)

Remove tags from a single specified leg.
"""
function removeitag(q::QSpace{T, QD}, leg::Integer, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, (leg,), base -> _remove_itag(base, tags))
end

"""    removeitag(q::QSpace, legs::LegList, tags)

Remove tags from the specified legs. `legs` can be any vector, range, or tuple.
"""
function removeitag(q::QSpace{T, QD}, legs::LegList, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, legs, base -> _remove_itag(base, tags))
end

"""    removeitag(q::QSpace, pred::Function, tags)

Remove tags from legs satisfying predicate.
"""
function removeitag(q::QSpace{T, QD}, pred::Function, tags::ITagQuerySpec) where {T, QD}
    return _modify_itag(q, findlegs(q, pred), base -> _remove_itag(base, tags))
end

"""
    setitag(q::QSpace, tags; dir, itag, plev, lock, rev=false)

Replace the entire tag string of matching legs with `tags`.
Use `rev=true` to act on legs that do *not* match the criteria.

# Examples
```julia
setitag(q, "bond")              # set all legs to tag "bond"
setitag(q, ""; dir='+')         # clear tags on incoming legs
setitag(q, "phys"; itag="site") # rename "site" → "phys" (full replacement)
setitag(q, "aux"; dir='+', rev=true)  # set tag on all non-incoming legs
```
"""
function setitag(q::QSpace{T, QD}, tags::AbstractString; dir=nothing, itag=nothing, plev=nothing, lock=nothing, rev::Bool=false) where {T, QD}
    legs = findlegs(q; dir=dir, itag=itag, plev=plev, lock=lock, rev=rev)
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::QSpace, leg::Integer, tags)

Set the entire tag string of a single specified leg.
"""
function setitag(q::QSpace{T, QD}, leg::Integer, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, (leg,), _ -> norm)
end

"""    setitag(q::QSpace, legs::LegList, tags)

Set the entire tag string of the specified legs. `legs` can be any vector, range, or tuple.
"""
function setitag(q::QSpace{T, QD}, legs::LegList, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, legs, _ -> norm)
end

"""    setitag(q::QSpace, pred::Function, tags)

Set the entire tag string of legs satisfying predicate.
"""
function setitag(q::QSpace{T, QD}, pred::Function, tags::AbstractString) where {T, QD}
    norm = _normalize_itag(tags)
    return _modify_itag(q, findlegs(q, pred), _ -> norm)
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

_qindex_plev_string(plev::Int) =
    plev == 0 ? "" : "p$(plev)"

_qindex_lock_string(lock::Int) = lock == 0 ? "" : "🔒$(lock)"

function _format_qindex(idx::QIndex)
    return "\"$(idx.itags)$(idx.dir)\"$(_qindex_plev_string(idx.plev))$(_qindex_lock_string(idx.lock))"
end

# Special pretty-printing for 0-dimensional QSpace (scalar result of full contraction).
function Base.show(io::IO, ::MIME"text/plain", qs::QSpace{T, 0, N, N}) where {T, N}
    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "0D QSpace{$T}, $N symmetries [$symm_names]: ", _fmt_scalar_str(qs[]))
end

function Base.show(io::IO, ::MIME"text/plain", qs::QSpace{T, QD, N, RD}) where {T, QD, N, RD}
    # --- Header: symmetries and leg dirs/tags on one line ---
    # Format:  QSpace{...}  [Sym1, Sym2]  ["tag1"+, "tag2"-', ...]
    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "$(QD)D QSpace, $N symmetries [$symm_names]")
    leg_strs = map(qs.inds) do idx
        raw = _format_qindex(idx)
        idx.dual ? "\e[32m$(raw)\e[0m" : raw
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
            _label_widths(qs.rows[i].cgrs, symm(qs))[n]
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
        _print_cgt_dims(io, r.cgrs, symm(qs))
        _print_qlabels(io, r.cgrs, widths)
        length(r.RMT.data) == 1 && print(io, "\t", lpad(_fmt_scalar_str(only(r.RMT.data)), scalar_width))
        QD == 2 && print(io, "\t√", _cgt_size_2d(r.cgrs, symm(qs)))
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
                S, cgr = symm(q)[i], r.cgrs[i]
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

    id_q = getIdentity((q, out_leg); itag=q.inds[out_leg].itags)
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
#   where dim_r = _cgt_size_2d(r.cgrs, symm(q)) = ∏_{non-abelian n} d_leg1^(n).
#
# ─────────────────────────────────────────────────────────────────────────────
function LinearAlgebra.norm(q::QSpace{T, QD, N}) where {T, QD, N}
    s = zero(Float64)
    if QD == 2
        for r in q.rows
            d = _cgt_size_2d(r.cgrs, symm(q))
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
            "(itag=\"$(inds1[i].itags)\", dir='$(inds1[i].dir)')")
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

function Base.:+(qs1::QSpace{T1, QD, N, RD, QT, PS, CGRS},
                 qs2::QSpace{T2, QD, N, RD, QT, PS, CGRS}) where {T1, T2, QD, N, RD, QT, PS, CGRS}
    @assert symm(qs1) == symm(qs2) "QSpace objects must share the same symmetry tuple"

    if qs1.inds != qs2.inds || qs1.spaces != qs2.spaces
        perm = _find_leg_permutation(qs1.inds, qs1.spaces, qs2.inds, qs2.spaces)
        qs2  = permutedims(qs2, perm)
    end

    T = promote_type(T1, T2)

    # Physical q-label key for a row: the cgp-permuted qlabels per symmetry.
    # Two rows belong to the same sector iff these match for every symmetry.
    row_key(r) = ntuple(Val(N)) do n
        cgr = r.cgrs[n]
        ntuple(l -> cgr.qlabels[cgr.cgp[l]], Val(QD))
    end

    dict1 = Dict(row_key(r) => r for r in qs1.rows)
    dict2 = Dict(row_key(r) => r for r in qs2.rows)

    new_rows = Vector{row{T, QD, N, RD, CGRS}}()

    for key in union(keys(dict1), keys(dict2))
        in1 = haskey(dict1, key)
        in2 = haskey(dict2, key)

        if in1 && !in2
            # Sector exists only in qs1 — copy with promoted element type.
            r = dict1[key]
            push!(new_rows, row(r.cgrs, LurTensor(T.(r.RMT.data))))

        elseif in2 && !in1
            # Sector exists only in qs2 — copy with promoted element type.
            r = dict2[key]
            push!(new_rows, row(r.cgrs, LurTensor(T.(r.RMT.data))))

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
            # TODO: When GPU support is added, 'Array{Float64, 2}' here can be generalized
            new_wmats = ntuple(Val(N)) do n
                LurTensor{Float64, 2, Array{Float64, 2}}[r1.cgrs[n].wmat, r2.cgrs[n].wmat]
            end
            new_RMTs  = LurTensor{T, RD, Array{T, RD}}[LurTensor(T.(r1.RMT.data)),
                                                       LurTensor(T.(r2.RMT.data))]
            new_qlabs = ntuple(Val(N)) do n
                (r1.cgrs[n].qlabels, r1.cgrs[n].cgp, r1.cgrs[n].legdir)
            end

            new_row = merge_new_row(new_wmats, new_RMTs, new_qlabs, PS, QD)
            isnothing(new_row) || push!(new_rows, new_row)
        end
    end

    return QSpace(PS, new_rows, qs1.inds, qs1.spaces)
end

Base.:-(qs1::QSpace, qs2::QSpace) = qs1 + (-1 * qs2)
Base.:+(q::QSpace, x::Number) = q + x * _identity_on_qspace(q)
Base.:+(x::Number, q::QSpace) = q + x
Base.:-(q::QSpace, x::Number) = q + (-x) * _identity_on_qspace(q)
Base.:-(x::Number, q::QSpace) = x * _identity_on_qspace(q) + (-q)

function _normalize_oplus_dims(dimensions, QD::Int; sort_dims::Bool=true)
    dims = dimensions isa Integer ? (Int(dimensions),) : Tuple(Int(d) for d in dimensions)
    isempty(dims) && throw(ArgumentError("oplus requires at least one dimension"))
    all(d -> 1 <= d <= QD, dims) || throw(ArgumentError(
        "oplus dimensions must lie in 1:$QD, got $(collect(dims))"))
    length(unique(dims)) == length(dims) || throw(ArgumentError(
        "oplus dimensions must be unique, got $(collect(dims))"))
    return sort_dims ? Tuple(sort(collect(dims))) : dims
end

function _normalize_oplus_matrix_dims(dimensions, QD::Int)
    dimensions isa Tuple && length(dimensions) == 2 || throw(ArgumentError(
        "matrix oplus requires exactly two axis-dimension specifications"))

    dims1 = _normalize_oplus_dims(dimensions[1], QD)
    dims2 = _normalize_oplus_dims(dimensions[2], QD)
    isempty(intersect(dims1, dims2)) || throw(ArgumentError(
        "matrix-axis oplus legs must be disjoint, got $(collect(dims1)) and $(collect(dims2))"))
    return dims1, dims2
end

function _splist_dim_map(splist::Vector)
    dims = Dict{Any, Int}()
    for (qlabels, dim) in splist
        if haskey(dims, qlabels)
            throw(ArgumentError("duplicate qlabel sector encountered in space list: $qlabels"))
        end
        dims[qlabels] = dim
    end
    return dims
end

function _sum_splists_many(splists)
    isempty(splists) && throw(ArgumentError("cannot sum an empty collection of space lists"))

    dims = Dict{Any, Int}()
    seen = Set{Any}()
    result = copy(first(splists))
    empty!(result)

    for splist in splists
        for (qlabels, dim) in splist
            dims[qlabels] = get(dims, qlabels, 0) + dim
            if qlabels ∉ seen
                push!(result, (qlabels, 0))
                push!(seen, qlabels)
            end
        end
    end

    for i in eachindex(result)
        qlabels, _ = result[i]
        result[i] = (qlabels, dims[qlabels])
    end

    return result
end

_copy_spaces_tuple(spaces::NTuple{QD, Vector}) where {QD} = ntuple(l -> copy(spaces[l]), QD)
_qspace_eltype(::QSpace{T}) where {T} = T

function _oplus_row_qlabel(r::row{T, QD, N}, leg::Int) where {T, QD, N}
    return Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[leg]] for n in 1:N)
end

function _pad_row_for_oplus(r::row{T, QD, N, RD},
                            dims_set::Set{Int},
                            start_dim_maps,
                            result_dim_maps) where {T, QD, N, RD}
    old_sizes = size(r.RMT.data)
    new_phys_sizes = collect(old_sizes[1:QD])
    starts = ones(Int, QD)

    for leg in 1:QD
        leg ∈ dims_set || continue
        qlabels = _oplus_row_qlabel(r, leg)
        total_dim = result_dim_maps[leg][qlabels]

        new_phys_sizes[leg] = total_dim
        starts[leg] = get(start_dim_maps[leg], qlabels, 1)
    end

    new_sizes = Tuple(vcat(new_phys_sizes, collect(old_sizes[QD+1:end])))
    new_data = zeros(T, new_sizes)
    fill_inds = ntuple(axis -> begin
        if axis <= QD && axis ∈ dims_set
            start = starts[axis]
            stop = start + old_sizes[axis] - 1
            start:stop
        else
            Colon()
        end
    end, RD)
    new_data[fill_inds...] = r.RMT.data

    return row(deepcopy(r.cgrs), LurTensor(new_data))
end

function _oplus_pad_qspace(q::QSpace{T, QD, N, RD},
                           result_spaces,
                           dims_tuple,
                           start_dim_maps,
                           result_dim_maps) where {T, QD, N, RD}
    dims_set = Set(dims_tuple)
    new_rows = row{T, QD, N, RD}[]
    for r in q.rows
        push!(new_rows, _pad_row_for_oplus(r, dims_set, start_dim_maps, result_dim_maps))
    end
    return QSpace(symm(q), new_rows, q.inds, _copy_spaces_tuple(result_spaces))
end

function _zero_qspace_with_spaces(symm::NTuple{N, Any},
                                  inds::NTuple{QD, QIndex},
                                  spaces::NTuple{QD, Vector};
                                  T::Type=Float64) where {N, QD}
    rows = Vector{row{T, QD, N, QD + N}}()
    return QSpace(symm, rows, inds, _copy_spaces_tuple(spaces))
end

function _accumulate_oplus_starts(qs, dims_tuple, QD::Int)
    dims_set = Set(dims_tuple)
    running = [Dict{Any, Int}() for _ in 1:QD]
    starts = Vector{Vector{Dict{Any, Int}}}(undef, length(qs))

    for (qi, q) in enumerate(qs)
        starts[qi] = [Dict{Any, Int}() for _ in 1:QD]
        for leg in 1:QD
            leg ∈ dims_set || continue
            for (qlabels, dim) in q.spaces[leg]
                start = get(running[leg], qlabels, 1)
                starts[qi][leg][qlabels] = start
                running[leg][qlabels] = start + dim
            end
        end
    end

    return starts
end

_qindex_match_for_oplus(a::QIndex, b::QIndex) =
    a.itags == b.itags && a.dir == b.dir && a.plev == b.plev && a.lock == b.lock

function _inds_match_for_oplus(inds1, inds2)
    length(inds1) == length(inds2) || return false
    return all(_qindex_match_for_oplus(idx1, idx2) for (idx1, idx2) in zip(inds1, inds2))
end

function _validate_oplus_common(qs)
    isempty(qs) && throw(ArgumentError("oplus requires at least one QSpace"))
    first(qs) isa QSpace || throw(ArgumentError("oplus entry 1 is not a QSpace"))

    ref = first(qs)
    for (i, q) in enumerate(qs)
        q isa QSpace || throw(ArgumentError("oplus entry $i is not a QSpace"))
        symm(q) == symm(ref) || throw(ArgumentError(
            "QSpace entry $i has a different symmetry tuple"))
        length(q.inds) == length(ref.inds) || throw(ArgumentError(
            "QSpace entry $i has rank $(length(q.inds)), expected $(length(ref.inds))"))
        _inds_match_for_oplus(q.inds, ref.inds) || throw(ArgumentError(
            "QSpace entry $i has different indices (ignoring green)"))
    end

    return ref
end

function _build_vector_oplus_spaces(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    QD = length(ref.inds)
    dims_set = Set(dims_tuple)

    return ntuple(leg -> begin
        if leg ∈ dims_set
            _sum_splists_many([q.spaces[leg] for q in qs])
        else
            target = ref.spaces[leg]
            for i in 2:length(qs)
                qs[i].spaces[leg] == target || throw(ArgumentError(
                    "space lists must match on non-oplus leg $leg"))
            end
            copy(target)
        end
    end, QD)
end

function _materialize_vector_oplus(qs, dims_tuple)
    ref = _validate_oplus_common(qs)
    result_spaces = _build_vector_oplus_spaces(qs, dims_tuple)
    QD = length(ref.inds)
    result_dim_maps = ntuple(leg -> _splist_dim_map(result_spaces[leg]), QD)
    start_maps = _accumulate_oplus_starts(qs, dims_tuple, QD)
    T = promote_type((_qspace_eltype(q) for q in qs)...)

    acc = _zero_qspace_with_spaces(symm(ref), ref.inds, result_spaces; T=T)
    for (q, qstarts) in zip(qs, start_maps)
        padded = _oplus_pad_qspace(q, result_spaces, dims_tuple, qstarts, result_dim_maps)
        padded = q.inds == ref.inds ? padded : QSpace(padded, ref.inds)
        acc = acc + padded
    end
    return acc
end

function _oplus_matrix_entry(mat, i::Int, j::Int)
    if applicable(isassigned, mat, i, j) && !isassigned(mat, i, j)
        return nothing
    end

    val = mat[i, j]
    if val === nothing || val === missing
        return nothing
    end
    val isa QSpace || throw(ArgumentError(
        "matrix oplus entry ($i, $j) is neither a QSpace nor an undefined entry"))
    return val
end

function _infer_zero_matrix_spaces(row_sources, col_sources, i::Int, j::Int, QD::Int)
    return ntuple(leg -> begin
        have_row = haskey(row_sources[i], leg)
        have_col = haskey(col_sources[j], leg)
        if have_row && have_col
            row_sources[i][leg] == col_sources[j][leg] || throw(ArgumentError(
                "cannot infer zero QSpace at ($i, $j): inconsistent spaces on leg $leg"))
            copy(row_sources[i][leg])
        elseif have_row
            copy(row_sources[i][leg])
        elseif have_col
            copy(col_sources[j][leg])
        else
            throw(ArgumentError(
                "cannot infer zero QSpace at ($i, $j): missing space information on leg $leg"))
        end
    end, QD)
end

"""
    oplus(qs::AbstractVector, dimensions)

Direct sum of a vector of `QSpace` objects along one or more physical legs.
"""
function oplus(qs::AbstractVector, dimensions)
    isempty(qs) && throw(ArgumentError("oplus requires at least one QSpace"))
    any(q -> q === nothing || q === missing, qs) && throw(ArgumentError(
        "vector oplus requires every entry to be well defined"))

    ref = _validate_oplus_common(collect(qs))
    dims_tuple = _normalize_oplus_dims(dimensions, length(ref.inds))
    return _materialize_vector_oplus(collect(qs), dims_tuple)
end

function oplus(q1::QSpace, q2::QSpace, dimensions)
    return oplus(QSpace[q1, q2], dimensions)
end

function _complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    size(mat, 1) > 0 && size(mat, 2) > 0 || throw(ArgumentError(
        "matrix oplus requires a non-empty matrix"))

    defined_positions = Tuple{Int, Int}[]
    defined_qs = QSpace[]
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        q === nothing && continue
        push!(defined_positions, (i, j))
        push!(defined_qs, q)
    end
    isempty(defined_qs) && throw(ArgumentError(
        "matrix oplus requires at least one defined QSpace to infer spaces"))

    ref = _validate_oplus_common(defined_qs)
    row_dims, col_dims = _normalize_oplus_matrix_dims(dimensions, length(ref.inds))
    row_dims_set = Set(row_dims)
    col_dims_set = Set(col_dims)

    row_sources = [Dict{Int, Any}() for _ in axes(mat, 1)]
    col_sources = [Dict{Int, Any}() for _ in axes(mat, 2)]

    for ((i, j), q) in zip(defined_positions, defined_qs)
        for leg in 1:length(ref.inds)
            if leg ∉ col_dims_set
                if haskey(row_sources[i], leg)
                    row_sources[i][leg] == q.spaces[leg] || throw(ArgumentError(
                        "row $i has incompatible spaces on leg $leg"))
                else
                    row_sources[i][leg] = copy(q.spaces[leg])
                end
            end
            if leg ∉ row_dims_set
                if haskey(col_sources[j], leg)
                    col_sources[j][leg] == q.spaces[leg] || throw(ArgumentError(
                        "column $j has incompatible spaces on leg $leg"))
                else
                    col_sources[j][leg] = copy(q.spaces[leg])
                end
            end
        end
    end

    T = promote_type((_qspace_eltype(q) for q in defined_qs)...)
    filled = Matrix{QSpace}(undef, size(mat, 1), size(mat, 2))
    for j in axes(mat, 2), i in axes(mat, 1)
        q = _oplus_matrix_entry(mat, i, j)
        if q === nothing
            spaces = _infer_zero_matrix_spaces(row_sources, col_sources, i, j, length(ref.inds))
            filled[i, j] = _zero_qspace_with_spaces(symm(ref), ref.inds, spaces; T=T)
        else
            filled[i, j] = q
        end
    end

    return filled, row_dims, col_dims
end

"""
    complete_oplus_matrix(mat::AbstractMatrix, dimensions)

Validate a matrix input for `oplus`, infer zero `QSpace` objects for undefined
entries, and return the completed `Matrix{QSpace}`.
"""
function complete_oplus_matrix(mat::AbstractMatrix, dimensions)
    filled, _, _ = _complete_oplus_matrix(mat, dimensions)
    return filled
end

function oplus(mat::AbstractMatrix, dimensions)
    filled, row_dims, col_dims = _complete_oplus_matrix(mat, dimensions)

    col_aggregates = Vector{QSpace}(undef, size(filled, 2))
    for j in axes(filled, 2)
        col_aggregates[j] = _materialize_vector_oplus(vec(filled[:, j]), row_dims)
    end

    return _materialize_vector_oplus(col_aggregates, col_dims)
end



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
        new_RMT = LurTensor(conj.(r.RMT.data))

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
            S = symm(cgr)
            ins, outs = cgr.qlabels[1:m], cgr.qlabels[m+1:m+k]
            ins_, _ = remove_zeros(S, ins)
            outs_, _ = remove_zeros(S, outs)
            new_wmat =
                if !isabelian(S) && ins_ == outs_
                    is1j = detect_1j(S, ins_, outs_)
                    cgt_oms  = get_CGTom(S, ins_, outs_, is1j)
                    perm_vec = get_conj_perm(cgt_oms)
                    LurTensor(cgr.wmat.data[perm_vec, :])
                else
                    deepcopy(cgr.wmat)
                end

            CGR(symm(cgr), Tuple(new_qlabels), new_wmat, new_cgp, new_legdir)
        end

        row(Tuple(new_cgrs), new_RMT)
    end
    if isempty(new_rows) new_rows = row{T, QD, N, RD}[] 
    else new_rows = collect(new_rows) end

    # spaces remain the same: physical qlabels at each leg don't change in conj,
    # only the CGR internal structure (incoming/outgoing) changes
    return QSpace(symm(q), new_rows, new_inds, q.spaces)
end

Base.adjoint(q::QSpace) = conj(q)

getsub(q::QSpace, selector) = QSpace(q, selector)

function _normalize_getsub_index(i::Int, dim::Int, sector, leg::Int)
    i == 0 && throw(ArgumentError(
        "selector for sector $sector on leg $leg cannot contain 0"))
    return i < 0 ? dim + i + 1 : i
end

function _normalize_getsub_indices(raw, dim::Int, sector, leg::Int)
    inds = if raw isa Integer
        [Int(raw)]
    elseif raw isa Tuple && all(i -> i isa Integer, raw)
        Int[Int(i) for i in raw]
    elseif raw isa AbstractRange{<:Integer}
        Int[i for i in raw]
    elseif raw isa AbstractVector{<:Integer}
        Int[i for i in raw]
    else
        throw(ArgumentError(
            "selector for sector $sector on leg $leg must be :, Int, integer tuple, AbstractRange, or AbstractVector{<:Integer}"))
    end

    isempty(inds) && throw(ArgumentError(
        "selector for sector $sector on leg $leg must not be empty"))
    inds = [_normalize_getsub_index(i, dim, sector, leg) for i in inds]
    length(unique(inds)) == length(inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains duplicate indices"))
    all(1 <= i <= dim for i in inds) || throw(ArgumentError(
        "selector for sector $sector on leg $leg contains out-of-bounds indices for dimension $dim"))
    return inds
end
function _slice_qspace_row_legs(r::row{T, QD, N, RD}, picks_by_leg::Dict{Int, Any}) where {T, QD, N, RD}
    all(pick -> pick isa Colon, values(picks_by_leg)) && return r
    selectors = ntuple(d -> get(picks_by_leg, d, Colon()), RD)
    return row(r.cgrs, LurTensor(r.RMT.data[selectors...]))
end

function _normalize_getsub_predicate_pick(raw, dim::Int, sector, leg::Int)
    raw isa Bool && throw(ArgumentError("getsub predicate for sector $sector on leg $leg must not return Bool; use Colon() or nothing explicitly"))
    raw === nothing && return nothing
    raw isa Colon && return Colon()
    return _normalize_getsub_indices(raw, dim, sector, leg)
end

function _collect_getsub_predicate_picks(q::QSpace, positions, pred::Function)
    selected_picks = Dict{Int, Dict{Any, Any}}()
    for leg in positions
        picks = Dict{Any, Any}()
        for (sector, dim) in q.spaces[leg]
            pick = _normalize_getsub_predicate_pick(pred(sector), dim, sector, leg)
            isnothing(pick) && continue
            picks[sector] = pick
        end
        selected_picks[leg] = picks
    end
    return selected_picks
end

function _apply_getsub_picks(q::QSpace{T, QD, N, RD},
                             positions,
                             selected_picks::Dict{Int, Dict{Any, Any}};
                             preserve_space::Bool=false) where {T, QD, N, RD}
    if preserve_space
        for leg in positions
            for (sector, pick) in selected_picks[leg]
                pick isa Colon && continue
                throw(ArgumentError(
                    "preserve_space=true is incompatible with slicing sector $sector on leg $leg; return Colon() or nothing instead"))
            end
        end
    end

    selected_leg_set = Set(positions)
    rows_out = eltype(q.rows)[]
    for r in q.rows
        picks_by_leg = Dict{Int, Any}()
        keep = true
        for leg in positions
            sector = _oplus_row_qlabel(r, leg)
            picks = selected_picks[leg]
            haskey(picks, sector) || (keep = false; break)
            picks_by_leg[leg] = picks[sector]
        end
        keep || continue
        push!(rows_out, _slice_qspace_row_legs(r, picks_by_leg))
    end

    spaces_out = if preserve_space
        _copy_spaces_tuple(q.spaces)
    else
        ntuple(l -> begin
            if l in selected_leg_set
                out = eltype(q.spaces[l])[]
                picks = selected_picks[l]
                for (sector, dim) in q.spaces[l]
                    haskey(picks, sector) || continue
                    pick = picks[sector]
                    push!(out, (sector, pick isa Colon ? dim : length(pick)))
                end
                out
            else
                copy(q.spaces[l])
            end
        end, QD)
    end

    return QSpace(symm(q), rows_out, q.inds, spaces_out)
end

function _normalize_getsub_predicate_legs(q::QSpace{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[leg for leg in legs]
    isempty(positions) && throw(ArgumentError("getsub requires at least one leg"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "getsub legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "getsub legs must be unique, got $positions"))
    return positions
end

"""
    getsub(q::QSpace, leg::Integer, pred::Function; preserve_space::Bool=false) -> QSpace

Return a new `QSpace` containing only rows whose sector on `leg` satisfies
`pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), only `q.spaces[leg]` is truncated to
the retained sectors and all other leg-space lists are copied unchanged. If
`preserve_space=true`, all cached leg-space lists are preserved exactly and only
the rows are filtered. This requires `pred` to keep whole sectors, so any
index-selection return value is rejected when `preserve_space=true`.
"""
function getsub(q::QSpace{T, QD, N, RD}, leg::Integer, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    return getsub(q, (Int(leg),), pred; preserve_space=preserve_space)
end

"""
    getsub(q::QSpace, legs, pred::Function; preserve_space::Bool=false) -> QSpace

Return a new `QSpace` containing only rows whose sectors on every selected leg
satisfy `pred`.

`pred(sector)` may return `nothing` to drop that sector, `Colon()` to keep the full
sector, or an integer / integer range / integer tuple / integer vector to keep

If `preserve_space=false` (the default), each selected leg keeps only the
matching entries in its space list, while unselected legs keep copies of their
original space lists. If `preserve_space=true`, all cached leg-space lists are
preserved exactly and only the rows are filtered. This requires `pred` to keep
whole sectors, so any index-selection return value is rejected when
`preserve_space=true`.
"""
function getsub(q::QSpace{T, QD, N, RD}, legs::LegList, pred::Function; preserve_space::Bool=false) where {T, QD, N, RD}
    positions = _normalize_getsub_predicate_legs(q, legs)
    selected_picks = _collect_getsub_predicate_picks(q, positions, pred)
    return _apply_getsub_picks(q, positions, selected_picks; preserve_space=preserve_space)
end

"""
    getsub(q::QSpace, pred::Function; preserve_space::Bool=false, dir=nothing,
           itag=nothing, plev=nothing, lock=nothing, rev=false) -> QSpace

Apply predicate-based `getsub` to every leg selected by the keyword criteria.
The leg selection follows the same matching rules as `findlegs`.
"""
function getsub(q::QSpace{T, QD, N, RD}, pred::Function; preserve_space::Bool=false,
                dir=nothing, itag=nothing, plev=nothing, lock=nothing,
                rev::Bool=false) where {T, QD, N, RD}
    legs = _resolve_matching_legs(q; dir=dir, itag=itag, plev=plev, lock=lock,
                                  rev=rev, opname="getsub")
    return getsub(q, legs, pred; preserve_space=preserve_space)
end

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
    QT = qlabeltype(symm)
    spaces = ntuple(_ -> Vector{Tuple{QT, Int}}(), QD)
    return QSpace(symm, rows, inds, spaces)
end

function empty_qspace(q::QSpace; T::Type=Float64)
    return empty_qspace(symm(q), q.inds; T=T)
end

function Base.zero(q::QSpace{T, QD, N, RD}) where {T, QD, N, RD}
    rows = Vector{row{T, QD, N, RD}}()
    return QSpace(symm(q), rows, q.inds, _copy_spaces_tuple(q.spaces))
end

"""
    qlabeltype(symm::NTuple{N, Any}) where {N}
    qlabeltype(q::QSpace)

Return the qlabel type for one leg sector over the symmetries in `symm` or `q`.

For example, `(U1, SU{3})` returns `Tuple{Tuple{Int}, NTuple{2, Int}}`.
"""
function qlabeltype(symm::NTuple{N, Any}) where {N}
    return Tuple{ntuple(n -> NTuple{nzops(symm[n]), Int}, N)...}
end

qlabeltype(::Type{<:ProductSymm{Syms}}) where {Syms} =
    Tuple{(NTuple{nzops(S), Int} for S in Syms.parameters)...}

qlabeltype(::QSpace{T, QD, N, RD, QT}) where {T, QD, N, RD, QT} = QT

"""
    zero_qlabels(symm::NTuple{N, Any}) where {N}
    zero_qlabels(q::QSpace)

Return the trivial qlabel for each symmetry in `symm` or `q`.

For example, `(SU{2}, SU{3})` returns `((0,), (0, 0))`.
"""
function zero_qlabels(symm::NTuple{N, Any}) where {N}
    return ntuple(n -> Tuple(0 for _ in 1:nzops(symm[n])), N)
end

zero_qlabels(q::QSpace) = zero_qlabels(symm(q))

function _is_singleton_leg(q::QSpace{T, QD, N}, leg::Int) where {T, QD, N}
    1 <= leg <= QD || throw(ArgumentError("leg must lie in 1:$QD, got $leg"))
    return length(q.spaces[leg]) == 1 && only(q.spaces[leg]) == (zero_qlabels(q), 1)
end

_singleton_legs(q::QSpace{T, QD}) where {T, QD} = [leg for leg in 1:QD if _is_singleton_leg(q, leg)]

function _normalize_delete_singleton_legs(q::QSpace{T, QD}, legs) where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one deletion leg must be specified"))
    all(1 <= leg <= QD for leg in positions) || throw(ArgumentError(
        "singleton deletion legs must lie in 1:$QD, got $positions"))
    length(unique(positions)) == length(positions) || throw(ArgumentError(
        "singleton deletion legs must be unique, got $positions"))
    sort!(positions)
    return positions
end


function _delete_singleton_cgr(cgr::CGR{QD, NZ}, positions) where {QD, NZ}
    old_m, _ = cgr.legdir
    keep = [leg for leg in 1:QD if leg ∉ positions]
    new_qd = length(keep)

    incoming = Vector{Tuple{NTuple{NZ, Int}, Int}}()
    outgoing = Vector{Tuple{NTuple{NZ, Int}, Int}}()

    for (new_leg, old_leg) in enumerate(keep)
        stored_pos = cgr.cgp[old_leg]
        target = stored_pos <= old_m ? incoming : outgoing
        push!(target, (cgr.qlabels[stored_pos], new_leg))
    end

    sort!(incoming; by=first, alg=MergeSort)
    sort!(outgoing; by=first, alg=MergeSort)

    m_new = length(incoming)
    new_cgp = zeros(Int, new_qd)
    for (stored_pos, (_, phys_leg)) in enumerate(incoming)
        new_cgp[phys_leg] = stored_pos
    end
    for (offset, (_, phys_leg)) in enumerate(outgoing)
        new_cgp[phys_leg] = m_new + offset
    end

    new_qlabels = (Tuple(first.(incoming))..., Tuple(first.(outgoing))...)
    new_wmat = LurTensor(copy(cgr.wmat.data))
    return CGR(symm(cgr), new_qlabels, new_wmat, Tuple(new_cgp),
               (m_new, length(outgoing)))
end

function _delete_singleton_rmt(rmt::LurTensor{T, RD}, positions, qd::Int, n_symm::Int) where {T, RD}
    selectors = ntuple(axis -> axis <= qd && axis ∈ positions ? 1 : Colon(), qd + n_symm)
    return LurTensor(copy(rmt.data[selectors...]))
end

function _delete_singleton_impl(q::QSpace{T, QD, N, RD}, positions) where {T, QD, N, RD}
    new_qd = QD - length(positions)
    new_rd = RD - length(positions)

    keep_inds = [q.inds[leg] for leg in 1:QD if leg ∉ positions]
    QT = qlabeltype(q)
    keep_spaces = Vector{Tuple{QT, Int}}[q.spaces[leg] for leg in 1:QD if leg ∉ positions]

    new_rows = row{T, new_qd, N, new_rd}[]
    for r in q.rows
        new_cgrs = ntuple(n -> _delete_singleton_cgr(r.cgrs[n], positions), N)
        new_rmt = _delete_singleton_rmt(r.RMT, positions, QD, N)
        push!(new_rows, row(new_cgrs, new_rmt))
    end

    return QSpace(symm(q), new_rows, Tuple(keep_inds), Tuple(keep_spaces))
end

"""
    deleteSingleton(q::QSpace; dir=nothing, itag=nothing, plev=nothing) -> QSpace

Delete singleton legs matching the supplied criteria.

With no keyword arguments, all singleton legs are deleted.
Only singleton legs are eligible for deletion. If none match, a warning is
emitted and `q` is returned unchanged.
"""
function deleteSingleton(q::QSpace{T, QD}; dir=nothing, itag=nothing, plev=nothing) where {T, QD}
    singleton_legs = if isnothing(dir) && isnothing(itag) && isnothing(plev)
        _singleton_legs(q)
    else
        candidate_legs = findlegs(q; dir=dir, itag=itag, plev=plev)
        [leg for leg in candidate_legs if _is_singleton_leg(q, leg)]
    end

    if isempty(singleton_legs)
        if isnothing(dir) && isnothing(itag) && isnothing(plev)
            @warn "deleteSingleton: no singleton legs found"
        else
            @warn "deleteSingleton: no singleton legs matched the requested criteria"
        end
        return q
    end
    return _delete_singleton_impl(q, singleton_legs)
end

"""
    deleteSingleton(q::QSpace, leg::Integer) -> QSpace
    deleteSingleton(q::QSpace, legs::LegList) -> QSpace

Delete the specified singleton legs from `q`.

Every selected leg must be singleton. Otherwise an `ArgumentError` is thrown.
"""
function deleteSingleton(q::QSpace, leg::Integer)
    return deleteSingleton(q, (leg,))
end

function deleteSingleton(q::QSpace{T, QD}, legs::LegList) where {T, QD}
    positions = _normalize_delete_singleton_legs(q, legs)
    bad = [leg for leg in positions if !_is_singleton_leg(q, leg)]
    isempty(bad) || throw(ArgumentError(
        "deleteSingleton requires singleton legs, but legs $bad are not singleton"))
    return _delete_singleton_impl(q, positions)
end

function _expand_singleton_kw(values, count::Int, name::AbstractString)
    if values isa AbstractString || values isa Char || values isa Integer
        return fill(values, count)
    end
    collected = collect(values)
    length(collected) == count || throw(ArgumentError(
        "$name must have length $count, got $(length(collected))"))
    return collected
end

function _singleton_insert_spec(q::QSpace{T, QD}, legs;
                                itag="", plev=0, lock=0, dir='+') where {T, QD}
    positions = legs isa Integer ? [Int(legs)] : Int[i for i in legs]
    isempty(positions) && throw(ArgumentError("at least one insertion leg must be specified"))

    count = length(positions)
    itag_vec = _expand_singleton_kw(itag, count, "itag")
    plev_vec = _expand_singleton_kw(plev, count, "plev")
    lock_vec = _expand_singleton_kw(lock, count, "lock")
    dir_vec  = _expand_singleton_kw(dir,  count, "dir")

    perm = sortperm(positions)
    positions = positions[perm]
    itag_vec = itag_vec[perm]
    plev_vec = plev_vec[perm]
    lock_vec = lock_vec[perm]
    dir_vec  = dir_vec[perm]

    final_qd = QD + count
    all(p -> 1 <= p <= final_qd, positions) || throw(ArgumentError(
        "singleton insertion legs must lie in 1:$final_qd, got $positions"))
    length(unique(positions)) == count || throw(ArgumentError(
        "singleton insertion legs must be unique, got $positions"))

    for direction in dir_vec
        direction in ('+', '-') || throw(ArgumentError(
            "added leg directions must be '+' or '-', got '$direction'"))
    end

    return positions, itag_vec, plev_vec, lock_vec, dir_vec
end

function _insert_singleton_cgr(cgr::CGR{QD, NZ},
                               positions,
                               dirs,
                               trivial_qlabel::NTuple{NZ, Int}) where {QD, NZ}
    old_m, _ = cgr.legdir
    new_qd = QD + length(positions)

    incoming = Vector{Tuple{NTuple{NZ, Int}, Int}}()
    outgoing = Vector{Tuple{NTuple{NZ, Int}, Int}}()

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            if dirs[insert_idx] == '+'
                push!(incoming, (trivial_qlabel, new_leg))
            else
                push!(outgoing, (trivial_qlabel, new_leg))
            end
            insert_idx += 1
            continue
        end

        stored_pos = cgr.cgp[old_leg]
        target = stored_pos <= old_m ? incoming : outgoing
        push!(target, (cgr.qlabels[stored_pos], new_leg))
        old_leg += 1
    end

    sort!(incoming; by=first, alg=MergeSort)
    sort!(outgoing; by=first, alg=MergeSort)

    m_new = length(incoming)
    new_cgp = zeros(Int, new_qd)
    for (stored_pos, (_, phys_leg)) in enumerate(incoming)
        new_cgp[phys_leg] = stored_pos
    end
    for (offset, (_, phys_leg)) in enumerate(outgoing)
        new_cgp[phys_leg] = m_new + offset
    end

    new_qlabels = (Tuple(first.(incoming))..., Tuple(first.(outgoing))...)
    new_wmat = LurTensor(copy(cgr.wmat.data))
    return CGR(symm(cgr), new_qlabels, new_wmat, Tuple(new_cgp),
               (m_new, length(outgoing)))
end

function _insert_singleton_rmt(rmt::LurTensor{T, RD},
                               positions,
                               qd::Int,
                               n_symm::Int) where {T, RD}
    old_phys = size(rmt.data)[1:qd]
    om_dims = size(rmt.data)[qd+1:qd+n_symm]

    new_phys = Int[]
    sizehint!(new_phys, qd + length(positions))

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:(qd + length(positions))
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            push!(new_phys, 1)
            insert_idx += 1
        else
            push!(new_phys, old_phys[old_leg])
            old_leg += 1
        end
    end

    new_data = copy(reshape(rmt.data, Tuple(vcat(new_phys, collect(om_dims)))))
    return LurTensor(new_data)
end

function _convert_rank2_singleton_normalization!(new_cgrs, new_rmt::LurTensor, old_cgrs)
    for n in eachindex(new_cgrs)
        w_val = old_cgrs[n].wmat[1]
        new_cgrs[n].wmat[:] .= 1.0
        new_rmt[:] .*= w_val
    end
    return nothing
end

"""
    addSingleton(q::QSpace, legs; itag="", plev=0, lock=0, dir='+')

Insert one or more singleton trivial legs into `q`.

`legs` may be a single integer or any iterable of integers. Each value is the
position of an added leg in the output tensor, whose rank is `ndims(q) +
length(legs)`. The original legs keep their relative order.

`itag`, `plev`, `lock`, and `dir` may each be either a scalar applied to
every added leg or an iterable with one value per inserted leg.
"""
function addSingleton(q::QSpace{T, QD, N, RD}, legs;
                      itag="", plev=0, lock=0, dir='+') where {T, QD, N, RD}
    positions, itag_vec, plev_vec, lock_vec, dir_vec =
        _singleton_insert_spec(q, legs; itag=itag, plev=plev, lock=lock, dir=dir)

    new_qd = QD + length(positions)
    new_rd = RD + length(positions)
    trivial_qlabels = zero_qlabels(q)

    new_inds = Vector{QIndex}(undef, new_qd)
    QT = qlabeltype(q)
    new_spaces = Vector{Vector{Tuple{QT, Int}}}(undef, new_qd)
    singleton_space = [(trivial_qlabels, 1)]

    old_leg = 1
    insert_idx = 1
    for new_leg in 1:new_qd
        if insert_idx <= length(positions) && positions[insert_idx] == new_leg
            new_inds[new_leg] = QIndex(String(itag_vec[insert_idx]), dir_vec[insert_idx],
                                       plev_vec[insert_idx], lock_vec[insert_idx])
            new_spaces[new_leg] = copy(singleton_space)
            insert_idx += 1
        else
            new_inds[new_leg] = q.inds[old_leg]
            new_spaces[new_leg] = q.spaces[old_leg]
            old_leg += 1
        end
    end

    new_rows = row{T, new_qd, N, new_rd}[]
    for r in q.rows
        new_cgrs = ntuple(n -> _insert_singleton_cgr(r.cgrs[n], positions, dir_vec, trivial_qlabels[n]), N)
        new_rmt = _insert_singleton_rmt(r.RMT, positions, QD, N)
        if QD == 2 && new_qd > 2
            _convert_rank2_singleton_normalization!(new_cgrs, new_rmt, r.cgrs)
        end
        push!(new_rows, row(new_cgrs, new_rmt))
    end

    return QSpace(symm(q), new_rows, Tuple(new_inds), Tuple(new_spaces))
end

"""
    getvac(q::QSpace, itags::Tuple{Vararg{AbstractString, 2}}=("", "")) -> QSpace

Build the rank-2 vacuum QSpace associated with `q`.

The result keeps the same symmetry tuple as `q`, has one incoming leg and one
outgoing leg, and contains exactly one trivial sector with RMT dimension 1 on
each leg. If `itags` is provided, it is used as the tags of the two legs.
"""
function getvac(q::QSpace{T, QD, N, RD},
                itags::Tuple{Vararg{AbstractString, 2}}=("", "")) where {T, QD, N, RD}
    trivial_qlabels = zero_qlabels(q)
    space_entry = (trivial_qlabels, 1)

    cgrs = ntuple(n -> begin
        trivial_q = trivial_qlabels[n]
        CGR(symm(q)[n], (trivial_q, trivial_q), LurTensor([1.0;;]), (1, 2), (1, 1))
    end, N)

    rmt_data = fill(one(T), ntuple(_ -> 1, N + 2))
    rows = row{T, 2, N, N + 2}[row(cgrs, LurTensor(rmt_data))]
    inds = (QIndex(itags[1], '+'), QIndex(itags[2], '-'))
    QT = qlabeltype(q)
    space_template = Vector{Tuple{QT, Int}}([space_entry])
    spaces = (copy(space_template), copy(space_template))

    return QSpace(symm(q), rows, inds, spaces)
end

function ⊗(q1::QSpace{T1, QD1, N, RD1},
           q2::QSpace{T2, QD2, N, RD2}) where {T1, T2, QD1, QD2, N, RD1, RD2}
    @assert symm(q1) == symm(q2) "QSpace objects must share the same symmetry tuple"

    q1_ext = addSingleton(q1, QD1 + 1; dir='-')
    q2_ext = addSingleton(q2, 1; dir='+')
    return contract(q1_ext, (QD1 + 1,), q2_ext, (1,))
end

LinearAlgebra.kron(q1::QSpace, q2::QSpace) = q1 ⊗ q2

include("getLocalSpace.jl")
include("getIdentity.jl")
include("contract.jl")
include("get1jtensor.jl")
include("svd.jl")
include("eig.jl")
include("permute.jl")


