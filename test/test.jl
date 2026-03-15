using LinearAlgebra
using Random
using Test
using LurCGT
using QSpaces
import QSpaces: _compute_spaces

include("test_utils.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: reconstruct original array from (U, SV) returned by svd_leg.
#
# U   : (dim_leg, χ)
# SV  : same shape as original array, but with the `leg`-th dim replaced by χ
#
# Reconstruction contracts U[i, χ] with SV[..., χ, ...] over χ,
# placing i back into position `leg`.
# ─────────────────────────────────────────────────────────────────────────────
function reconstruct(U::AbstractArray, SV::AbstractArray, leg::Integer)
    N    = ndims(SV)
    chi  = size(SV, leg)

    # Move χ axis of SV to the front  →  (χ, other dims...)
    other_legs   = [i for i in 1:N if i != leg]
    perm_to_front = (leg, other_legs...)
    SV_front     = permutedims(SV, perm_to_front)

    # Matrix multiply:  U (dim_leg × χ)  *  SV_front_mat (χ × rest)
    rec_mat = U * reshape(SV_front, chi, :)

    # Reshape and permute back to original leg order
    rec_shape = (size(U, 1), [size(SV, i) for i in other_legs]...)
    rec_perm  = invperm(collect(perm_to_front))
    return permutedims(reshape(rec_mat, rec_shape...), rec_perm)
end

# ─────────────────────────────────────────────────────────────────────────────

@testset "lock reduction in contract" begin
    test_lock_reduce(FermionSOptions(U1, SU{2}, nothing, 1))
end

@testset "auto contract requires matching spaces" begin
    test_contract_requires_matching_spaces_in_star(
        FermionSOptions(U1, SU{2}, nothing, 1))
end

@testset "contract_v2 vs contract" begin
    test_contract_v2(FermionSOptions(U1, SU{2}, nothing, 1))
    test_contract_v2(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "spaces of svdQS" begin
    test_spaces_svdQS(FermionSOptions(U1, SU{2}, nothing, 1))
    test_spaces_svdQS(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "svd truncation of QSpace" begin
    test_truncate_svdQS(FermionSOptions(U1, SU{2}, nothing, 1))
    test_truncate_svdQS(FermionSOptions(U1, SU{2}, SU{3}, 3))
end


@testset "spaces of eigQS" begin
    test_spaces_eigQS(FermionSOptions(U1, SU{2}, nothing, 1))
    test_spaces_eigQS(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "missing spaces of eigQS" begin
    test_missing_spaces_eigQS(FermionSOptions(U1, SU{2}, nothing, 1))
    test_missing_spaces_eigQS(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "truncate missing zero spaces of eigQS" begin
    test_truncate_missing_zero_spaces_eigQS(FermionSOptions(U1, SU{2}, nothing, 1))
    test_truncate_missing_zero_spaces_eigQS(FermionSOptions(U1, SU{2}, SU{3}, 3))
end


@testset "conjugation of QSpace test" begin
    test_conj(FermionSOptions(U1, SU{2}, nothing, 1))
    test_conj(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "norm of QSpace" begin
    test_norm(FermionSOptions(U1, SU{2}, nothing, 1))
    test_norm(FermionSOptions(U1, SU{2}, SU{3}, 3))
end


@testset "svd test" begin
    option = FermionSOptions(U1, SU{2}, SU{3}, 3)
    q   = getLocalSpace(option)
    qi1 = QSpace(q.I, ("lur1", "lur1"))
    qi2 = QSpace(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2))
    qf  = QSpace(q.F, ("lur2", "lur2", "op"))
    ct  = qf * a
    test_svdQS(ct, [2, 4])
    test_svdQS(ct, [1, 4])
    test_svdQS(ct, [1, 2])
end

@testset "eig of QSpace" begin
    test_eigQS(FermionSOptions(U1, SU{2}, nothing, 1))
    test_eigQS(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "eig truncation of QSpace" begin
    test_truncate_eigQS(FermionSOptions(U1, SU{2}, nothing, 1))
    test_truncate_eigQS(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "eig full discard of QSpace" begin
    test_eigQS_full_discard(FermionSOptions(U1, SU{2}, nothing, 1))
    test_eigQS_full_discard(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "eig discard tags of QSpace" begin
    test_discard_eigQS_tags(FermionSOptions(U1, SU{2}, nothing, 1))
    test_discard_eigQS_tags(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

function permtest()
    S = SU{2}
    upsp = ((1,), (2,), (2,), (2,), (3,), (3,))
    dnsp = ((1,), (2,), (4,))

    tags = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
    dirs = ('-', '+', '-', '+', '-', '-', '-', '+', '-')
    inds = Tuple(QIndex(tags[i], dirs[i]) for i=1:9)

    om = get_CGTom(S, upsp, dnsp).totalOM
    wmat = randn(om, 1); wmat /= norm(wmat)
    wmat = QTensor(wmat)
    RMT = QTensor(reshape([1.0], Tuple(1 for _=1:10)...))

    qlabels = (upsp..., dnsp...)
    cgp = (5, 8, 2, 7, 1, 6, 3, 9, 4)
    cgr = CGR(S, qlabels, wmat, cgp, (6, 3))
    rows = [row((cgr,), RMT)]
    
    # Build spaces: for each physical leg, get its qlabel via cgp mapping
    # RMT dim = 1 for all legs, qlabel wrapped in tuple for N=1 symmetry
    # Use _compute_spaces helper instead of manual construction
    spaces = _compute_spaces(rows)
    
    q = QSpace((S,), rows, inds, spaces)
    println("q created")
    pq = permuteQS(q, (1, 2, 4, 8, 5, 6, 9, 3, 7))
    println("pq created")

    qarr = to_sparse_array(q)
    pqarr1 = permutedims(qarr, (1, 2, 4, 8, 5, 6, 9, 3, 7))
    pqarr2 = to_sparse_array(pq)

    println(norm(pqarr1 - pqarr2))
    return pqarr1, pqarr2
end

arr1, arr2 = permtest()
println("asdfadsf")

@testset "svd_leg" begin

    # ------------------------------------------------------------------
    # 1. Shape check and exact reconstruction on a random 3-D array
    # ------------------------------------------------------------------
    @testset "shape and reconstruction (random 3-D, each leg)" begin
        A = randn(3, 4, 5)

        for leg in 1:3
            U, SV, S = svd_leg(A, leg; cutoff=1e-12)
            chi = length(S)

            # SV must have same ndims as A, with dim `leg` replaced by χ
            expected_size = ntuple(i -> i == leg ? chi : size(A, i), 3)
            @test size(SV) == expected_size
            @test size(U)  == (size(A, leg), chi)

            rec = reconstruct(U, SV, leg)
            @test size(rec) == size(A)
            @test norm(A - rec) < 1e-9
        end
    end

    # ------------------------------------------------------------------
    # 2. Low-rank array: SVD should recover rank exactly, truncating zeros
    # ------------------------------------------------------------------
    @testset "low-rank truncation" begin
        # Build a rank-2 matrix embedded in 3-D: A[i,j,k] = u1[i]*v1[j]*w1[k]
        #                                                   + u2[i]*v2[j]*w2[k]
        u1, u2 = randn(4), randn(4)
        v1, v2 = randn(3), randn(3)
        w1, w2 = randn(5), randn(5)
        A = reshape(u1, 4, 1, 1) .* reshape(v1, 1, 3, 1) .* reshape(w1, 1, 1, 5) .+
            reshape(u2, 4, 1, 1) .* reshape(v2, 1, 3, 1) .* reshape(w2, 1, 1, 5)

        # Decompose along leg 1 (rank ≤ 2 after flattening other legs)
        U, SV, S = svd_leg(A, 1; cutoff=1e-10)
        @test length(S) == 2          # exactly 2 singular values survive
        rec = reconstruct(U, SV, 1)
        @test norm(A - rec) < 1e-9
    end

    # ------------------------------------------------------------------
    # 3. maxdim keyword caps the number of singular values kept
    # ------------------------------------------------------------------
    @testset "maxdim truncation" begin
        A = randn(6, 5, 4)

        # Without maxdim, all singular values above cutoff are kept
        _, _, S_full = svd_leg(A, 2; cutoff=1e-12)

        # With maxdim=3, at most 3 are kept
        U3, SV3, S3 = svd_leg(A, 2; cutoff=1e-12, maxdim=3)
        @test length(S3) <= 3
        @test length(S3) <= length(S_full)

        # The 3 kept values should be the largest ones
        @test S3 ≈ S_full[1:length(S3)]
    end

    # ------------------------------------------------------------------
    # 4. QTensor overload returns QTensor wrappers with correct shapes
    # ------------------------------------------------------------------
    @testset "QTensor overload" begin
        A  = randn(3, 4, 5)
        qt = QTensor(A)

        for leg in 1:3
            Uq, SVq, Sq = svd_leg(qt, leg; cutoff=1e-12)
            U,  SV,  S  = svd_leg(A,  leg; cutoff=1e-12)

            @test Uq  isa QTensor
            @test SVq isa QTensor
            @test size(Uq)  == size(U)
            @test size(SVq) == size(SV)
            @test Sq ≈ S

            rec = reconstruct(Uq.data, SVq.data, leg)
            @test norm(A - rec) < 1e-9
        end
    end

    # ------------------------------------------------------------------
    # 5. 2-D degenerate case: standard matrix SVD
    # ------------------------------------------------------------------
    @testset "2-D matrix (standard SVD)" begin
        M = randn(5, 4)

        # leg=1: U is (5,χ), SV is (5,4) with dim 1 replaced by χ → (χ,4)
        # leg=2: U is (4,χ), SV is (5,χ)
        for leg in 1:2
            U, SV, S = svd_leg(M, leg; cutoff=1e-12)
            rec = reconstruct(U, SV, leg)
            @test norm(M - rec) < 1e-9
        end
    end

end

@testset "cgt_metadata test" begin
    ql1 = ((6,), (7,), (2,), (3,))
    ql2 = ((2,), (3,), (4,), (7,), (8,))
    ld1 = (2, 2); ld2 = (3, 2)
    cp1 = (4, 1, 3, 2); cp2 = (3, 5, 1, 2, 4)
    free1 = [1, 2]; legs1 = (3, 4)
    free2 = [1, 2, 4]; legs2 = (3, 5)

    new_qlabels, new_cgp, new_legdir = get_new_cgp(
        ql1, ld1, cp1, free1, legs1,
        ql2, ld2, cp2, free2, legs2)

    @test new_qlabels == ((3,), (4,), (6,), (3,), (8,))
    @test new_cgp == (4, 3, 2, 5, 1)
    @test new_legdir == (3, 2)

    ql1 = ((4,), (7,), (2,), (3,))
    new_qlabels, new_cgp, new_legdir = get_new_cgp(
        ql1, ld1, cp1, free1, legs1,
        ql2, ld2, cp2, free2, legs2)

    @test new_qlabels == ((3,), (4,), (4,), (3,), (8,))
    @test new_cgp == (4, 2, 3, 5, 1)
    @test new_legdir == (3, 2)
end

@testset "compress_sector test" begin
    test_compress_sector(2, 7, 3; verbose=false)
    test_compress_sector(3, 5, 4; verbose=false)
end


function lur(option::LocalSpaceOptions)
    option = FermionSOptions(U1, SU{2}, nothing, 1)
    q, a, ct, arr1, arr2 = test_FAcont(option)
    println(norm(arr1 - arr2))
    @test arr1 ≈ arr2
end


@testset "Generating 1jpair of QSpace test" begin
    test_1jpair(FermionSOptions(U1, SU{2}, nothing, 1))
    test_1jpair(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

@testset "contraction of QSpace test" begin
    test_FAcont(FermionSOptions(U1, SU{2}, nothing, 1))
    test_FAcont(FermionSOptions(U1, SU{2}, SU{3}, 3))
end

function example()
    opt = FermionSOptions(U1, SU{2}, nothing, 1)
    q = getLocalSpace(opt, ("lur", "lur", "op"))
    nloc = lock(q.F', 2) * q.F
    return nloc
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a 3-leg test QSpace with known QIndex properties.
#
#   leg 1: dir='+', itags="site1", plev=0, lock=0
#   leg 2: dir='-', itags="site2", plev=0, lock=0
#   leg 3: dir='-', itags="op",    plev=0, lock=0
#
# Built from a real operator QSpace so the internal row data is valid.
# ─────────────────────────────────────────────────────────────────────────────
function _make_test_qspace()
    option = FermionSOptions(U1, SU{2}, SU{3}, 3)
    q0 = getLocalSpace(option)
    # q0.F is a 3-leg QSpace: dirs=('+','-','-'), all plev=0, all lock=0.
    # QSpace(q, tags) creates a copy with new tags (only the itags field changes).
    return QSpace(q0.F, ("site1", "site2", "op"))
end

@testset "QIndex modifier functions" begin
    q = _make_test_qspace()
    # Fixture legs:
    #   leg 1: dir='+', itags="site1", plev=0, lock=0
    #   leg 2: dir='-', itags="site2", plev=0, lock=0
    #   leg 3: dir='-', itags="op",    plev=0, lock=0

    # ── findlegs ──────────────────────────────────────────────────────────────
    @testset "findlegs" begin
        @test findlegs(q; dir='+')       == [1]
        @test findlegs(q; dir='-')       == [2, 3]
        @test findlegs(q; itags="site1") == [1]
        @test findlegs(q; itags="site2") == [2]
        @test findlegs(q; itags="op")    == [3]
        @test findlegs(q; plev=0)        == [1, 2, 3]
        @test findlegs(q; lock=0)        == [1, 2, 3]
        # with non-zero plev/lock set by a modifier
        q_p = prime(q, 2)
        @test findlegs(q_p; plev=1) == [2]
        @test findlegs(q_p; plev=0) == [1, 3]
        q_k = lock(q, 3)
        @test findlegs(q_k; lock=0) == [1, 2]
        @test findlegs(q_k; lock=1) == [3]
        # rev (complement selection)
        @test findlegs(q; dir='+',      rev=true) == [2, 3]
        @test findlegs(q; dir='-',      rev=true) == [1]
        @test findlegs(q; itags="site1", rev=true) == [2, 3]
        @test findlegs(q; plev=0,        rev=true) == []
        # multi-criteria (AND logic)
        @test findlegs(q; dir='-', itags="op")           == [3]
        @test findlegs(q; dir='-', itags="op", rev=true) == [1, 2]
    end

    # ── findleg ───────────────────────────────────────────────────────────────
    @testset "findleg" begin
        @test findleg(q; dir='+')             == 1
        @test findleg(q; dir='-')             == 2   # first match
        @test findleg(q; itags="op")          == 3
        @test findleg(q; plev=0)              == 1   # first of all legs
        @test findleg(q; dir='+', rev=true)   == 2   # first non-'+' leg
        @test findleg(q; plev=0, rev=true)   === nothing  # no leg with plev≠0
        @test findleg(q; itags="nope")        === nothing
    end

    # ── lock (leg / LegList forms) ────────────────────────────────────────────
    @testset "lock – leg and LegList" begin
        q2 = lock(q, 1)
        @test q2.inds[1].lock == 1
        @test q2.inds[2].lock == 0
        @test q2.inds[3].lock == 0

        q2 = lock(q, 3; inc=2)
        @test q2.inds[3].lock == 2
        @test q2.inds[1].lock == 0

        # LegList: vector
        q2 = lock(q, [1, 3])
        @test q2.inds[1].lock == 1
        @test q2.inds[3].lock == 1
        @test q2.inds[2].lock == 0

        # LegList: tuple
        q2 = lock(q, (2, 3))
        @test q2.inds[2].lock == 1
        @test q2.inds[3].lock == 1
        @test q2.inds[1].lock == 0
    end

    # ── lock (criteria form) ──────────────────────────────────────────────────
    @testset "lock – criteria" begin
        q2 = lock(q; dir='+')      # leg 1 only
        @test q2.inds[1].lock == 1
        @test q2.inds[2].lock == 0
        @test q2.inds[3].lock == 0

        q2 = lock(q; inc=3, itags="op")
        @test q2.inds[3].lock == 3
        @test q2.inds[1].lock == 0

        # rev: legs 2 and 3
        q2 = lock(q; dir='+', rev=true)
        @test q2.inds[1].lock == 0
        @test q2.inds[2].lock == 1
        @test q2.inds[3].lock == 1

        # permanent lock (-1) is never incremented further
        q_perm = lockp(q, 1)
        q2 = lock(q_perm, 1; inc=5)
        @test q2.inds[1].lock == -1
    end

    # ── lockp ─────────────────────────────────────────────────────────────────
    @testset "lockp" begin
        q2 = lockp(q, 2)
        @test q2.inds[2].lock == -1
        @test q2.inds[1].lock == 0

        # LegList: vector
        q2 = lockp(q, [1, 3])
        @test q2.inds[1].lock == -1
        @test q2.inds[3].lock == -1
        @test q2.inds[2].lock == 0

        # criteria form
        q2 = lockp(q; itags="site2")
        @test q2.inds[2].lock == -1
        @test q2.inds[1].lock == 0

        # rev: legs 2 and 3
        q2 = lockp(q; dir='+', rev=true)
        @test q2.inds[2].lock == -1
        @test q2.inds[3].lock == -1
        @test q2.inds[1].lock == 0
    end

    # ── unlock ────────────────────────────────────────────────────────────────
    @testset "unlock" begin
        q_locked = lock(q, [1, 2])
        q2 = unlock(q_locked, 1)
        @test q2.inds[1].lock == 0
        @test q2.inds[2].lock == 1   # unchanged

        # unlock also removes permanent lock
        q_perm = lockp(q, 3)
        q2 = unlock(q_perm, 3)
        @test q2.inds[3].lock == 0

        # criteria form: legs 2 and 3
        q_locked = lock(q, [2, 3])
        q2 = unlock(q_locked; dir='-')
        @test q2.inds[2].lock == 0
        @test q2.inds[3].lock == 0
        @test q2.inds[1].lock == 0   # was already 0

        # rev: only leg 1
        q2 = unlock(q_locked; dir='-', rev=true)
        @test q2.inds[1].lock == 0
        @test q2.inds[2].lock == 1   # still locked
        @test q2.inds[3].lock == 1
    end

    # ── prime (leg / LegList forms) ───────────────────────────────────────────
    @testset "prime – leg and LegList" begin
        q2 = prime(q, 1)
        @test q2.inds[1].plev == 1
        @test q2.inds[2].plev == 0

        q2 = prime(q, 2; inc=3)
        @test q2.inds[2].plev == 3
        @test q2.inds[1].plev == 0

        # LegList: vector
        q2 = prime(q, [1, 3])
        @test q2.inds[1].plev == 1
        @test q2.inds[3].plev == 1
        @test q2.inds[2].plev == 0

        # LegList: tuple
        q2 = prime(q, (1, 2); inc=2)
        @test q2.inds[1].plev == 2
        @test q2.inds[2].plev == 2
        @test q2.inds[3].plev == 0
    end

    # ── prime (criteria form) ─────────────────────────────────────────────────
    @testset "prime – criteria" begin
        q2 = prime(q)           # all legs
        @test all(q2.inds[i].plev == 1 for i in 1:3)

        q2 = prime(q; inc=2)
        @test all(q2.inds[i].plev == 2 for i in 1:3)

        # clamping: negative inc on all-zero plev stays at 0
        q2 = prime(q; inc=-5)
        @test all(q2.inds[i].plev == 0 for i in 1:3)

        q2 = prime(q; dir='+')  # leg 1 only
        @test q2.inds[1].plev == 1
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0

        # rev: legs 2, 3
        q2 = prime(q; dir='+', rev=true)
        @test q2.inds[1].plev == 0
        @test q2.inds[2].plev == 1
        @test q2.inds[3].plev == 1
    end

    # ── setprime ──────────────────────────────────────────────────────────────
    @testset "setprime" begin
        # LegList: vector (legs 1,2 differ in dir → unique QIndex)
        q2 = setprime(q, [1, 2], 7)
        @test q2.inds[1].plev == 7
        @test q2.inds[2].plev == 7
        @test q2.inds[3].plev == 0

        # LegList: tuple (legs 1,3 differ in dir → unique QIndex)
        q2 = setprime(q, (1, 3), 5)
        @test q2.inds[1].plev == 5
        @test q2.inds[3].plev == 5
        @test q2.inds[2].plev == 0

        # criteria form: outgoing legs (2 and 3 differ in itags → unique)
        q2 = setprime(q, 3; dir='-')
        @test q2.inds[2].plev == 3
        @test q2.inds[3].plev == 3
        @test q2.inds[1].plev == 0

        # rev: incoming only (leg 1)
        q2 = setprime(q, 3; dir='-', rev=true)
        @test q2.inds[1].plev == 3
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0

        @test_throws ArgumentError setprime(q, -1)
    end

    # ── noprime ───────────────────────────────────────────────────────────────
    @testset "noprime" begin
        q_primed = prime(q)   # all plev=1

        q2 = noprime(q_primed)
        @test all(q2.inds[i].plev == 0 for i in 1:3)

        q2 = noprime(q_primed, 2)
        @test q2.inds[2].plev == 0
        @test q2.inds[1].plev == 1
        @test q2.inds[3].plev == 1

        # LegList: vector
        q2 = noprime(q_primed, [1, 3])
        @test q2.inds[1].plev == 0
        @test q2.inds[3].plev == 0
        @test q2.inds[2].plev == 1

        # criteria form: leg 1 only
        q2 = noprime(q_primed; dir='+')
        @test q2.inds[1].plev == 0
        @test q2.inds[2].plev == 1
        @test q2.inds[3].plev == 1

        # rev: legs 2, 3
        q2 = noprime(q_primed; dir='+', rev=true)
        @test q2.inds[1].plev == 1
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0
    end

    # ── addtags ───────────────────────────────────────────────────────────────
    @testset "addtags" begin
        # criteria form: all legs (tags sorted alphabetically)
        q2 = addtags(q, "new")
        @test q2.inds[1].itags == "new,site1"   # site1 + new → sorted: new,site1
        @test q2.inds[2].itags == "new,site2"   # site2 + new → sorted: new,site2
        @test q2.inds[3].itags == "new,op"      # op    + new → sorted: new,op

        # single leg
        q2 = addtags(q, 1, "u1")
        @test q2.inds[1].itags == "site1,u1"
        @test q2.inds[2].itags == "site2"   # unchanged

        # LegList: vector
        q2 = addtags(q, [2, 3], "phys")
        @test q2.inds[2].itags == "phys,site2"
        @test q2.inds[3].itags == "op,phys"
        @test q2.inds[1].itags == "site1"

        # LegList: tuple
        q2 = addtags(q, (1, 3), "x")
        @test q2.inds[1].itags == "site1,x"
        @test q2.inds[3].itags == "op,x"
        @test q2.inds[2].itags == "site2"

        # criteria with selector: leg 1 only
        q2 = addtags(q, "u1"; dir='+')
        @test q2.inds[1].itags == "site1,u1"
        @test q2.inds[2].itags == "site2"   # unchanged

        # rev: legs 2, 3
        q2 = addtags(q, "u1"; dir='+', rev=true)
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "site2,u1"
        @test q2.inds[3].itags == "op,u1"
    end

    # ── removetags ────────────────────────────────────────────────────────────
    @testset "removetags" begin
        # criteria form: only leg 1 has "site1"
        q2 = removetags(q, "site1")
        @test q2.inds[1].itags == ""       # "site1" removed → empty
        @test q2.inds[2].itags == "site2"  # unchanged (no "site1")
        @test q2.inds[3].itags == "op"     # unchanged

        # single leg
        q2 = removetags(q, 2, "site2")
        @test q2.inds[2].itags == ""
        @test q2.inds[1].itags == "site1"

        # LegList: vector
        q_extra = addtags(q, "extra")
        q2 = removetags(q_extra, [1, 3], "extra")
        @test q2.inds[1].itags == "site1"
        @test q2.inds[3].itags == "op"
        @test q2.inds[2].itags == "extra,site2"   # unchanged

        # LegList: tuple
        q2 = removetags(q_extra, (2, 3), "extra")
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
        @test q2.inds[1].itags == "extra,site1"   # unchanged

        # criteria with selector: leg 1
        q2 = removetags(q_extra, "extra"; dir='+')
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "extra,site2"   # unchanged

        # rev: legs 2, 3
        q2 = removetags(q_extra, "extra"; dir='+', rev=true)
        @test q2.inds[1].itags == "extra,site1"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
    end

    # ── replacetags ───────────────────────────────────────────────────────────
    @testset "replacetags" begin
        # criteria form (all legs): removes "site1", adds "link" to EVERY selected leg,
        # regardless of whether "site1" was present.
        q2 = replacetags(q, "site1", "link")
        @test q2.inds[1].itags == "link"          # "site1" removed, "link" added
        @test q2.inds[2].itags == "link,site2"    # "site1" absent; "link" added to "site2"
        @test q2.inds[3].itags == "link,op"       # "site1" absent; "link" added to "op"

        # single leg: clean targeted replacement
        q2 = replacetags(q, 1, "site1", "link")
        @test q2.inds[1].itags == "link"
        @test q2.inds[2].itags == "site2"   # unchanged

        # LegList: vector
        q2 = replacetags(q, [1, 2], "site1", "link")
        @test q2.inds[1].itags == "link"
        @test q2.inds[2].itags == "link,site2"   # "site1" absent; "link" added
        @test q2.inds[3].itags == "op"            # unchanged

        # use itags selector to restrict to legs that actually carry the old tag
        q2 = replacetags(q, "site1", "link"; itags="site1")
        @test q2.inds[1].itags == "link"
        @test q2.inds[2].itags == "site2"   # skipped (no "site1")
        @test q2.inds[3].itags == "op"      # skipped

        # rev: applies only to leg 1 (not dir='-' → leg 1 only)
        q2 = replacetags(q, "site1", "link"; dir='-', rev=true)
        @test q2.inds[1].itags == "link"    # "site1" removed, "link" added
        @test q2.inds[2].itags == "site2"   # excluded by rev
        @test q2.inds[3].itags == "op"      # excluded by rev
    end

    # ── settags ───────────────────────────────────────────────────────────────
    @testset "settags" begin
        # single leg
        q2 = settags(q, 3, "phys")
        @test q2.inds[3].itags == "phys"
        @test q2.inds[1].itags == "site1"

        # LegList: vector (legs 1,2 differ in dir → unique QIndex)
        q2 = settags(q, [1, 2], "lur")
        @test q2.inds[1].itags == "lur"
        @test q2.inds[2].itags == "lur"
        @test q2.inds[3].itags == "op"

        # LegList: tuple (legs 1,3 differ in dir → unique QIndex)
        q2 = settags(q, (1, 3), "x")
        @test q2.inds[1].itags == "x"
        @test q2.inds[3].itags == "x"
        @test q2.inds[2].itags == "site2"

        # criteria: only leg 1 (dir='+')
        q2 = settags(q, "phys"; dir='+')
        @test q2.inds[1].itags == "phys"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"

        # rev: only leg 1 (not dir='-' means not legs 2,3)
        q2 = settags(q, "phys"; dir='-', rev=true)
        @test q2.inds[1].itags == "phys"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
    end

    # ── orthogonality: only the targeted leg changes ──────────────────────────
    @testset "non-targeted legs are unmodified" begin
        q2 = prime(q, 1)
        for i in 2:3
            @test q2.inds[i].plev  == q.inds[i].plev
            @test q2.inds[i].lock  == q.inds[i].lock
            @test q2.inds[i].itags == q.inds[i].itags
            @test q2.inds[i].dir   == q.inds[i].dir
        end

        q2 = lock(q, 1)
        for i in 2:3
            @test q2.inds[i].lock  == q.inds[i].lock
            @test q2.inds[i].plev  == q.inds[i].plev
            @test q2.inds[i].itags == q.inds[i].itags
        end

        q2 = addtags(q, 1, "extra")
        for i in 2:3
            @test q2.inds[i].itags == q.inds[i].itags
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests for empty_qspace
# ─────────────────────────────────────────────────────────────────────────────

@testset "empty_qspace" begin
    # ── rank-1 to rank-5 construction ────────────────────────────────────────
    @testset "rank-$QD construction" for QD in 1:5
        symm = (SU{2},)
        inds = ntuple(i -> QIndex("l$i", i == 1 ? '+' : '-'), QD)
        q = empty_qspace(symm, inds)

        @test length(q.rows)  == 0
        @test length(q.inds)  == QD
        @test q.symm          == symm
        @test q.inds          == inds
        @test all(isempty(q.spaces[l]) for l in 1:QD)
    end

    # ── multiple symmetries ───────────────────────────────────────────────────
    @testset "multi-symmetry construction" begin
        symm = (U1, SU{2})
        inds = (QIndex("a", '+'), QIndex("b", '-'), QIndex("c", '-'))
        q = empty_qspace(symm, inds)

        @test length(q.symm) == 2
        @test length(q.rows) == 0
        @test length(q.inds) == 3
        @test q.symm == symm
        # spaces: 3 empty vectors, one per leg
        @test length(q.spaces) == 3
        @test all(isempty(q.spaces[l]) for l in 1:3)
    end

    # ── element type keyword ──────────────────────────────────────────────────
    @testset "element type" begin
        symm = (SU{2},)
        inds = (QIndex("a", '+'), QIndex("b", '-'))
        qf64 = empty_qspace(symm, inds; T=Float64)
        qc64 = empty_qspace(symm, inds; T=ComplexF64)

        # N=1, QD=2, RD=3
        @test qf64.rows isa Vector{row{Float64,     2, 1, 3}}
        @test qc64.rows isa Vector{row{ComplexF64,  2, 1, 3}}
    end

    # ── QIndex modifier operations on an empty QSpace ─────────────────────────
    @testset "modifier ops on empty QSpace" begin
        symm = (SU{2}, U1)
        inds = (QIndex("a", '+'), QIndex("b", '-'), QIndex("c", '-'))
        q = empty_qspace(symm, inds)

        # prime
        @test prime(q).inds[1].plev          == 1
        @test prime(q, 2).inds[2].plev        == 1
        @test prime(q, 2).inds[1].plev        == 0
        @test prime(q; dir='+').inds[1].plev  == 1
        @test prime(q; dir='+').inds[2].plev  == 0

        # lock / unlock / lockp
        ql = lock(q, 1)
        @test ql.inds[1].lock == 1
        @test ql.inds[2].lock == 0
        @test unlock(ql, 1).inds[1].lock == 0
        @test lockp(q, 2).inds[2].lock   == -1

        # tag operations
        @test addtags(q, "x").inds[1].itags          == "a,x"
        @test removetags(q, "a").inds[1].itags        == ""
        @test settags(q, "new"; dir='+').inds[1].itags == "new"
        @test replacetags(q, "a", "z").inds[1].itags  == "z"

        # findlegs / findleg
        @test findlegs(q; dir='+') == [1]
        @test findlegs(q; dir='-') == [2, 3]
        @test findleg(q; dir='+')  == 1
        @test findleg(q; dir='-')  == 2

        # scalar multiplication of empty QSpace produces empty QSpace
        q_scaled = 3.0 * q
        @test length(q_scaled.rows) == 0
    end

    # ── show does not error ───────────────────────────────────────────────────
    @testset "show on empty QSpace" begin
        symm = (SU{2},)
        inds = (QIndex("a", '+'), QIndex("b", '-'))
        q = empty_qspace(symm, inds)
        buf = IOBuffer()
        # must not throw
        @test (show(buf, MIME"text/plain"(), q); true)
        # output should mention "(empty)"
        @test occursin("empty", String(take!(buf)))
    end
end

@testset "getvac" begin
    @testset "single trivial sector with default tags" begin
        option = FermionSOptions(U1, SU{2}, nothing, 1)
        q0 = getLocalSpace(option)
        vac = getvac(q0.I)

        @test vac.symm == q0.I.symm
        @test length(vac.rows) == 1
        @test vac.inds == (QIndex("", '+'), QIndex("", '-'))
        @test length(vac.spaces[1]) == 1
        @test length(vac.spaces[2]) == 1

        trivial = ntuple(n -> Tuple(0 for _ in 1:nzops(vac.symm[n])), length(vac.symm))
        @test vac.spaces[1][1] == (1, trivial)
        @test vac.spaces[2][1] == (1, trivial)

        r = vac.rows[1]
        @test size(r.RMT.data) == ntuple(_ -> 1, length(vac.symm) + 2)
        @test only(r.RMT.data) == one(eltype(r.RMT.data))
        for n in 1:length(vac.symm)
            @test r.cgrs[n].qlabels == (trivial[n], trivial[n])
            @test r.cgrs[n].cgp == (1, 2)
            @test r.cgrs[n].legdir == (1, 1)
            @test size(r.cgrs[n].wmat.data) == (1, 1)
            @test r.cgrs[n].wmat[1] == 1.0
        end
    end

    @testset "optional tags are applied" begin
        option = FermionSOptions(U1, SU{2}, nothing, 1)
        q0 = getLocalSpace(option)
        vac = getvac(q0.F, ("vin", "vout"))

        @test vac.inds[1] == QIndex("vin", '+')
        @test vac.inds[2] == QIndex("vout", '-')
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a rank-4 test QSpace from getIdentity with 3 input legs.
#
#   leg 1 ('+', "l1"), leg 2 ('+', "l2"), leg 3 ('+', "l3"),
#   leg 4 ('-', "fused")
# ─────────────────────────────────────────────────────────────────────────────
function _make_test_qspace_rank4()
    option = FermionSOptions(U1, SU{2}, nothing, 1)
    q0  = getLocalSpace(option)
    qi1 = QSpace(q0.I, ("b1a", "b1b"))
    qi2 = QSpace(q0.I, ("b2a", "b2b"))
    qi3 = QSpace(q0.I, ("b3a", "b3b"))
    a4  = getIdentity((qi1, 2), (qi2, 2), (qi3, 2); itags="fused")
    return QSpace(a4, ("l1", "l2", "l3", "fused"))
end


@testset "scalar add/subtract on rank-2 QSpace" begin
    option = FermionSOptions(U1, SU{2}, nothing, 1)
    q0 = getLocalSpace(option)
    q = QSpace(q0.I, ("left", "right"))

    idq = getIdentity((q, 2); itags=q.inds[2].itags)
    idq = QSpace(idq, q.inds)

    @test norm((q + 2.5) - (q + 2.5 * idq)) < 1e-10
    @test norm((q - 2.5) - (q - 2.5 * idq)) < 1e-10
    @test norm((2.5 + q) - (q + 2.5 * idq)) < 1e-10
    @test norm((2.5 - q) - (2.5 * idq - q)) < 1e-10

    q_bad_rank = _make_test_qspace_rank4()
    @test_throws AssertionError q_bad_rank + 1.0

    q_bad_dirs = getIdentity((q, 1), (q, 2); itags="fused")
    @test_throws AssertionError q_bad_dirs + 1.0
end


# ─────────────────────────────────────────────────────────────────────────────
# Tests for arbitrary-rank (rank ≥ 4) QSpace
# ─────────────────────────────────────────────────────────────────────────────

@testset "arbitrary rank QSpace" begin
    q4 = _make_test_qspace_rank4()

    # ── basic structure ───────────────────────────────────────────────────────
    @testset "rank-4 structure" begin
        @test length(q4.inds)  == 4
        @test length(q4.spaces) == 4
        @test q4.inds[1].dir   == '+'
        @test q4.inds[2].dir   == '+'
        @test q4.inds[3].dir   == '+'
        @test q4.inds[4].dir   == '-'
        @test q4.inds[1].itags == "l1"
        @test q4.inds[4].itags == "fused"
        @test length(q4.rows)  > 0
        @test all(!isempty(q4.spaces[l]) for l in 1:4)
    end

    # ── findlegs / findleg ────────────────────────────────────────────────────
    @testset "findlegs on rank-4" begin
        @test findlegs(q4; dir='+')       == [1, 2, 3]
        @test findlegs(q4; dir='-')        == [4]
        @test findlegs(q4; itags="l1")     == [1]
        @test findlegs(q4; itags="fused")  == [4]
        @test findleg(q4; dir='+')         == 1
        @test findleg(q4; dir='-')         == 4
        @test findleg(q4; itags="fused")   == 4
        @test findlegs(q4; dir='+', rev=true) == [4]
    end

    # ── prime on rank-4 ───────────────────────────────────────────────────────
    @testset "prime on rank-4" begin
        q_p_all = prime(q4)
        @test all(q_p_all.inds[i].plev == 1 for i in 1:4)

        q_p3 = prime(q4, 3)
        @test q_p3.inds[3].plev == 1
        @test q_p3.inds[1].plev == 0
        @test q_p3.inds[2].plev == 0
        @test q_p3.inds[4].plev == 0

        q_p_in = prime(q4; dir='+')
        @test all(q_p_in.inds[i].plev == 1 for i in 1:3)
        @test q_p_in.inds[4].plev == 0
    end

    # ── lock on rank-4 ────────────────────────────────────────────────────────
    @testset "lock on rank-4" begin
        q_lk = lock(q4, [1, 3])
        @test q_lk.inds[1].lock == 1
        @test q_lk.inds[2].lock == 0
        @test q_lk.inds[3].lock == 1
        @test q_lk.inds[4].lock == 0

        q_lkp = lockp(q4, [2, 4])
        @test q_lkp.inds[2].lock == -1
        @test q_lkp.inds[4].lock == -1
        @test q_lkp.inds[1].lock == 0
    end

    # ── tag ops on rank-4 ─────────────────────────────────────────────────────
    @testset "tag ops on rank-4" begin
        q_a = addtags(q4, "phys"; dir='+')
        @test all(occursin("phys", q_a.inds[i].itags) for i in 1:3)
        @test !occursin("phys", q_a.inds[4].itags)

        q_r = removetags(q_a, "phys"; dir='+')
        @test all(q_r.inds[i].itags == q4.inds[i].itags for i in 1:4)

        q_s = settags(q4, "new"; dir='-')
        @test q_s.inds[4].itags == "new"
        @test q_s.inds[1].itags == "l1"  # unchanged
    end

    # ── scalar multiplication ─────────────────────────────────────────────────
    @testset "scalar multiplication on rank-4" begin
        q_scaled = 2.5 * q4
        @test length(q_scaled.rows) == length(q4.rows)
        arr_orig   = Array(to_sparse_array(q4))
        arr_scaled = Array(to_sparse_array(q_scaled))
        @test norm(arr_scaled - 2.5 .* arr_orig) < 1e-10
    end

    # ── addition: q4 + q4 ≈ 2*q4 ─────────────────────────────────────────────
    @testset "addition on rank-4: q + q ≈ 2*q" begin
        q_sum    = q4 + q4
        q_double = 2.0 * q4
        arr_sum    = Array(to_sparse_array(q_sum))
        arr_double = Array(to_sparse_array(q_double))
        @test norm(arr_sum - arr_double) < 1e-10
    end

    # ── conj: conj(conj(q)) ≈ q ───────────────────────────────────────────────
    @testset "double-conj roundtrip on rank-4" begin
        qc  = conj(q4)
        # all leg directions must flip
        @test all(qc.inds[i].dir != q4.inds[i].dir for i in 1:4)
        qcc = conj(qc)
        # directions restored
        @test all(qcc.inds[i].dir == q4.inds[i].dir for i in 1:4)
        # tensor values preserved
        arr_orig = Array(to_sparse_array(q4))
        arr_cc   = Array(to_sparse_array(qcc))
        @test norm(arr_orig - arr_cc) < 1e-10
    end

    # ── show does not error ───────────────────────────────────────────────────
    @testset "show on rank-4 QSpace" begin
        buf = IOBuffer()
        @test (show(buf, MIME"text/plain"(), q4); true)
        out = String(take!(buf))
        @test occursin("4D QSpace", out)
    end
end


