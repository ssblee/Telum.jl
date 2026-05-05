@testset "re-exported LurCGT symmetries" begin
    @test :Z in names(Telum)
    @test :U1 in names(Telum)
    @test :SU in names(Telum)
    @test :SO in names(Telum)
    @test :Sp in names(Telum)

    @test Telum.Z === LurCGT.Z
    @test Telum.U1 === LurCGT.U1
    @test Telum.SU === LurCGT.SU
    @test Telum.SO === LurCGT.SO
    @test Telum.Sp === LurCGT.Sp
end

@testset "lock reduction in contract" begin
    test_lock_reduce(FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "spin local space" begin
    test_spin_local_space()
end

struct NonCommutingSymmetryOptions <: LocalSpaceOptions end

function Telum.getSymmetryInfo(::NonCommutingSymmetryOptions)
    symm = (U1, SU{2})
    weights = ([(1,), (-1,)], [(1,), (-1,)])
    lowering_ops = (Matrix{Int}[], [sparse([0 0; 1 0])])

    mwirops = Dict{Symbol, Tuple{AbstractMatrix{Int}, Float64}}()
    mwirops[:I] = (sparse(I, 2, 2), 1.0)
    return symm, weights, lowering_ops, mwirops
end

@testset "getLocalSpace validates cross-symmetry commutation" begin
    err = try
        getLocalSpace(NonCommutingSymmetryOptions())
        nothing
    catch caught
        caught
    end

    @test err isa ArgumentError
    @test occursin("must commute", sprint(showerror, err))
    @test occursin("weight[1]", sprint(showerror, err))
    @test occursin("lowering[1]", sprint(showerror, err))
end

@testset "auto contract requires matching spaces" begin
    test_contract_requires_matching_spaces_in_star(
        FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "contract vs contract_old" begin
    test_contract_default(FermionSOptions(1, :U1, :SU2, nothing))
    test_contract_default(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contract abelian w-matrices stay unit" begin
    test_contract_abelian_wmats_are_unit(FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "getIdentity direct contraction" begin
    test_getIdentity_direct_contract(FermionSOptions(1, :U1, :SU2, nothing))
    test_getIdentity_direct_contract(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "spaces of svd" begin
    test_spaces_svdQS(FermionSOptions(1, :U1, :SU2, nothing))
    test_spaces_svdQS(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD preprocessing for svd" begin
    test_svd_cgtsvd_preprocess(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_preprocess(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD intermediate qlabels for svd" begin
    test_svd_cgtsvd_intermediate_qrows(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_intermediate_qrows(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD intermediate qlabel row classes for svd" begin
    test_svd_cgtsvd_intermediate_qrow_equivclasses(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_intermediate_qrow_equivclasses(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD signature order for svd" begin
    test_svd_cgtsvd_signature_order(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_signature_order(FermionSOptions(3, :U1, :SU2, :SU3))
    test_svd_cgr_split_spaces_preserves_physical_leg_order()
end

@testset "CGTSVD block reduction for svd" begin
    test_svd_cgtsvd_block_reduction(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_block_reduction(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD full svd factorization" begin
    test_svd_cgtsvd_factorization(FermionSOptions(1, :U1, :SU2, nothing))
    test_svd_cgtsvd_factorization(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "CGTSVD truncation" begin
    test_truncate_svd_cgtsvd(FermionSOptions(1, :U1, :SU2, nothing))
    test_truncate_svd_cgtsvd(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "svd truncation of TLArray" begin
    test_truncate_svdQS(FermionSOptions(1, :U1, :SU2, nothing))
    test_truncate_svdQS(FermionSOptions(3, :U1, :SU2, :SU3))
end


@testset "spaces of eigen" begin
    test_spaces_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_spaces_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "missing spaces of eigen" begin
    test_missing_spaces_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_missing_spaces_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "truncate missing zero spaces of eigen" begin
    test_truncate_missing_zero_spaces_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_truncate_missing_zero_spaces_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end


@testset "conjugation of TLArray test" begin
    test_conj(FermionSOptions(1, :U1, :SU2, nothing))
    test_conj(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "norm of TLArray" begin
    test_norm(FermionSOptions(1, :U1, :SU2, nothing))
    test_norm(FermionSOptions(3, :U1, :SU2, :SU3))
end


@testset "svd test" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q   = getLocalSpace(option)
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a   = getIdentity((qi1, 2), (qi2, 2))
    qf  = TLArray(q.F123, ("lur2", "lur2", "op"))
    ct  = qf * a
    test_svdQS(ct, [2, 4])
    test_svdQS(ct, [1, 4])
    test_svdQS(ct, [1, 2])
end

@testset "eig of TLArray" begin
    test_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig autodetect of TLArray" begin
    test_eigen_autodetect(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_autodetect(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig permuted input of TLArray" begin
    test_eigen_permuted_input(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_permuted_input(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig Hermitian leg guard of TLArray" begin
    test_eigen_hermitian_leg_guard(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_hermitian_leg_guard(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig truncation of TLArray" begin
    test_discard_eigen(FermionSOptions(1, :U1, :SU2, nothing))
    test_discard_eigen(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig truncation tol of TLArray" begin
    test_discard_eigen_tol(FermionSOptions(1, :U1, :SU2, nothing))
    test_discard_eigen_tol(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig full discard of TLArray" begin
    test_eigen_general_discard(FermionSOptions(1, :U1, :SU2, nothing))
    test_eigen_general_discard(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "eig discard itag of TLArray" begin
    test_discard_eigen_itag(FermionSOptions(1, :U1, :SU2, nothing))
    test_discard_eigen_itag(FermionSOptions(3, :U1, :SU2, :SU3))
end

function permtest()
    S = SU{2}
    upsp = ((1,), (2,), (2,), (2,), (3,), (3,))
    dnsp = ((1,), (2,), (4,))

    tags = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
    dirs = ('-', '+', '-', '+', '-', '-', '-', '+', '-')
    inds = Tuple(TLIndex(tags[i], dirs[i]) for i=1:9)

    om = get_CGTom(S, upsp, dnsp).totalOM
    wmat = randn(om, 1); wmat /= norm(wmat)
    wmat = LurTensor(wmat)
    RMT = LurTensor(reshape([1.0], Tuple(1 for _=1:10)...))

    qlabels = (upsp..., dnsp...)
    cgp = (5, 8, 2, 7, 1, 6, 3, 9, 4)
    cgr = CGR(S, qlabels, wmat, cgp, (6, 3))
    rows = [row((cgr,), RMT)]
    
    # Build spaces: for each physical leg, get its qlabel via cgp mapping
    # RMT dim = 1 for all legs, qlabel wrapped in tuple for N=1 symmetry
    # Use _compute_spaces helper instead of manual construction
    spaces = _compute_spaces(rows)
    
    q = TLArray((S,), rows, inds, spaces)
    println("q created")
    pq = permutedims(q, (1, 2, 4, 8, 5, 6, 9, 3, 7))
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
    # 4. LurTensor overload returns LurTensor wrappers with correct shapes
    # ------------------------------------------------------------------
    @testset "LurTensor overload" begin
        A  = randn(3, 4, 5)
        qt = LurTensor(A)

        for leg in 1:3
            Uq, SVq, Sq = svd_leg(qt, leg; cutoff=1e-12)
            U,  SV,  S  = svd_leg(A,  leg; cutoff=1e-12)

            @test Uq  isa LurTensor
            @test SVq isa LurTensor
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
    test_qr_shared_isometry_rank1_fastpath()
    test_compress_sector(2, 1, 3; verbose=false)
    test_compress_sector(2, 7, 3; verbose=false)
    test_compress_sector(3, 5, 4; verbose=false)
    test_compress_sector_zero_wmat_shortcircuits()
end


function lur(option::LocalSpaceOptions)
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q, a, ct, arr1, arr2 = test_FAcont(option)
    println(norm(arr1 - arr2))
    @test arr1 ≈ arr2
end


@testset "Generating 1jtensor of TLArray test" begin
    test_1jpair(FermionSOptions(1, :U1, :SU2, nothing))
    test_1jpair(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "Keyword get1jtensor and legflip" begin
    test_get1jtensor_and_legflip_keywords(FermionSOptions(1, :U1, :SU2, nothing))
    test_get1jtensor_and_legflip_keywords(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contract verify_legs checks green" begin
    test_contract_verify_legs_checks_green(FermionSOptions(1, :U1, :SU2, nothing))
    test_contract_verify_legs_checks_green(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contraction of TLArray test" begin
    test_FAcont(FermionSOptions(1, :U1, :SU2, nothing))
    test_FAcont(FermionSOptions(3, :U1, :SU2, :SU3))
end

function example()
    opt = FermionSOptions(1, :U1, :SU2, nothing)
    q = getLocalSpace(opt, ("lur", "lur", "op"))
    nloc = lock(q.F1', 2) * q.F1
    return nloc
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a 3-leg test TLArray with known TLIndex properties.
#
#   leg 1: dir='+', itag="site1", plev=0, lock=0
#   leg 2: dir='-', itag="site2", plev=0, lock=0
#   leg 3: dir='-', itag="op",    plev=0, lock=0
#
# Built from a real operator TLArray so the internal row data is valid.
# ─────────────────────────────────────────────────────────────────────────────
function _make_test_qspace()
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    # q0.F123 is a 3-leg TLArray: dir=('+','-','-'), all plev=0, all lock=0.
    # TLArray(q, tags) creates a copy with new tags (only the itags field changes).
    return TLArray(q0.F123, ("site1", "site2", "op"))
end

@testset "TLIndex modifier functions" begin
    q = _make_test_qspace()
    # Fixture legs:
    #   leg 1: dir='+', itag="site1", plev=0, lock=0
    #   leg 2: dir='-', itag="site2", plev=0, lock=0
    #   leg 3: dir='-', itag="op",    plev=0, lock=0

    # ── findlegs ──────────────────────────────────────────────────────────────
    @testset "findlegs" begin
        @test findlegs(q; dir='+')       == [1]
        @test findlegs(q; dir='-')       == [2, 3]
        @test findlegs(q; itag="site1") == [1]
        @test findlegs(q; itag="site2") == [2]
        @test findlegs(q; itag="op")    == [3]
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
        @test findlegs(q; itag="site1", rev=true) == [2, 3]
        @test findlegs(q; plev=0,        rev=true) == []
        # multi-criteria (AND logic)
        @test findlegs(q; dir='-', itag="op")           == [3]
        @test findlegs(q; dir='-', itag="op", rev=true) == [1, 2]

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        @test findlegs(q_tagsets; itag="aaa,bbb") == [1]
        @test findlegs(q_tagsets; itag=("aaa,bbb", "aaa,ccc")) == [1, 2]
        @test findlegs(q_tagsets; itag=["aaa,ccc", "bbb,ccc"]) == [2, 3]
    end

    # ── findleg ───────────────────────────────────────────────────────────────
    @testset "findleg" begin
        @test findleg(q; dir='+')             == 1
        @test findleg(q; dir='-')             == 2   # first match
        @test findleg(q; itag="op")          == 3
        @test findleg(q; plev=0)              == 1   # first of all legs
        @test findleg(q; dir='+', rev=true)   == 2   # first non-'+' leg
        @test findleg(q; plev=0, rev=true)   === nothing  # no leg with plev≠0
        @test findleg(q; itag="nope")        === nothing

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        @test findleg(q_tagsets; itag=("aaa,ccc", "bbb,ccc")) == 2
        @test findleg(q_tagsets; itag=["missing", "bbb,ccc"]) == 3
    end

    @testset "matching / unmatching" begin
        q_adj = q'

        @test matching(q, q_adj) == 1
        @test matchings(q, q_adj) == [1, 2, 3]
        @test unmatching(q, q_adj) === nothing
        @test unmatchings(q, q_adj) == Int[]

        q_selective = TLArray(q_adj, (
            TLIndex("site1", '-', 1, 0, false),
            TLIndex("site2", '+', 0, 0, true),
            TLIndex("op", '+', 0, 7, false),
        ))

        @test matching(q, q_selective) == 3
        @test matchings(q, q_selective) == [3]
        @test unmatching(q, q_selective) == 1
        @test unmatchings(q, q_selective) == [1, 2]

        @test matching(q, q_selective; dir='+') === nothing
        @test matchings(q, q_selective; itag="op") == [3]
        @test matchings(q, q_selective; dir='+', rev=true) == [3]
        @test unmatching(q, q_selective; itag="op") === nothing
        @test unmatchings(q, q_selective; dir='-', rev=true) == [1]

        q_match_locked = lock(q, 3)
        q_unmatch_locked = lock(q, 1)
        @test matchings(q_match_locked, q_selective; lock=1) == [3]
        @test matchings(q_match_locked, q_selective; lock=0) == Int[]
        @test unmatchings(q_unmatch_locked, q_selective; lock=1) == [1]

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        q_tagsets_adj = q_tagsets'
        @test matchings(q_tagsets, q_tagsets_adj; itag=("aaa,bbb", "bbb,ccc")) == [1, 3]
        @test matchings(q_tagsets, q_tagsets_adj; itag=["aaa,ccc", "bbb,ccc"], rev=true) == [1]
    end

    @testset "contractable / uncontractable" begin
        q_adj = q'

        @test contractable(q, q_adj) == 1
        @test contractables(q, q_adj) == [1, 2, 3]
        @test uncontractable(q, q_adj) === nothing
        @test uncontractables(q, q_adj) == Int[]

        q_selective = TLArray(q_adj, (
            TLIndex("site1", '-', 1, 0, false),
            TLIndex("site2", '+', 0, 0, true),
            TLIndex("op", '+', 0, 7, false),
        ))

        @test contractable(q, q_selective) === nothing
        @test contractables(q, q_selective) == Int[]
        @test uncontractable(q, q_selective) == 1
        @test uncontractables(q, q_selective) == [1, 2, 3]

        q_a_locked = lock(q, 3)
        q_b_locked = lock(q_adj, 2)
        @test contractables(q_a_locked, q_adj) == [1, 2]
        @test uncontractables(q_a_locked, q_adj) == [3]
        @test contractables(q, q_b_locked) == [1, 3]
        @test uncontractables(q, q_b_locked) == [2]

        @test contractables(q, q_adj; dir='-') == [2, 3]
        @test contractables(q, q_adj; itag="op") == [3]
        @test contractables(q_a_locked, q_adj; lock=1) == Int[]
        @test uncontractables(q_a_locked, q_adj; lock=1) == [3]
        @test uncontractables(q, q_b_locked; dir='+', rev=true) == [2]

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        q_tagsets_adj = q_tagsets'
        @test contractables(q_tagsets, q_tagsets_adj; itag=("aaa,bbb", "aaa,ccc")) == [1, 2]
        @test uncontractables(lock(q_tagsets, 2), q_tagsets_adj; itag=["aaa,ccc", "bbb,ccc"]) == [2]
    end

    @testset "Itag predicate equality" begin
        q_unsorted = TLArray(q, ("beta,alpha", "site2", "op"))

        @test q_unsorted.inds[1].itags isa Itag
        @test q_unsorted.inds[1].itags == "alpha,beta"
        @test q_unsorted.inds[1].itags == "beta,alpha"
        @test "beta,alpha" == q_unsorted.inds[1].itags
        @test findleg(q_unsorted, idx -> idx.itags == "beta,alpha") == 1
        @test findlegs(q_unsorted, idx -> idx.itags == "beta,alpha") == [1]
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

        q2 = lock(q; inc=3, itag="op")
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
        q2 = lockp(q; itag="site2")
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
        # LegList: vector (legs 1,2 differ in dir → unique TLIndex)
        q2 = setprime(q, [1, 2], 7)
        @test q2.inds[1].plev == 7
        @test q2.inds[2].plev == 7
        @test q2.inds[3].plev == 0

        # LegList: tuple (legs 1,3 differ in dir → unique TLIndex)
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

    # ── additag ───────────────────────────────────────────────────────────────
    @testset "additag" begin
        # criteria form: all legs (tags sorted alphabetically)
        q2 = additag(q, "new")
        @test q2.inds[1].itags == "new,site1"   # site1 + new → sorted: new,site1
        @test q2.inds[2].itags == "new,site2"   # site2 + new → sorted: new,site2
        @test q2.inds[3].itags == "new,op"      # op    + new → sorted: new,op

        # single leg
        q2 = additag(q, 1, "u1")
        @test q2.inds[1].itags == "site1,u1"
        @test q2.inds[2].itags == "site2"   # unchanged

        # LegList: vector
        q2 = additag(q, [2, 3], "phys")
        @test q2.inds[2].itags == "phys,site2"
        @test q2.inds[3].itags == "op,phys"
        @test q2.inds[1].itags == "site1"

        # LegList: tuple
        q2 = additag(q, (1, 3), "x")
        @test q2.inds[1].itags == "site1,x"
        @test q2.inds[3].itags == "op,x"
        @test q2.inds[2].itags == "site2"

        # criteria with selector: leg 1 only
        q2 = additag(q, "u1"; dir='+')
        @test q2.inds[1].itags == "site1,u1"
        @test q2.inds[2].itags == "site2"   # unchanged

        # rev: legs 2, 3
        q2 = additag(q, "u1"; dir='+', rev=true)
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "site2,u1"
        @test q2.inds[3].itags == "op,u1"
    end

    # ── removeitag ────────────────────────────────────────────────────────────
    @testset "removeitag" begin
        # criteria form: only leg 1 has "site1"
        q2 = removeitag(q, "site1")
        @test q2.inds[1].itags == ""       # "site1" removed → empty
        @test q2.inds[2].itags == "site2"  # unchanged (no "site1")
        @test q2.inds[3].itags == "op"     # unchanged

        # single leg
        q2 = removeitag(q, 2, "site2")
        @test q2.inds[2].itags == ""
        @test q2.inds[1].itags == "site1"

        # LegList: vector
        q_extra = additag(q, "extra")
        q2 = removeitag(q_extra, [1, 3], "extra")
        @test q2.inds[1].itags == "site1"
        @test q2.inds[3].itags == "op"
        @test q2.inds[2].itags == "extra,site2"   # unchanged

        # LegList: tuple
        q2 = removeitag(q_extra, (2, 3), "extra")
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
        @test q2.inds[1].itags == "extra,site1"   # unchanged

        # tuple/vector tag queries only remove fully matched groups
        q_grouped = TLArray(q, ("aaa,bbb", "aaa,bbb,ccc", "bbb,ccc"))
        q2 = removeitag(q_grouped, ("aaa,bbb", "ccc"))
        @test q2.inds[1].itags == ""
        @test q2.inds[2].itags == ""
        @test q2.inds[3].itags == "bbb"

        q2 = removeitag(q_grouped, 2, ["aaa,bbb", "ccc"])
        @test q2.inds[1].itags == "aaa,bbb"
        @test q2.inds[2].itags == ""
        @test q2.inds[3].itags == "bbb,ccc"

        # collection tags also support keyword leg selection
        q2 = removeitag(q_grouped, ("aaa,bbb", "ccc"); dir='-')
        @test q2.inds[1].itags == "aaa,bbb"
        @test q2.inds[2].itags == ""
        @test q2.inds[3].itags == "bbb"

        q2 = removeitag(q_grouped, ["aaa,bbb", "ccc"]; dir='-', rev=true)
        @test q2.inds[1].itags == ""
        @test q2.inds[2].itags == "aaa,bbb,ccc"
        @test q2.inds[3].itags == "bbb,ccc"

        # criteria with selector: leg 1
        q2 = removeitag(q_extra, "extra"; dir='+')
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "extra,site2"   # unchanged

        # rev: legs 2, 3
        q2 = removeitag(q_extra, "extra"; dir='+', rev=true)
        @test q2.inds[1].itags == "extra,site1"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
    end

    # ── setitag ───────────────────────────────────────────────────────────────
    @testset "setitag" begin
        # single leg
        q2 = setitag(q, 3, "phys")
        @test q2.inds[3].itags == "phys"
        @test q2.inds[1].itags == "site1"

        # LegList: vector (legs 1,2 differ in dir → unique TLIndex)
        q2 = setitag(q, [1, 2], "lur")
        @test q2.inds[1].itags == "lur"
        @test q2.inds[2].itags == "lur"
        @test q2.inds[3].itags == "op"

        # LegList: tuple (legs 1,3 differ in dir → unique TLIndex)
        q2 = setitag(q, (1, 3), "x")
        @test q2.inds[1].itags == "x"
        @test q2.inds[3].itags == "x"
        @test q2.inds[2].itags == "site2"

        # criteria: only leg 1 (dir='+')
        q2 = setitag(q, "phys"; dir='+')
        @test q2.inds[1].itags == "phys"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"

        # rev: only leg 1 (not dir='-' means not legs 2,3)
        q2 = setitag(q, "phys"; dir='-', rev=true)
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

        q2 = additag(q, 1, "extra")
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
        inds = ntuple(i -> TLIndex("l$i", i == 1 ? '+' : '-'), QD)
        q = empty_qspace(symm, inds)

        @test length(q.rows)  == 0
        @test length(q.inds)  == QD
        @test Telum.symm(q) == symm
        @test q.inds          == inds
        @test all(isempty(q.spaces[l]) for l in 1:QD)
    end

    # ── multiple symmetries ───────────────────────────────────────────────────
    @testset "multi-symmetry construction" begin
        symm = (U1, SU{2})
        inds = (TLIndex("a", '+'), TLIndex("b", '-'), TLIndex("c", '-'))
        q = empty_qspace(symm, inds)

        @test length(Telum.symm(q)) == 2
        @test length(q.rows) == 0
        @test length(q.inds) == 3
        @test Telum.symm(q) == symm
        # spaces: 3 empty vectors, one per leg
        @test length(q.spaces) == 3
        @test all(isempty(q.spaces[l]) for l in 1:3)
    end

    # ── element type keyword ──────────────────────────────────────────────────
    @testset "element type" begin
        symm = (SU{2},)
        inds = (TLIndex("a", '+'), TLIndex("b", '-'))
        qf64 = empty_qspace(symm, inds; T=Float64)
        qc64 = empty_qspace(symm, inds; T=ComplexF64)

        # N=1, QD=2, RD=3
        @test eltype(qf64.rows) <: row{Float64, 2, 1, 3}
        @test eltype(qc64.rows) <: row{ComplexF64, 2, 1, 3}
    end

    # ── TLIndex modifier operations on an empty TLArray ─────────────────────────
    @testset "modifier ops on empty TLArray" begin
        symm = (SU{2}, U1)
        inds = (TLIndex("a", '+'), TLIndex("b", '-'), TLIndex("c", '-'))
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
        @test additag(q, "x").inds[1].itags           == "a,x"
        @test removeitag(q, "a").inds[1].itags        == ""
        @test setitag(q, "new"; dir='+').inds[1].itags == "new"
        @test !isdefined(Telum, :replaceitag)

        # findlegs / findleg
        @test findlegs(q; dir='+') == [1]
        @test findlegs(q; dir='-') == [2, 3]
        @test findleg(q; dir='+')  == 1
        @test findleg(q; dir='-')  == 2

        # scalar multiplication of empty TLArray produces empty TLArray
        q_scaled = 3.0 * q
        @test length(q_scaled.rows) == 0
    end

    @testset "zero preserves metadata on TLArray" begin
        option = FermionSOptions(3, :U1, :SU2, :SU3)
        q0 = getLocalSpace(option)
        q = TLArray(q0.F123, ("site1", "site2", "op"))
        qz = zero(q)

        @test qz isa TLArray
        @test isempty(qz.rows)
        @test symm(qz) == symm(q)
        @test qz.inds == q.inds
        @test qz.spaces == q.spaces
        @test qz.spaces !== q.spaces
        @test all(qz.spaces[leg] !== q.spaces[leg] for leg in eachindex(q.spaces))
    end

    # ── show does not error ───────────────────────────────────────────────────
    @testset "show on empty TLArray" begin
        symm = (SU{2},)
        inds = (TLIndex("a", '+'), TLIndex("b", '-'))
        q = empty_qspace(symm, inds)
        buf = IOBuffer()
        # must not throw
        @test (show(buf, MIME"text/plain"(), q); true)
        # output should mention "(empty)"
        @test occursin("empty", String(take!(buf)))
    end
end

@testset "zero_qlabels" begin
    q_empty = empty_qspace((SU{2}, SU{3}), (TLIndex('+'), TLIndex('-')))
    @test zero_qlabels(q_empty) == ((0,), (0, 0))
    @test zero_qlabels(symm(q_empty)) == ((0,), (0, 0))

    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    @test zero_qlabels(q0.I) == ((0,), (0,))
end

@testset "qlabeltype" begin
    q_empty = empty_qspace((U1, SU{3}), (TLIndex('+'), TLIndex('-')))
    expected = Tuple{Tuple{Int}, NTuple{2, Int}}
    expected_ps = ProductSymm{Tuple{U1, SU{3}}}

    @test qlabeltype(q_empty) == expected
    @test qlabeltype(symm(q_empty)) == expected
    @test typeof(q_empty).parameters[5] == expected
    @test typeof(q_empty).parameters[6] == expected_ps
    @test typeof(q_empty).parameters[7] == Tuple{CGR{2, 1, U1}, CGR{2, 2, SU{3}}}
    @test !(:symm in fieldnames(typeof(q_empty)))
    @test productsymm(q_empty) == expected_ps
    @test productsymm(symm(q_empty)) == expected_ps
    @test symm(q_empty) == (U1, SU{3})
    @test @inferred(symm(q_empty)) == (U1, SU{3})
    @test product_symms(q_empty) == (U1, SU{3})
    @test nsymms(q_empty) == 2
    @test eltype(q_empty.spaces[1]) == Tuple{expected, Int}
    entries1 = Telum._contracted_qlabel_entries(expected, q_empty.rows, (1,))
    entries2 = Telum._contracted_qlabel_entries(expected, q_empty.rows, (1, 2))
    @test entries1 isa Vector
    @test entries2 isa Vector

    q_multi = empty_qspace((U1, SU{2}, SU{3}), (TLIndex('+'),))
    @test qlabeltype(q_multi) == Tuple{Tuple{Int}, Tuple{Int}, NTuple{2, Int}}
    @test typeof(q_multi).parameters[5] == Tuple{Tuple{Int}, Tuple{Int}, NTuple{2, Int}}
    @test productsymm(q_multi) == ProductSymm{Tuple{U1, SU{2}, SU{3}}}
    @test typeof(q_multi).parameters[7] == Tuple{CGR{1, 1, U1}, CGR{1, 1, SU{2}}, CGR{1, 2, SU{3}}}

    q_local = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).I
    @test !(:symm in fieldnames(typeof(q_local)))
    @test all(!(:symm in fieldnames(typeof(cgr))) for r in q_local.rows for cgr in r.cgrs)
    @test typeof(q_local.rows).parameters[1].parameters[5] == Tuple{CGR{2, 1, U1}, CGR{2, 1, SU{2}}}
    info = Telum.leginfo(q_local, 1)
    @test !(:symm in fieldnames(typeof(info)))
    @test symm(info) == symm(q_local)
    @test productsymm(info) == productsymm(q_local)
    @test product_symms(info) == product_symms(q_local)
    @test nsymms(info) == nsymms(q_local)
    @test qlabeltype(info) == qlabeltype(q_local)
    @test typeof(info).parameters[2] == qlabeltype(q_local)
    @test typeof(info).parameters[3] == productsymm(q_local)
    @test eltype(info.splist) == Tuple{qlabeltype(q_local), Int}
    @test typeof(Telum.leginfo(symm(q_local), q_local.inds[1], q_local.spaces[1])) == typeof(info)
end

@testset "getsub sector slicing" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    candidate = nothing
    for base in values(q0)
        for legcand in 1:length(base.spaces)
            qcand = oplus([base, 2.0 * base, 3.0 * base], legcand)
            reorder_idx = findfirst(i -> qcand.spaces[legcand][i][2] >= 3, eachindex(qcand.spaces[legcand]))
            isnothing(reorder_idx) && continue

            full_idx = findfirst(i -> i != reorder_idx, eachindex(qcand.spaces[legcand]))
            isnothing(full_idx) && continue

            candidate = (
                q = qcand,
                leg = legcand,
                sector_reorder = qcand.spaces[legcand][reorder_idx][1],
                dim_reorder = qcand.spaces[legcand][reorder_idx][2],
                sector_full = qcand.spaces[legcand][full_idx][1],
                dim_full = qcand.spaces[legcand][full_idx][2],
            )
            break
        end
        !isnothing(candidate) && break
    end

    @test !isnothing(candidate)

    q = candidate.q
    leg = candidate.leg
    sector_reorder = candidate.sector_reorder
    dim_reorder = candidate.dim_reorder
    sector_full = candidate.sector_full
    dim_full = candidate.dim_full

    row_sector(qs::TLArray, r) = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[leg]] for n in 1:length(symm(qs)))
    rows_for_sector(qs::TLArray, sector) = [r for r in qs.rows if row_sector(qs, r) == sector]

    q_pred_pairs = Telum.getsub(q, leg,
                                  sector -> sector == sector_full ? Colon() :
                                            sector == sector_reorder ? (dim_reorder, 1) :
                                            nothing)

    expected_spaces_leg = [
        (sector, sector == sector_full ? dim_full : 2)
        for (sector, _) in q.spaces[leg]
        if sector == sector_full || sector == sector_reorder
    ]

    @test q_pred_pairs.spaces[leg] == expected_spaces_leg
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_pred_pairs.spaces[other_leg] == q.spaces[other_leg]
    end
    @test Set(row_sector(q_pred_pairs, r) for r in q_pred_pairs.rows) == Set([sector_full, sector_reorder])

    full_rows = rows_for_sector(q_pred_pairs, sector_full)
    orig_full_rows = rows_for_sector(q, sector_full)
    @test length(full_rows) == length(orig_full_rows)
    for (full_row, orig_full_row) in zip(full_rows, orig_full_rows)
        @test full_row.RMT.data == orig_full_row.RMT.data
    end

    reorder_rows = rows_for_sector(q_pred_pairs, sector_reorder)
    orig_reorder_rows = rows_for_sector(q, sector_reorder)
    @test length(reorder_rows) == length(orig_reorder_rows)
    reorder_inds = [dim_reorder, 1]
    for (reorder_row, orig_reorder_row) in zip(reorder_rows, orig_reorder_rows)
        reorder_selector = ntuple(d -> d == leg ? reorder_inds : Colon(), ndims(orig_reorder_row.RMT.data))
        @test reorder_row.RMT.data == orig_reorder_row.RMT.data[reorder_selector...]
    end

    q_single = Telum.getsub(q, leg, sector -> sector == sector_reorder ? 2 : nothing)
    single_rows = rows_for_sector(q_single, sector_reorder)
    @test q_single.spaces[leg] == [(sector_reorder, 1)]
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_single.spaces[other_leg] == q.spaces[other_leg]
    end
    @test length(single_rows) == length(orig_reorder_rows)
    for (single_row, orig_reorder_row) in zip(single_rows, orig_reorder_rows)
        single_selector = ntuple(d -> d == leg ? [2] : Colon(), ndims(orig_reorder_row.RMT.data))
        @test single_row.RMT.data == orig_reorder_row.RMT.data[single_selector...]
    end

    q_range = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (1:2) : nothing)
    @test q_range.spaces[leg] == [(sector_reorder, 2)]
    range_rows = rows_for_sector(q_range, sector_reorder)
    @test length(range_rows) == length(orig_reorder_rows)
    for (range_row, orig_reorder_row) in zip(range_rows, orig_reorder_rows)
        range_selector = ntuple(d -> d == leg ? [1, 2] : Colon(), ndims(orig_reorder_row.RMT.data))
        @test range_row.RMT.data == orig_reorder_row.RMT.data[range_selector...]
    end

    q_negative = Telum.getsub(q, leg, sector -> sector == sector_reorder ? -1 : nothing)
    negative_rows = rows_for_sector(q_negative, sector_reorder)
    @test q_negative.spaces[leg] == [(sector_reorder, 1)]
    @test length(negative_rows) == length(orig_reorder_rows)
    for (negative_row, orig_reorder_row) in zip(negative_rows, orig_reorder_rows)
        negative_selector = ntuple(d -> d == leg ? [dim_reorder] : Colon(), ndims(orig_reorder_row.RMT.data))
        @test negative_row.RMT.data == orig_reorder_row.RMT.data[negative_selector...]
    end

    q_negative_range = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (-2:-1) : nothing)
    negative_range_rows = rows_for_sector(q_negative_range, sector_reorder)
    @test q_negative_range.spaces[leg] == [(sector_reorder, 2)]
    @test length(negative_range_rows) == length(orig_reorder_rows)
    for (negative_range_row, orig_reorder_row) in zip(negative_range_rows, orig_reorder_rows)
        negative_range_selector = ntuple(d -> d == leg ? [dim_reorder - 1, dim_reorder] : Colon(), ndims(orig_reorder_row.RMT.data))
        @test negative_range_row.RMT.data == orig_reorder_row.RMT.data[negative_range_selector...]
    end

    q_mixed = Telum.getsub(q, leg, sector -> sector == sector_reorder ? [-1, 1] : nothing)
    mixed_rows = rows_for_sector(q_mixed, sector_reorder)
    @test q_mixed.spaces[leg] == [(sector_reorder, 2)]
    @test length(mixed_rows) == length(orig_reorder_rows)
    for (mixed_row, orig_reorder_row) in zip(mixed_rows, orig_reorder_rows)
        mixed_selector = ntuple(d -> d == leg ? [dim_reorder, 1] : Colon(), ndims(orig_reorder_row.RMT.data))
        @test mixed_row.RMT.data == orig_reorder_row.RMT.data[mixed_selector...]
    end

    q_tuple_pick = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (dim_reorder, 1) : nothing)
    @test q_tuple_pick.spaces == q_mixed.spaces
    tuple_pick_rows = rows_for_sector(q_tuple_pick, sector_reorder)
    @test length(tuple_pick_rows) == length(mixed_rows)
    for (tuple_pick_row, mixed_row) in zip(tuple_pick_rows, mixed_rows)
        @test tuple_pick_row.RMT.data == mixed_row.RMT.data
    end

    q_empty = Telum.getsub(q, leg, _ -> nothing)
    @test isempty(q_empty.rows)
    @test isempty(q_empty.spaces[leg])
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_empty.spaces[other_leg] == q.spaces[other_leg]
    end

    @test_throws ArgumentError Telum.getsub(q, 0, _ -> Colon())
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? [1, 1] : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? [1, -dim_reorder] : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? Int[] : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? 0 : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? (dim_reorder + 1) : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? -(dim_reorder + 1) : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? "bad" : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? 1 : nothing; preserve_space=true)
end

@testset "getsub sector predicate" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    candidate = nothing
    for base in values(q0)
        for legcand in 1:length(base.spaces)
            qcand = oplus([base, 2.0 * base, 3.0 * base], legcand)
            length(qcand.spaces[legcand]) >= 2 || continue
            candidate = (q = qcand, leg = legcand, target_sector = qcand.spaces[legcand][1][1])
            break
        end
        !isnothing(candidate) && break
    end

    @test !isnothing(candidate)

    q = candidate.q
    leg = candidate.leg
    target_sector = candidate.target_sector

    row_sector(qs::TLArray, r) = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[leg]] for n in 1:length(symm(qs)))
    expected_rows = [r for r in q.rows if row_sector(q, r) == target_sector]
    expected_leg_spaces = [entry for entry in q.spaces[leg] if entry[1] == target_sector]

    q_exact = Telum.getsub(q, leg, sector -> sector == target_sector ? Colon() : nothing)
    @test _rows_equal(q_exact.rows, expected_rows)
    @test q_exact.spaces[leg] == expected_leg_spaces
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_exact.spaces[other_leg] == q.spaces[other_leg]
    end

    q_component = Telum.getsub(q, leg, sector -> sector[1] == target_sector[1] ? Colon() : nothing)
    expected_component_rows = [r for r in q.rows if row_sector(q, r)[1] == target_sector[1]]
    expected_component_spaces = [entry for entry in q.spaces[leg] if entry[1][1] == target_sector[1]]
    @test _rows_equal(q_component.rows, expected_component_rows)
    @test q_component.spaces[leg] == expected_component_spaces

    q_preserved = Telum.getsub(q, leg, sector -> sector == target_sector ? Colon() : nothing; preserve_space=true)
    @test _rows_equal(q_preserved.rows, expected_rows)
    @test q_preserved.spaces == q.spaces
    @test all(q_preserved.spaces[legidx] !== q.spaces[legidx] for legidx in 1:length(q.spaces))

    q_none = Telum.getsub(q, leg, _ -> nothing)
    @test isempty(q_none.rows)
    @test isempty(q_none.spaces[leg])
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_none.spaces[other_leg] == q.spaces[other_leg]
    end

    q_none_preserved = Telum.getsub(q, leg, _ -> nothing; preserve_space=true)
    @test isempty(q_none_preserved.rows)
    @test q_none_preserved.spaces == q.spaces

    @test_throws ArgumentError Telum.getsub(q, leg, _ -> false)
    @test_throws ArgumentError Telum.getsub(q, leg, _ -> true)
    @test_throws ArgumentError Telum.getsub(q, 0, _ -> Colon())
end

@testset "getsub multi-leg sector predicate" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    base = getLocalSpace(option, ("sel,left", "sel,right", "op")).I
    q = oplus([base, 2.0 * base, 3.0 * base], (1, 2))
    legs = (1, 2)

    @test length(q.spaces[1]) >= 2
    @test length(q.spaces[2]) >= 2
    @test !isempty(q.rows)

    row_sector_at(r, leg) = Tuple(r.cgrs[n].qlabels[r.cgrs[n].cgp[leg]] for n in 1:length(symm(q)))
    allowed = Set{Any}([row_sector_at(q.rows[1], 1), row_sector_at(q.rows[1], 2)])
    pred = sector -> sector in allowed ? Colon() : nothing

    expected_rows = [
        r for r in q.rows
        if all(row_sector_at(r, leg) in allowed for leg in legs)
    ]
    expected_spaces_1 = [entry for entry in q.spaces[1] if entry[1] in allowed]
    expected_spaces_2 = [entry for entry in q.spaces[2] if entry[1] in allowed]

    q_multi = Telum.getsub(q, legs, pred)
    @test _rows_equal(q_multi.rows, expected_rows)
    @test q_multi.spaces[1] == expected_spaces_1
    @test q_multi.spaces[2] == expected_spaces_2
    for other_leg in 1:length(q.spaces)
        other_leg in legs && continue
        @test q_multi.spaces[other_leg] == q.spaces[other_leg]
    end

    pred_slice = sector -> sector in allowed ? 1 : nothing
    q_multi_sliced = Telum.getsub(q, legs, pred_slice)
    @test q_multi_sliced.spaces[1] == [(entry[1], 1) for entry in q.spaces[1] if entry[1] in allowed]
    @test q_multi_sliced.spaces[2] == [(entry[1], 1) for entry in q.spaces[2] if entry[1] in allowed]
    for other_leg in 1:length(q.spaces)
        other_leg in legs && continue
        @test q_multi_sliced.spaces[other_leg] == q.spaces[other_leg]
    end
    @test length(q_multi_sliced.rows) == length(expected_rows)
    for (sliced_row, orig_row) in zip(q_multi_sliced.rows, expected_rows)
        slice_selector = ntuple(d -> d in legs ? [1] : Colon(), ndims(orig_row.RMT.data))
        @test sliced_row.RMT.data == orig_row.RMT.data[slice_selector...]
    end

    q_multi_range = Telum.getsub(q, 1:2, pred)
    @test _rows_equal(q_multi_range.rows, q_multi.rows)
    @test q_multi_range.spaces == q_multi.spaces

    q_multi_preserved = Telum.getsub(q, legs, pred; preserve_space=true)
    @test _rows_equal(q_multi_preserved.rows, expected_rows)
    @test q_multi_preserved.spaces == q.spaces

    q_multi_kw = Telum.getsub(q, pred; itag="sel")
    @test _rows_equal(q_multi_kw.rows, q_multi.rows)
    @test q_multi_kw.spaces == q_multi.spaces

    q_multi_kw_preserved = Telum.getsub(q, pred; itag="sel", preserve_space=true)
    @test _rows_equal(q_multi_kw_preserved.rows, q_multi.rows)
    @test q_multi_kw_preserved.spaces == q.spaces

    @test_throws ArgumentError Telum.getsub(q, Int[], pred)
    @test_throws ArgumentError Telum.getsub(q, (1, 1), pred)
    @test_throws ArgumentError Telum.getsub(q, (0, 1), pred)
    @test_throws ArgumentError Telum.getsub(q, legs, pred_slice; preserve_space=true)
    @test_throws ArgumentError Telum.getsub(q, pred; itag="missing")
end

@testset "getvac" begin
    @testset "single trivial sector with default tags" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        vac = getvac(q0.I)

        @test symm(vac) == symm(q0.I)
        @test length(vac.rows) == 1
        @test vac.inds == (TLIndex("", '+'), TLIndex("", '-'))
        @test length(vac.spaces[1]) == 1
        @test length(vac.spaces[2]) == 1

        trivial = zero_qlabels(vac)
        @test vac.spaces[1][1] == (trivial, 1)
        @test vac.spaces[2][1] == (trivial, 1)

        r = vac.rows[1]
        @test size(r.RMT.data) == ntuple(_ -> 1, length(symm(vac)) + 2)
        @test only(r.RMT.data) == one(eltype(r.RMT.data))
        for n in 1:length(symm(vac))
            @test r.cgrs[n].qlabels == (trivial[n], trivial[n])
            @test r.cgrs[n].cgp == (1, 2)
            @test r.cgrs[n].legdir == (1, 1)
            @test size(r.cgrs[n].wmat.data) == (1, 1)
            @test r.cgrs[n].wmat[1] == 1.0
        end
    end

    @testset "optional tags are applied" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        vac = getvac(q0.F1, ("vin", "vout"))

        @test vac.inds[1] == TLIndex("vin", '+')
        @test vac.inds[2] == TLIndex("vout", '-')
    end
end

@testset "addSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F1
    q_rank2 = TLArray(q0.I, ("lin", "lout"))

    q_default = addSingleton(q, 2)
    @test q_default.inds[1] == q.inds[1]
    @test q_default.inds[2] == TLIndex("", '+')
    @test q_default.inds[3] == q.inds[2]
    @test q_default.inds[4] == q.inds[3]

    trivial = zero_qlabels(q)
    @test q_default.spaces[2] == [(trivial, 1)]

    q_added = addSingleton(q, (1, 4);
                           itag=("left_aux", "right_aux"),
                           plev=(2, 3),
                           lock=(0, 1),
                           dir=('-', '+'))

    @test q_added.inds[1] == TLIndex("left_aux", '-', 2, 0)
    @test q_added.inds[2] == q.inds[1]
    @test q_added.inds[3] == q.inds[2]
    @test q_added.inds[4] == TLIndex("right_aux", '+', 3, 1)
    @test q_added.inds[5] == q.inds[3]
    @test q_added.spaces[1] == [(trivial, 1)]
    @test q_added.spaces[4] == [(trivial, 1)]

    arr_ref = _dense_addSingleton_ref(Array(to_sparse_array(q)), (1, 4))
    arr_added = Array(to_sparse_array(q_added))
    @test size(arr_added) == size(arr_ref)
    @test norm(arr_added - arr_ref) < 1e-10

    q_rank2_added = addSingleton(q_rank2, 2)
    arr_rank2_ref = _dense_addSingleton_ref(Array(to_sparse_array(q_rank2)), 2)
    arr_rank2_added = Array(to_sparse_array(q_rank2_added))
    @test size(arr_rank2_added) == size(arr_rank2_ref)
    @test norm(arr_rank2_added - arr_rank2_ref) < 1e-10
    @test all(all(cgr.wmat[1] == 1.0 for cgr in r.cgrs) for r in q_rank2_added.rows)
end

@testset "deleteSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F1
    q_rank2 = TLArray(q0.I, ("lin", "lout"))

    q_one = addSingleton(q, 2; itag="aux", plev=3, dir='-')
    q_two = addSingleton(q, (1, 4);
                         itag=("left_aux", "right_aux"),
                         plev=(2, 5),
                         dir=('-', '+'))
    q_rank2_added = addSingleton(q_rank2, 2; itag="mid_aux", dir='+')

    q_deleted_all = deleteSingleton(q_two)
    @test q_deleted_all.inds == q.inds
    @test q_deleted_all.spaces == q.spaces
    @test Array(to_sparse_array(q_deleted_all)) == Array(to_sparse_array(q))

    q_deleted_leg = deleteSingleton(q_one, 2)
    @test q_deleted_leg.inds == q.inds
    @test q_deleted_leg.spaces == q.spaces
    @test Array(to_sparse_array(q_deleted_leg)) == Array(to_sparse_array(q))

    q_deleted_legs = deleteSingleton(q_two, (1, 4))
    @test q_deleted_legs.inds == q.inds
    @test q_deleted_legs.spaces == q.spaces
    @test Array(to_sparse_array(q_deleted_legs)) == Array(to_sparse_array(q))

    q_deleted_kw_tag = deleteSingleton(q_two; itag="left_aux")
    @test length(q_deleted_kw_tag.inds) == 4
    @test q_deleted_kw_tag.inds[1] == q.inds[1]
    @test q_deleted_kw_tag.inds[2] == q.inds[2]
    @test q_deleted_kw_tag.inds[3] == TLIndex("right_aux", '+', 5, 0)
    @test q_deleted_kw_tag.inds[4] == q.inds[3]
    @test Array(to_sparse_array(q_deleted_kw_tag)) == Array(to_sparse_array(addSingleton(q, 3; itag="right_aux", plev=5, dir='+')))

    q_deleted_kw_dir = deleteSingleton(q_two; dir='-')
    @test length(q_deleted_kw_dir.inds) == 4
    @test q_deleted_kw_dir.inds[1] == q.inds[1]
    @test q_deleted_kw_dir.inds[2] == q.inds[2]
    @test q_deleted_kw_dir.inds[3] == TLIndex("right_aux", '+', 5, 0)
    @test q_deleted_kw_dir.inds[4] == q.inds[3]

    q_deleted_kw_plev = deleteSingleton(q_two; plev=5)
    @test length(q_deleted_kw_plev.inds) == 4
    @test q_deleted_kw_plev.inds[1] == TLIndex("left_aux", '-', 2, 0)
    @test q_deleted_kw_plev.inds[2] == q.inds[1]
    @test q_deleted_kw_plev.inds[3] == q.inds[2]
    @test q_deleted_kw_plev.inds[4] == q.inds[3]

    q_rank2_roundtrip = deleteSingleton(q_rank2_added)
    @test q_rank2_roundtrip.inds == q_rank2.inds
    @test q_rank2_roundtrip.spaces == q_rank2.spaces
    @test Array(to_sparse_array(q_rank2_roundtrip)) == Array(to_sparse_array(q_rank2))

    @test_throws ArgumentError deleteSingleton(q, 1)
    @test_throws ArgumentError deleteSingleton(q_two, (1, 2))
    @test_throws ArgumentError deleteSingleton(q_two, Int[])
    @test_throws ArgumentError deleteSingleton(q_two, (1, 1))
    @test_throws ArgumentError deleteSingleton(q_two, 0)

    @test_logs (:warn, r"no singleton legs found") deleteSingleton(q)
    @test_logs (:warn, r"no singleton legs matched") deleteSingleton(q_two; itag="ain")
end

@testset "TLArray tensor product" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    q1 = TLArray(q0.I, ("l1_in", "l1_out"))
    q2 = TLArray(q0.F1, ("l2_in", "l2_out", "l2_op"))

    q12 = Telum.:⊗(q1, q2)
    q12_kron = kron(q1, q2)

    @test symm(q12) == symm(q1) == symm(q2)
    @test q12.inds == (q1.inds..., q2.inds...)
    @test q12.spaces == (q1.spaces..., q2.spaces...)
    @test q12_kron.inds == q12.inds
    @test q12_kron.spaces == q12.spaces

    arr_ref = _dense_tensor_product_ref(Array(to_sparse_array(q1)), Array(to_sparse_array(q2)))
    arr_q12 = Array(to_sparse_array(q12))
    arr_q12_kron = Array(to_sparse_array(q12_kron))
    @test size(arr_q12) == size(arr_ref)
    @test norm(arr_q12 - arr_ref) < 1e-10
    @test norm(arr_q12_kron - arr_q12) < 1e-10
end

# Helper: build a rank-4 test TLArray from getIdentity with 3 input legs.
#
#   leg 1 ('+', "l1"), leg 2 ('+', "l2"), leg 3 ('+', "l3"),
#   leg 4 ('-', "fused")
# ─────────────────────────────────────────────────────────────────────────────
function _make_test_qspace_rank4()
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0  = getLocalSpace(option)
    qi1 = TLArray(q0.I, ("b1a", "b1b"))
    qi2 = TLArray(q0.I, ("b2a", "b2b"))
    qi3 = TLArray(q0.I, ("b3a", "b3b"))
    a4  = getIdentity((qi1, 2), (qi2, 2), (qi3, 2); itag="fused")
    return TLArray(a4, ("l1", "l2", "l3", "fused"))
end


@testset "scalar add/subtract on rank-2 TLArray" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    q = TLArray(q0.I, ("left", "right"))

    idq_pairs = getIdentity((q, 2); itag=q.inds[2].itags)
    idq = getIdentity(q, 2; itag=q.inds[2].itags)
    @test idq.inds == idq_pairs.inds
    @test idq.spaces == idq_pairs.spaces
    @test _rows_equal(idq.rows, idq_pairs.rows)
    idq = TLArray(idq, q.inds)

    @test norm((q + 2.5) - (q + 2.5 * idq)) < 1e-10
    @test norm((q - 2.5) - (q - 2.5 * idq)) < 1e-10
    @test norm((2.5 + q) - (q + 2.5 * idq)) < 1e-10
    @test norm((2.5 - q) - (2.5 * idq - q)) < 1e-10

    q_bad_rank = _make_test_qspace_rank4()
    @test_throws AssertionError q_bad_rank + 1.0

    q_bad_dirs = getIdentity(q, 1, 2; itag="fused")
    @test_throws AssertionError q_bad_dirs + 1.0
end


# ─────────────────────────────────────────────────────────────────────────────
# Tests for arbitrary-rank (rank ≥ 4) TLArray
# ─────────────────────────────────────────────────────────────────────────────

@testset "arbitrary rank TLArray" begin
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
        @test findlegs(q4; itag="l1")     == [1]
        @test findlegs(q4; itag="fused")  == [4]
        @test findleg(q4; dir='+')         == 1
        @test findleg(q4; dir='-')         == 4
        @test findleg(q4; itag="fused")   == 4
        @test findlegs(q4; dir='+', rev=true) == [4]
    end

    @testset "svd keyword leg selection" begin
        U_ref, S_ref, Vd_ref = svd(q4, (1, 2, 3), "kwL", "kwR")
        U_kw, S_kw, Vd_kw = svd(q4, "kwL", "kwR"; dir='+')

        @test U_kw.inds == U_ref.inds
        @test S_kw.inds == S_ref.inds
        @test Vd_kw.inds == Vd_ref.inds

        @test U_kw.spaces == U_ref.spaces
        @test S_kw.spaces == S_ref.spaces
        @test Vd_kw.spaces == Vd_ref.spaces

        @test norm(U_kw - U_ref) < 1e-12
        @test norm(S_kw - S_ref) < 1e-12
        @test norm(Vd_kw - Vd_ref) < 1e-12

        @test_throws ArgumentError svd(q4; itag="missing")
        @test_throws ArgumentError svd(q4; lock=0)
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
        q_a = additag(q4, "phys"; dir='+')
        @test all(occursin("phys", q_a.inds[i].itags) for i in 1:3)
        @test !occursin("phys", q_a.inds[4].itags)

        q_r = removeitag(q_a, "phys"; dir='+')
        @test all(q_r.inds[i].itags == q4.inds[i].itags for i in 1:4)

        q_s = setitag(q4, "new"; dir='-')
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
    @testset "show on rank-4 TLArray" begin
        buf = IOBuffer()
        @test (show(buf, MIME"text/plain"(), q4); true)
        out = String(take!(buf))
        @test occursin("4D TLArray", out)
    end
end

