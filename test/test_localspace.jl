@testset "re-exported LurCGT symmetries" begin
    @test :Z in names(Telum)
    @test :U1 in names(Telum)
    @test :SU in names(Telum)
    @test :SO in names(Telum)
    @test :Sp in names(Telum)
    @test :printmeta in names(Telum)
    @test :getSymmetryInfo in names(Telum)

    @test Telum.Z === LurCGT.Z
    @test Telum.U1 === LurCGT.U1
    @test Telum.SU === LurCGT.SU
    @test Telum.SO === LurCGT.SO
    @test Telum.Sp === LurCGT.Sp
end

@testset "SpinOptions local operators across symmetries and spins" begin
    for symmetry in (:SU2, :U1, nothing), s2 in 1:5
        q = @test_nowarn getLocalSpace(SpinOptions(symmetry, s2))
        expected_ops = symmetry == :SU2 ? (:S, :I) : (:Sp, :Sz, :Sm, :I)

        for op in expected_ops
            @test hasproperty(q, op)
            tensor = getproperty(q, op)
            @test !isempty(tensor.RMTs)
            @test any(tensor.isdefined)
        end
    end
end

@testset "FermionS full-channel IROP names omit channel suffixes" begin
    q1 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    @test hasproperty(q1, :F)
    @test !hasproperty(q1, :F1)

    q3 = getLocalSpace(FermionSOptions(3, :U1, :SU2, :SU3))
    @test hasproperty(q3, :F)
    @test !hasproperty(q3, :F123)
end

@testset "three-channel SO(3) local space" begin
    opts = FermionSOptions(3, :U1, :SU2, :SO3)
    symmetries, weights, lowering, mwirops = getSymmetryInfo(opts)
    @test symmetries == (U1, SU{2}, SO{3})

    basic = Telum.FermionS_basicops(3)
    n1 = basic.NN[1, 1] + basic.NN[2, 1]
    n3 = basic.NN[1, 3] + basic.NN[2, 3]
    z = 2 * (n1 - n3)
    minus = sqrt(2) * sum(
        basic.FF[spin, to]' * basic.FF[spin, from]
        for spin in 1:2 for (to, from) in ((2, 1), (3, 2)))
    @test weights[3] == [(Int(value),) for value in diag(z)]
    @test lowering[3][1] ≈ minus
    @test LurCGT.comm(z, minus) ≈ -2 * minus
    @test LurCGT.comm(minus', minus) ≈ z
    @test mwirops[:F] == basic.FF[2, 3]
    @test mwirops[:S] ≈ -sum(basic.SS[i]' for i in 1:3) / sqrt(2)

    q = getLocalSpace(opts)
    @test symm(q.I) == (U1, SU{2}, SO{3})
    @test hasproperty(q, :F)
    @test hasproperty(q, :S)
    @test Array(to_sparse_array(q.I)) ≈ Matrix{Float64}(I, 64, 64)
    @test any(x -> !iszero(x), Array(to_sparse_array(q.F)))
    @test any(x -> !iszero(x), Array(to_sparse_array(q.S)))

    alias_q = getLocalSpace(FermionSOptions(3, :U1, :SU2, :SO))
    @test symm(alias_q.I) == symm(q.I)
    so_only = getLocalSpace(FermionSOptions(3, nothing, nothing, :SO3))
    @test symm(so_only.I) == (SO{3},)
    @test_throws ArgumentError getSymmetryInfo(FermionSOptions(3, :U1, :SU2, :SO4))
    @test_throws ArgumentError getSymmetryInfo(FermionSOptions(2, :U1, :SU2, :SO3))
end

@testset "Fermion local space IROPs" begin
    q1 = getLocalSpace(FermionOptions(U1))
    @test hasproperty(q1, :F)
    @test hasproperty(q1, :Z)
    @test hasproperty(q1, :I)
    @test !hasproperty(q1, :F1)
    @test size(to_sparse_array(q1.I)) == (2, 2)
    @test Array(to_sparse_array(q1.I)) == Array(_test_to_sparse_array(q1.I))

    q3 = getLocalSpace(FermionOptions(3, :U1, :SU3))
    @test hasproperty(q3, :F)
    @test !hasproperty(q3, :F123)
    @test size(to_sparse_array(q3.I)) == (8, 8)
end

@testset "no-symmetry local spaces" begin
    spin = getLocalSpace(SpinOptions(nothing, 1))
    @test symm(spin.I) == ()
    @test typeof(spin.I).parameters[6] === ProductSymm{Tuple{}}
    @test Telum.stored_qlabels(spin.I) == [((), ())]
    @test Telum.spaces(spin.I) == ([(() , 2)], [(() , 2)])
    @test _test_sector_rmt(spin.I, 1) ≈ Matrix{Float64}(I, 2, 2)
    @test _test_sector_rmt(spin.Sz, 1) ≈ [0.5 0.0; 0.0 -0.5]
    @test _test_sector_rmt(spin.Sp, 1) ≈ (-1 / sqrt(2)) * [0.0 1.0; 0.0 0.0]
    @test _test_sector_rmt(spin.Sm, 1) ≈ (1 / sqrt(2)) * [0.0 0.0; 1.0 0.0]

    fermion = getLocalSpace(FermionOptions(1, nothing, nothing))
    @test symm(fermion.I) == ()
    @test typeof(fermion.I).parameters[6] === ProductSymm{Tuple{}}
    @test Telum.stored_qlabels(fermion.I) == [((), ())]
    @test Telum.spaces(fermion.I) == ([(() , 2)], [(() , 2)])
    @test _test_sector_rmt(fermion.I, 1) ≈ Matrix{Float64}(I, 2, 2)
    @test size(_test_sector_rmt(fermion.F, 1)) == (2, 2)
    @test size(_test_sector_rmt(fermion.Z, 1)) == (2, 2)

    spinful = getLocalSpace(FermionSOptions(1, nothing, nothing, nothing))
    @test symm(spinful.I) == ()
    @test typeof(spinful.I).parameters[6] === ProductSymm{Tuple{}}
    @test Telum.stored_qlabels(spinful.I) == [((), ())]
    @test Telum.spaces(spinful.I) == ([(() , 4)], [(() , 4)])
    @test _test_sector_rmt(spinful.I, 1) ≈ Matrix{Float64}(I, 4, 4)
    @test size(_test_sector_rmt(spinful.Fu, 1)) == (4, 4)
    @test size(_test_sector_rmt(spinful.Fd, 1)) == (4, 4)
    @test size(_test_sector_rmt(spinful.Z, 1)) == (4, 4)
end

struct NonCommutingSymmetryOptions <: LocalSpaceOptions end

function Telum.getSymmetryInfo(::NonCommutingSymmetryOptions)
    symm = (U1, SU{2})
    weights = ([(1,), (-1,)], [(1,), (-1,)])
    lowering_ops = (Matrix{Int}[], [sparse([0 0; 1 0])])

    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    mwirops[:I] = spdiagm(0 => ones(Float64, 2))
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
