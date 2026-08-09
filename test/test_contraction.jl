@testset "lock reduction in contract" begin
    test_lock_reduce(FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "auto contract requires matching spaces" begin
    test_contract_requires_matching_spaces_in_star(
        FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "contract X-symbol cache shares duplicated non-Abelian symmetry" begin
    option = FermionSOptions(1, :SU2, :SU2, nothing)
    q = getLocalSpace(option)
    PS = Telum.productsymm(q.I)
    QT = qlabeltype(q.I)
    caches = Telum._new_xsym_caches(PS, QT, Val(1), Val(1), Val(1))

    @test length(caches) == 1
    @test @inferred(Telum._xsym_cache_slot(PS, Val(1))) == 1
    @test @inferred(Telum._xsym_cache_slot(PS, Val(2))) == 1
    @test Telum._xsym_cache_for(SU{2}, caches, PS, Val(1)) ===
          Telum._xsym_cache_for(SU{2}, caches, PS, Val(2))

    qi1 = TLArray(q.I, ("dup1", "dup1"))
    qi2 = TLArray(q.I, ("dup2", "dup2"))
    a = getIdentity((qi1, 2), (qi2, 2))
    @test contract(qi1, (2,), a, (1,); lazy=false) isa TLArray
end

@testset "getIdentity direct contraction" begin
    test_getIdentity_direct_contract(FermionSOptions(1, :U1, :SU2, nothing))
    test_getIdentity_direct_contract(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contract sparse equivalence edge cases" begin
    test_contract_sparse_equivalence_diag_rmt()
end

@testset "sum sparse equivalence mixed RMT and eltypes" begin
    test_sum_sparse_equivalence_mixed_rmt_eltypes()
end

@testset "no-symmetry contraction and sum" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))
    contracted = contract(left, (2,), right, (1,); lazy=false)
    @test _test_sector_rmt(contracted, 1) ≈ _test_sector_rmt(q.I, 1) * _test_sector_rmt(q.Sz, 1)

    summed = q.Sz + q.Sz
    @test norm(summed - 2 * q.Sz) < 1e-12
    tuple_sum = sum((q.Sz, q.Sz, -q.Sz))
    @test norm(tuple_sum - q.Sz) < 1e-12
end

@testset "lazy contraction materialization" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))
    eager = contract(left, (2,), right, (1,); lazy=false)
    lazy = contract(left, (2,), right, (1,))

    @test lazy isa TLArrayContraction
    @test !Telum.is_sector_defined(lazy, 1)
    @test Telum.sector_rmt_axis_dim(lazy, 1, 1) == size(_test_sector_rmt(eager, 1), 1)
    @test_throws ArgumentError Telum.sector_rmt(lazy, 1)
    @test !Telum.is_sector_defined(lazy, 1)

    materialized = Telum.materialize(lazy)
    @test materialized === lazy
    @test Telum.is_sector_defined(lazy, 1)
    @test _test_sector_rmt(materialized, 1) ≈ _test_sector_rmt(eager, 1)
    concrete = copy(lazy)
    @test concrete isa TLArray
    @test _test_sector_rmt(concrete, 1) ≈ _test_sector_rmt(eager, 1)
end

@testset "lazy getsub on contraction" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))

    lazy_full = contract(left, (2,), right, (1,))
    sub_full = Telum.getsub(lazy_full, 1, _ -> Colon())

    @test sub_full isa Telum.SubTLArray
    @test !Telum.is_sector_defined(lazy_full, 1)
    @test sub_full.source_sectors == [1]
    @test_throws ArgumentError Telum.sector_rmt(sub_full, 1)

    Telum.compute_sectors(sub_full, [1])
    @test Telum.is_sector_defined(lazy_full, 1)
    @test Telum.is_sector_defined(sub_full, 1)
    @test sub_full.RMTs[1] === lazy_full.RMTs[1]

    lazy_slice = contract(left, (2,), right, (1,))
    sub_slice = Telum.getsub(lazy_slice, 1, _ -> 1)
    Telum.compute_sectors(sub_slice, [1])
    slice_selector = ntuple(d -> d == 1 ? [1] : Colon(), ndims(lazy_slice.RMTs[1]))
    @test sub_slice.RMTs[1] == lazy_slice.RMTs[1][slice_selector...]
    @test sub_slice.RMTs[1] !== lazy_slice.RMTs[1]

    lazy_view = 2 * contract(left, (2,), right, (1,))
    sub_view = Telum.getsub(lazy_view, 1, _ -> Colon())
    @test sub_view isa Telum.SubTLArray
    @test sub_view.scale == 2
end

@testset "TLArray conversion aliases materialized contraction storage" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))
    lazy = contract(left, (2,), right, (1,))

    converted = TLArray(lazy)

    @test converted isa TLArray
    @test Telum.is_sector_defined(lazy, 1)
    @test converted.qlabels === lazy.qlabels
    @test converted.wmatdata === lazy.wmatdata
    @test converted.wmatinfo === lazy.wmatinfo
    @test converted.RMTs === lazy.RMTs
    @test converted.isdefined === lazy.isdefined
    @test converted.iszero === lazy.iszero
    @test converted.RMTs[1] === lazy.RMTs[1]

    copied = Telum.to_concrete(lazy)
    @test copied.RMTs !== lazy.RMTs
    @test copied.RMTs[1] !== lazy.RMTs[1]
    @test _test_sector_rmt(converted, 1) ≈ _test_sector_rmt(copied, 1)
end

test_contract_tlarrayview_inputs()

@testset "Generating 1jtensor of TLArray test" begin
    test_1jpair(FermionSOptions(1, :U1, :SU2, nothing))
    test_1jpair(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "Keyword get1jtensor and legflip" begin
    test_get1jtensor_and_legflip_keywords(FermionSOptions(1, :U1, :SU2, nothing))
    test_get1jtensor_and_legflip_keywords(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contract verify_legs checks dual" begin
    test_contract_verify_legs_checks_dual(FermionSOptions(1, :U1, :SU2, nothing))
    test_contract_verify_legs_checks_dual(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contraction of TLArray test" begin
    test_FAcont(FermionSOptions(1, :U1, :SU2, nothing))
    test_FAcont(FermionSOptions(3, :U1, :SU2, :SU3))
end
