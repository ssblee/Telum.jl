@testset "lock reduction in contract" begin
    test_lock_reduce(FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "auto contract requires matching spaces" begin
    test_contract_requires_matching_spaces_in_star(
        FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "contract abelian w-matrices stay unit" begin
    test_contract_abelian_wmats_are_unit(FermionSOptions(1, :U1, :SU2, nothing))
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
    @test qi1 * a isa TLArray
end

@testset "getIdentity direct contraction" begin
    test_getIdentity_direct_contract(FermionSOptions(1, :U1, :SU2, nothing))
    test_getIdentity_direct_contract(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "cgt_metadata test" begin
    ql1 = ((6,), (7,), (2,), (3,))
    ql2 = ((2,), (3,), (4,), (7,), (8,))
    ld1 = (2, 2)
    ld2 = (3, 2)
    cp1 = (4, 1, 3, 2)
    cp2 = (3, 5, 1, 2, 4)
    free1 = [1, 2]
    legs1 = (3, 4)
    free2 = [1, 2, 4]
    legs2 = (3, 5)

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
    test_contract_xsym_wmat()
    test_accumulate_mkl_matches_generic_all_orders()
    test_qr_shared_isometry_rank1_fastpath()
    test_qr_shared_isometry_rank3_splits_factors()
    test_compress_sector(2, 1, 3; verbose=false)
    test_compress_sector(2, 7, 3; verbose=false)
    test_compress_sector(3, 5, 4; verbose=false)
    test_compress_sector_zero_wmat_shortcircuits()
    test_contract_compress_sector_rmt_optimizer()
    test_diag_rmt_storage_and_prepared_cache()
    test_contract_compress_sector_diag_rmt()
    test_contract_diag_rmt_tlarray()
    test_diag_rmt_producers_and_metadata_ops()
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
