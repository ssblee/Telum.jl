function _assert_qlabel_storage(q::TLArray)
    @test size(q.qlabels) == (ndims(q), Telum.nsectors(q))
    for sector_index in 1:Telum.nsectors(q)
        @test Tuple(q.qlabels[:, sector_index]) ==
              ntuple(leg -> Telum.sector_qlabel(q, sector_index, leg), ndims(q))
    end
end

function _assert_wmat_storage(q::TLArray)
    nonabelian = Telum.nonabelian_symmetry_indices(Telum.productsymm(q))
    @test size(q.wmats) == (length(nonabelian), Telum.nsectors(q))
    for sector_index in 1:Telum.nsectors(q), n in 1:length(symm(q))
        wmat = Telum.sector_wmat(q, sector_index, n)
        if n in nonabelian
            slot = Telum.nonabelian_wmat_slot(Telum.productsymm(q), n)
            @test wmat === q.wmats[slot, sector_index]
        else
            @test size(wmat) == (1, 1)
            @test wmat[1] == 1.0
        end
    end
end

function _assert_metadata_inferred(q::TLArray)
    Telum.nsectors(q) == 0 && return
    @inferred Telum.sector_qlabel(q, 1, 1)
    @inferred Telum.sector_wmat(q, 1, 1)
    @inferred Telum._sector_cgt_metadata(q, 1, 1)
end

@testset "TLArray qlabel storage orientation" begin
    fermion_u1 = getLocalSpace(FermionOptions(U1))
    fermion_su2 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing), ("in", "out", "op"))

    samples = (
        fermion_u1.I,
        fermion_u1.F,
        fermion_su2.I,
        fermion_su2.F,
        empty_qspace((U1, SU{2}), (TLIndex("empty_in", '+'), TLIndex("empty_out", '-'))),
        getvac(fermion_su2.I),
    )

    @testset "storage columns are sectors" begin
        for q in samples
            _assert_qlabel_storage(q)
            _assert_wmat_storage(q)
        end
    end

    @testset "metadata helpers infer and preserve ordering" begin
        for q in samples
            _assert_metadata_inferred(q)
            for sector_index in 1:Telum.nsectors(q), n in 1:length(symm(q))
                qlabels, cgp, legdir = Telum._sector_cgt_metadata(q, sector_index, n)
                stored_to_phys = Telum._stored_leg_order(q, sector_index, n)
                @test qlabels == ntuple(i -> Telum.sector_qlabel(q, sector_index, stored_to_phys[i])[n], ndims(q))
                @test cgp == ntuple(leg -> findfirst(==(leg), stored_to_phys), ndims(q))
                @test legdir == (count(leg -> q.inds[leg].dir == '+', 1:ndims(q)),
                                 count(leg -> q.inds[leg].dir == '-', 1:ndims(q)))
            end
        end
    end

    @testset "field operations keep orientation" begin
        q = fermion_su2.F
        _assert_qlabel_storage(copy(q))
        _assert_qlabel_storage(TLArray(q, :))
        _assert_qlabel_storage(permutedims(q, (2, 1, 3)))
        _assert_qlabel_storage(conj(q))
        _assert_qlabel_storage(q + q)
        @test norm(q + q) ≈ 2norm(q)

        sorted = copy(q)
        Telum.sort_sectors!(sorted)
        _assert_qlabel_storage(sorted)
    end

    @testset "decomposition assembly keeps orientation" begin
        U, S, Vd = svd(fermion_su2.F, (1,))
        _assert_qlabel_storage(U)
        _assert_qlabel_storage(S)
        _assert_qlabel_storage(Vd)

        eres = eigen(TLArray(fermion_su2.I, ("eig", "eig")); hermitian=true)
        _assert_qlabel_storage(eres.V)
        _assert_qlabel_storage(eres.D)
    end
end
