function _assert_tlarray_structural_equal(a::TLArray, b::TLArray)
    @test symm(a) == symm(b)
    @test a.inds == b.inds
    @test a.spaces == b.spaces
    @test a.qlabels == b.qlabels
    @test a.isdefined == b.isdefined
    @test a.iszero == b.iszero
    @test a.wmatdata ≈ b.wmatdata
    @test a.wmatinfo == b.wmatinfo
    @test typeof(a.RMTs) == typeof(b.RMTs)

    for sector in Telum.sector_slots(a)
        if a.iszero[sector]
            @test !isassigned(b.RMTs, sector)
            continue
        end
        @test isassigned(b.RMTs, sector)
        ar = Telum.sector_rmt(a, sector)
        br = Telum.sector_rmt(b, sector)
        if ar isa DiagRMT
            @test br isa DiagRMT
            @test ar.axis == br.axis
            @test ar.diag ≈ br.diag
        else
            @test ar ≈ br
        end
    end
end

function _assert_tlarray_roundtrip(a::TLArray, b::TLArray; tol=1e-12)
    _assert_tlarray_structural_equal(a, b)
    @test norm(a - b) / max(norm(a), 1.0) < tol
end

function _with_saved_tlarray(f, q; name="tensor")
    path = tempname() * ".h5"
    try
        save_tlarray(path, q; name)
        loaded = load_tlarray(path; name)
        f(loaded)
    finally
        isfile(path) && rm(path)
    end
end

function _with_zero_sector(q::TLArray)
    qlabels = copy(q.qlabels)
    push!(qlabels, first(q.qlabels))

    wmatdata = copy(q.wmatdata)
    wmatinfo = copy(q.wmatinfo)
    push!(wmatinfo, Telum._empty_wmat_info(Val(length(first(q.wmatinfo)))))

    RMTs = similar(q.RMTs, length(q.RMTs) + 1)
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        RMTs[sector] = deepcopy(Telum.sector_rmt(q, sector))
    end

    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, q.inds, q.spaces)
end

@testset "HDF5 TLArray round trip" begin
    q0 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing), ("left", "right", "op"))
    q = q0.F

    q_u1 = getLocalSpace(FermionOptions(U1), ("in", "out", "op")).F
    _with_saved_tlarray(q_u1) do loaded
        _assert_tlarray_roundtrip(q_u1, loaded)
    end

    _with_saved_tlarray(q) do loaded
        _assert_tlarray_roundtrip(q, loaded)
    end

    q_zero = _with_zero_sector(q)
    @test q_zero.iszero[end]
    @test !q_zero.isdefined[end]
    _with_saved_tlarray(q_zero) do loaded
        _assert_tlarray_roundtrip(q_zero, loaded)
        @test loaded.qlabels[end] == q_zero.qlabels[end]
        @test loaded.iszero[end]
        @test !loaded.isdefined[end]
        @test !isassigned(loaded.RMTs, length(loaded.RMTs))
    end

    first_active = first(sector for sector in Telum.sector_slots(q) if !q.iszero[sector])
    complex_RMT_type = Array{ComplexF64, ndims(Telum.sector_rmt(q, first_active))}
    complex_RMTs = Vector{complex_RMT_type}(undef, length(q.RMTs))
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        complex_RMTs[sector] = ComplexF64.(Telum.sector_rmt(q, sector))
    end
    q_complex = TLArray(symm(q), copy(q.qlabels), copy(q.wmatdata), copy(q.wmatinfo),
                        complex_RMTs, q.inds, q.spaces)
    _with_saved_tlarray(q_complex) do loaded
        _assert_tlarray_roundtrip(q_complex, loaded)
    end

    q_view = permutedims(q, (2, 1, 3))
    q_view_eager = copy(q_view)
    _with_saved_tlarray(q_view) do loaded
        _assert_tlarray_roundtrip(q_view_eager, loaded)
    end

    q_diag = get1jtensor(q0.I, 1)
    @test typeof(q_diag.RMTs) <: Vector{<:DiagRMT}
    _with_saved_tlarray(q_diag) do loaded
        _assert_tlarray_roundtrip(q_diag, loaded)
    end
end
