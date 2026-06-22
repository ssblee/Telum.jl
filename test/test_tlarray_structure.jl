@testset "empty_qspace" begin
    # ── rank-1 to rank-5 construction ────────────────────────────────────────
    @testset "rank-$QD construction" for QD in 1:5
        symm = (SU{2},)
        inds = ntuple(i -> TLIndex("l$i", i == 1 ? '+' : '-'), QD)
        q = empty_qspace(symm, inds)

        @test isempty(q.qlabels)
        @test isempty(q.RMTs)
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
        @test isempty(q.qlabels)
        @test isempty(q.RMTs)
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

        @test eltype(qf64.RMTs) <: Array{Float64, 3}
        @test eltype(qc64.RMTs) <: Array{ComplexF64, 3}
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
        @test replaceitag(q, "a"=>"x").inds[1].itags  == "x"
        @test setitag(q, "new"; dir='+').inds[1].itags == "new"

        # findlegs / findleg
        @test findlegs(q; dir='+') == [1]
        @test findlegs(q; dir='-') == [2, 3]
        @test findleg(q; dir='+')  == 1
        @test findleg(q; dir='-')  == 2

        # scalar multiplication of empty TLArray produces empty TLArray
        q_scaled = copy(3.0 * q)
        @test isempty(q_scaled.qlabels)
        @test isempty(q_scaled.RMTs)
    end

    @testset "zero preserves metadata on TLArray" begin
        option = FermionSOptions(3, :U1, :SU2, :SU3)
        q0 = getLocalSpace(option)
        q = TLArray(q0.F, ("site1", "site2", "op"))
        qz = zero(q)

        @test qz isa TLArray
        @test isempty(qz.qlabels)
        @test isempty(qz.RMTs)
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

    @testset "printmeta on TLArray" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        q = TLArray(q0.F, ("site1", "site2", "op"))

        meta = sprint(printmeta, q)
        shown = sprint(show, MIME"text/plain"(), q)

        @test meta == first(split(shown, '\n'))
        @test occursin("3D TLArray", meta)
        @test occursin("site1", meta)
        @test !occursin('\n', meta)
        @test !occursin("1.\t", meta)
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
    @test !(:symm in fieldnames(typeof(q_empty)))
    @test productsymm(q_empty) == expected_ps
    @test productsymm(symm(q_empty)) == expected_ps
    @test symm(q_empty) == (U1, SU{3})
    @test @inferred(symm(q_empty)) == (U1, SU{3})
    @test product_symms(q_empty) == (U1, SU{3})
    @test nsymms(q_empty) == 2
    @test eltype(q_empty.spaces[1]) == Tuple{expected, Int}

    q_multi = empty_qspace((U1, SU{2}, SU{3}), (TLIndex('+'),))
    @test qlabeltype(q_multi) == Tuple{Tuple{Int}, Tuple{Int}, NTuple{2, Int}}
    @test typeof(q_multi).parameters[5] == Tuple{Tuple{Int}, Tuple{Int}, NTuple{2, Int}}
    @test productsymm(q_multi) == ProductSymm{Tuple{U1, SU{2}, SU{3}}}

    q_local = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).I
    @test !(:symm in fieldnames(typeof(q_local)))
    @test eltype(q_local.qlabels) == NTuple{2, qlabeltype(q_local)}
    @test all(q_local.qlabels[i] isa NTuple{2, qlabeltype(q_local)}
              for i in _test_defined_sector_indices(q_local))
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

    row_sector(qs::TLArray, sector_index::Int) = qs.qlabels[sector_index][leg]
    rows_for_sector(qs::TLArray, sector) =
        [i for i in _test_defined_sector_indices(qs) if row_sector(qs, i) == sector]

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
    @test Set(row_sector(q_pred_pairs, i) for i in _test_defined_sector_indices(q_pred_pairs)) ==
          Set([sector_full, sector_reorder])

    full_rows = rows_for_sector(q_pred_pairs, sector_full)
    orig_full_rows = rows_for_sector(q, sector_full)
    @test length(full_rows) == length(orig_full_rows)
    for (full_idx, orig_full_idx) in zip(full_rows, orig_full_rows)
        @test Telum.sector_rmt(q_pred_pairs, full_idx) == Telum.sector_rmt(q, orig_full_idx)
    end

    reorder_rows = rows_for_sector(q_pred_pairs, sector_reorder)
    orig_reorder_rows = rows_for_sector(q, sector_reorder)
    @test length(reorder_rows) == length(orig_reorder_rows)
    reorder_inds = [dim_reorder, 1]
    for (reorder_idx, orig_reorder_idx) in zip(reorder_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        reorder_selector = ntuple(d -> d == leg ? reorder_inds : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_pred_pairs, reorder_idx) == orig_rmt[reorder_selector...]
    end

    q_single = Telum.getsub(q, leg, sector -> sector == sector_reorder ? 2 : nothing)
    single_rows = rows_for_sector(q_single, sector_reorder)
    @test q_single.spaces[leg] == [(sector_reorder, 1)]
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_single.spaces[other_leg] == q.spaces[other_leg]
    end
    @test length(single_rows) == length(orig_reorder_rows)
    for (single_idx, orig_reorder_idx) in zip(single_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        single_selector = ntuple(d -> d == leg ? [2] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_single, single_idx) == orig_rmt[single_selector...]
    end

    q_range = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (1:2) : nothing)
    @test q_range.spaces[leg] == [(sector_reorder, 2)]
    range_rows = rows_for_sector(q_range, sector_reorder)
    @test length(range_rows) == length(orig_reorder_rows)
    for (range_idx, orig_reorder_idx) in zip(range_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        range_selector = ntuple(d -> d == leg ? [1, 2] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_range, range_idx) == orig_rmt[range_selector...]
    end

    q_negative = Telum.getsub(q, leg, sector -> sector == sector_reorder ? -1 : nothing)
    negative_rows = rows_for_sector(q_negative, sector_reorder)
    @test q_negative.spaces[leg] == [(sector_reorder, 1)]
    @test length(negative_rows) == length(orig_reorder_rows)
    for (negative_idx, orig_reorder_idx) in zip(negative_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        negative_selector = ntuple(d -> d == leg ? [dim_reorder] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_negative, negative_idx) == orig_rmt[negative_selector...]
    end

    q_negative_range = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (-2:-1) : nothing)
    negative_range_rows = rows_for_sector(q_negative_range, sector_reorder)
    @test q_negative_range.spaces[leg] == [(sector_reorder, 2)]
    @test length(negative_range_rows) == length(orig_reorder_rows)
    for (negative_range_idx, orig_reorder_idx) in zip(negative_range_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        negative_range_selector = ntuple(d -> d == leg ? [dim_reorder - 1, dim_reorder] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_negative_range, negative_range_idx) == orig_rmt[negative_range_selector...]
    end

    q_mixed = Telum.getsub(q, leg, sector -> sector == sector_reorder ? [-1, 1] : nothing)
    mixed_rows = rows_for_sector(q_mixed, sector_reorder)
    @test q_mixed.spaces[leg] == [(sector_reorder, 2)]
    @test length(mixed_rows) == length(orig_reorder_rows)
    for (mixed_idx, orig_reorder_idx) in zip(mixed_rows, orig_reorder_rows)
        orig_rmt = Telum.sector_rmt(q, orig_reorder_idx)
        mixed_selector = ntuple(d -> d == leg ? [dim_reorder, 1] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_mixed, mixed_idx) == orig_rmt[mixed_selector...]
    end

    q_tuple_pick = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (dim_reorder, 1) : nothing)
    @test q_tuple_pick.spaces == q_mixed.spaces
    tuple_pick_rows = rows_for_sector(q_tuple_pick, sector_reorder)
    @test length(tuple_pick_rows) == length(mixed_rows)
    for (tuple_pick_idx, mixed_idx) in zip(tuple_pick_rows, mixed_rows)
        @test Telum.sector_rmt(q_tuple_pick, tuple_pick_idx) == Telum.sector_rmt(q_mixed, mixed_idx)
    end

    q_empty = Telum.getsub(q, leg, _ -> nothing)
    @test isempty(_test_defined_sector_indices(q_empty))
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

    row_sector(qs::TLArray, sector_index::Int) = qs.qlabels[sector_index][leg]
    expected_rows = [i for i in _test_defined_sector_indices(q) if row_sector(q, i) == target_sector]
    expected_leg_spaces = [entry for entry in q.spaces[leg] if entry[1] == target_sector]

    q_exact = Telum.getsub(q, leg, sector -> sector == target_sector ? Colon() : nothing)
    _test_tlarrays_same_sector_payloads(q_exact, TLArray(q, expected_rows))
    @test q_exact.spaces[leg] == expected_leg_spaces
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_exact.spaces[other_leg] == q.spaces[other_leg]
    end

    q_component = Telum.getsub(q, leg, sector -> sector[1] == target_sector[1] ? Colon() : nothing)
    expected_component_rows = [i for i in _test_defined_sector_indices(q) if row_sector(q, i)[1] == target_sector[1]]
    expected_component_spaces = [entry for entry in q.spaces[leg] if entry[1][1] == target_sector[1]]
    _test_tlarrays_same_sector_payloads(q_component, TLArray(q, expected_component_rows))
    @test q_component.spaces[leg] == expected_component_spaces

    q_preserved = Telum.getsub(q, leg, sector -> sector == target_sector ? Colon() : nothing; preserve_space=true)
    _test_tlarrays_same_sector_payloads(q_preserved, TLArray(q, expected_rows))
    @test q_preserved.spaces == q.spaces
    @test all(q_preserved.spaces[legidx] !== q.spaces[legidx] for legidx in 1:length(q.spaces))

    q_none = Telum.getsub(q, leg, _ -> nothing)
    @test isempty(_test_defined_sector_indices(q_none))
    @test isempty(q_none.spaces[leg])
    for other_leg in 1:length(q.spaces)
        other_leg == leg && continue
        @test q_none.spaces[other_leg] == q.spaces[other_leg]
    end

    q_none_preserved = Telum.getsub(q, leg, _ -> nothing; preserve_space=true)
    @test isempty(_test_defined_sector_indices(q_none_preserved))
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
    @test !isempty(_test_defined_sector_indices(q))

    row_sector_at(sector_index, leg) = q.qlabels[sector_index][leg]
    first_sector = first(_test_defined_sector_indices(q))
    allowed = Set{Any}([row_sector_at(first_sector, 1), row_sector_at(first_sector, 2)])
    pred = sector -> sector in allowed ? Colon() : nothing

    expected_rows = [
        i for i in _test_defined_sector_indices(q)
        if all(row_sector_at(i, leg) in allowed for leg in legs)
    ]
    expected_spaces_1 = [entry for entry in q.spaces[1] if entry[1] in allowed]
    expected_spaces_2 = [entry for entry in q.spaces[2] if entry[1] in allowed]

    q_multi = Telum.getsub(q, legs, pred)
    _test_tlarrays_same_sector_payloads(q_multi, TLArray(q, expected_rows))
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
    @test length(_test_defined_sector_indices(q_multi_sliced)) == length(expected_rows)
    for (sliced_idx, orig_idx) in zip(_test_defined_sector_indices(q_multi_sliced), expected_rows)
        orig_rmt = Telum.sector_rmt(q, orig_idx)
        slice_selector = ntuple(d -> d in legs ? [1] : Colon(), ndims(orig_rmt))
        @test Telum.sector_rmt(q_multi_sliced, sliced_idx) == orig_rmt[slice_selector...]
    end

    q_multi_range = Telum.getsub(q, 1:2, pred)
    _test_tlarrays_same_sector_payloads(q_multi_range, q_multi)
    @test q_multi_range.spaces == q_multi.spaces

    q_multi_preserved = Telum.getsub(q, legs, pred; preserve_space=true)
    _test_tlarrays_same_sector_payloads(q_multi_preserved, TLArray(q, expected_rows))
    @test q_multi_preserved.spaces == q.spaces

    q_multi_kw = Telum.getsub(q, pred; itag="sel")
    _test_tlarrays_same_sector_payloads(q_multi_kw, q_multi)
    @test q_multi_kw.spaces == q_multi.spaces

    q_multi_kw_preserved = Telum.getsub(q, pred; itag="sel", preserve_space=true)
    _test_tlarrays_same_sector_payloads(q_multi_kw_preserved, q_multi)
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
        @test length(_test_defined_sector_indices(vac)) == 1
        @test vac.inds == (TLIndex("", '+'), TLIndex("", '-'))
        @test length(vac.spaces[1]) == 1
        @test length(vac.spaces[2]) == 1

        trivial = zero_qlabels(vac)
        @test vac.spaces[1][1] == (trivial, 1)
        @test vac.spaces[2][1] == (trivial, 1)

        sector_index = only(_test_defined_sector_indices(vac))
        rmt = Telum.sector_rmt(vac, sector_index)
        @test size(rmt) == ntuple(_ -> 1, length(symm(vac)) + 2)
        @test only(rmt) == one(eltype(rmt))
        for n in 1:length(symm(vac))
            @test vac.qlabels[sector_index][1][n] == trivial[n]
            @test vac.qlabels[sector_index][2][n] == trivial[n]
            wmat = Telum.sector_wmat(vac, sector_index, n)
            @test size(wmat) == (1, 1)
            @test wmat[1] == 1.0
        end
    end

    @testset "optional tags are applied" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        vac = getvac(q0.F, ("vin", "vout"))

        @test vac.inds[1] == TLIndex("vin", '+')
        @test vac.inds[2] == TLIndex("vout", '-')
    end
end

@testset "addSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F
    q_rank2 = TLArray(q0.I, ("lin", "lout"))

    q_default = addSingleton(q, 2)
    @test q_default.inds[1] == q.inds[1]
    @test q_default.inds[2] == TLIndex("", '+')
    @test q_default.inds[3] == q.inds[2]
    @test q_default.inds[4] == q.inds[3]

    trivial = zero_qlabels(q)
    @test q_default.spaces[2] == [(trivial, 1)]

    q_append_default = addSingleton(q)
    @test q_append_default.inds[1:3] == q.inds
    @test q_append_default.inds[4] == TLIndex("", '+')
    @test q_append_default.spaces[4] == [(trivial, 1)]
    @test Array(to_sparse_array(q_append_default)) ==
          Array(to_sparse_array(addSingleton(q, 4)))

    q_append_two = addSingleton(q; nlegs=2,
                                itag=("tail_left", "tail_right"),
                                plev=(1, 2),
                                lock=(0, 1),
                                dir=('+', '-'))
    @test q_append_two.inds[1:3] == q.inds
    @test q_append_two.inds[4] == TLIndex("tail_left", '+', 1, 0)
    @test q_append_two.inds[5] == TLIndex("tail_right", '-', 2, 1)
    @test q_append_two.spaces[4] == [(trivial, 1)]
    @test q_append_two.spaces[5] == [(trivial, 1)]
    @test Array(to_sparse_array(q_append_two)) ==
          Array(to_sparse_array(addSingleton(q, (4, 5);
                                             itag=("tail_left", "tail_right"),
                                             plev=(1, 2),
                                             lock=(0, 1),
                                             dir=('+', '-'))))

    @test_throws ArgumentError addSingleton(q; nlegs=0)

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
    @test Telum.sector_wmat(q_rank2, 1, 2) == Telum.sector_wmat(q_rank2_added, 1, 2)
    @test pointer(Telum.sector_rmt(q_rank2, 1)) ==
          pointer(Telum.sector_rmt(q_rank2_added, 1))
end

@testset "deleteSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F
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
    @test Telum.sector_wmat(q_rank2_added, 1, 2) == Telum.sector_wmat(q_rank2_roundtrip, 1, 2)
    @test pointer(Telum.sector_rmt(q_rank2_added, 1)) ==
          pointer(Telum.sector_rmt(q_rank2_roundtrip, 1))

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
    q2 = TLArray(q0.F, ("l2_in", "l2_out", "l2_op"))

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
    _test_tlarrays_same_sector_payloads(idq, idq_pairs)
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
        @test !isempty(_test_defined_sector_indices(q4))
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
        ref = svd(q4, (1, 2, 3), "kwL", "kwR")
        kw = svd(q4, "kwL", "kwR"; dir='+')
        U_ref, S_ref, Vd_ref = ref.U, ref.S, ref.Vd
        U_kw, S_kw, Vd_kw = kw.U, kw.S, kw.Vd

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
        @test length(_test_defined_sector_indices(copy(q_scaled))) == length(_test_defined_sector_indices(q4))
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
