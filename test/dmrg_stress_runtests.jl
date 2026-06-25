using Test

include("common.jl")
include("support/dmrg.jl")

function _dmrg_stress_fixture(; nsites::Int=20, nkeep::Int=30, nsweep::Int=1)
    mpo = HubbardMPO(4.0, 1.5, 1.0, nsites)
    mps, energy = do_dmrg_result(mpo, nkeep, nkeep, nsweep; time_blocks=false)
    return (; mpo, mps, energy, bond=nsites ÷ 2, nkeep)
end

function _required_leg(q::AbstractTLArray; itag::AbstractString)
    leg = findleg(q; itag)
    @test !isnothing(leg)
    return leg
end

function _largest_rmt_length(q::TLArray)
    lengths = Int[]
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        push!(lengths, length(Telum.sector_rmt(q, sector)))
    end
    return isempty(lengths) ? 0 : maximum(lengths)
end

@testset "DMRG large-RMT contraction and SVD stress" begin
    fixture = _dmrg_stress_fixture()
    mps = fixture.mps
    bond = fixture.bond

    dimensions, _ = central_bond_rmt_dimensions(mps; bond)
    @test sum(dimensions) == fixture.nkeep
    @test !isempty(dimensions)

    left = mps[bond]
    right = mps[bond + 1]
    left_bond_leg = _required_leg(left; itag="SB,$bond")
    right_bond_leg = _required_leg(right; itag="SB,$bond")

    two_site = _test_contract_matches_sparse_and_preserves_inputs(
        left, (left_bond_leg,), right, (right_bond_leg,); atol=1e-9, rtol=1e-9)
    @test ndims(two_site) == 4
    @test _largest_rmt_length(two_site) >= fixture.nkeep

    next_site = mps[bond + 2]
    two_site_next_leg = _required_leg(two_site; itag="SB,$(bond + 1)")
    next_site_leg = _required_leg(next_site; itag="SB,$(bond + 1)")
    three_site = _test_contract_matches_sparse_and_preserves_inputs(
        two_site, (two_site_next_leg,), next_site, (next_site_leg,); atol=1e-9, rtol=1e-9)
    @test ndims(three_site) == 5
    @test _largest_rmt_length(three_site) >= fixture.nkeep

    test_svdQS(two_site, [1, 2]; cutoff=1e-12, tol=1e-8, verbose=false)
    test_svdQS(three_site, [1, 2, 3]; cutoff=1e-12, tol=1e-8, verbose=false)
end
