"""Sparse site transitions in the scalar hopping-MPO automaton."""
struct HubbardMPOSitePlan
    site::Int
    active_before::Vector{Int}
    active_after::Vector{Int}
    starts::Vector{Int}
    ends::Vector{Tuple{Int,Float64}}
    passes::Vector{Int}
end

"""Plan the hopping channels used by the converted `GenPseudoMPO` construction.

An active channel is labeled by the site where its fermion operator started.
It stays active across Jordan–Wigner parity sites until an interaction-map
edge closes it. The Telum materializer assigns the corresponding SU(2) virtual
sectors and local irreducible operator tensors.
"""
function plan_hubbard_mpo(bonds, L::Int)
    1 <= L || throw(ArgumentError("L must be positive"))
    weights = Dict{Tuple{Int,Int},Float64}()
    for (left,right,coupling) in bonds
        1 <= left < right <= L || throw(ArgumentError("expected 1 ≤ left < right ≤ L"))
        weights[(left,right)] = Float64(coupling)
    end
    plans = Vector{HubbardMPOSitePlan}(undef,L)
    for site in 1:L
        before = sort!(unique([start for (start,finish) in keys(weights)
                               if start < site <= finish]))
        after = sort!(unique([start for (start,finish) in keys(weights)
                              if start <= site < finish]))
        starts = site < L && any(start == site && finish > site
                                 for (start,finish) in keys(weights)) ? [site] : Int[]
        ends = sort!([(start,weights[(start,site)]) for start in before
                      if haskey(weights,(start,site))]; by=first)
        passes = intersect(before,after)
        plans[site] = HubbardMPOSitePlan(site,before,after,starts,ends,passes)
    end
    return plans
end
