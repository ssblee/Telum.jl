"""Independent fixed-particle-number Fock-basis Hubbard Hamiltonian for validation."""
function hubbard_ed_matrix(nsite::Int, bonds, U::Real, Ntot::Int; mu::Real=0,
                           sz2::Union{Nothing,Int}=nothing)
    0 <= Ntot <= 2nsite || throw(ArgumentError("invalid particle number"))
    states = UInt(0):UInt((UInt(1) << (2nsite))-1)
    upmask = foldl(|, (UInt(1) << (2site-2) for site in 1:nsite); init=UInt(0))
    dnmask = foldl(|, (UInt(1) << (2site-1) for site in 1:nsite); init=UInt(0))
    basis = UInt[s for s in states if count_ones(s) == Ntot &&
        (sz2 === nothing ||
         count_ones(s & upmask) - count_ones(s & dnmask) == sz2)]
    index = Dict(s => i for (i,s) in enumerate(basis))
    H = zeros(Float64, length(basis), length(basis))
    paritymask(i) = (UInt(1) << (i-1))-1
    function apply_c(s::UInt, mode::Int)
        bit = UInt(1) << (mode-1)
        (s & bit) == 0 && return nothing
        return (s ⊻ bit, isodd(count_ones(s & paritymask(mode))) ? -1.0 : 1.0)
    end
    function apply_cd(s::UInt, mode::Int)
        bit = UInt(1) << (mode-1)
        (s & bit) != 0 && return nothing
        return (s | bit, isodd(count_ones(s & paritymask(mode))) ? -1.0 : 1.0)
    end
    for (col,s) in enumerate(basis)
        diag = -Float64(mu)*count_ones(s)
        for site in 1:nsite
            up = (s >> (2site-2)) & 1
            dn = (s >> (2site-1)) & 1
            diag += Float64(U)*up*dn
        end
        H[col,col] = diag
        for (i,j,t) in bonds, spin in 0:1
            # Add both orientations of c†ᵢ cⱼ + h.c. using explicit Fock signs.
            for (dst,src) in ((i,j),(j,i))
                a = apply_c(s, 2src-1+spin)
                a === nothing && continue
                b = apply_cd(a[1], 2dst-1+spin)
                b === nothing && continue
                H[index[b[1]],col] += -Float64(t)*a[2]*b[2]
            end
        end
    end
    return Hermitian(H), basis
end

function hubbard_ed_ground_energy(nsite::Int, bonds, U::Real, Ntot::Int; mu::Real=0,
                                  sz2::Union{Nothing,Int}=nothing)
    H, basis = hubbard_ed_matrix(nsite, bonds, U, Ntot; mu, sz2)
    return eigmin(H), basis
end
