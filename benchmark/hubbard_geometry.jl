"""ZigZag ordering and Hubbard interaction map ported from IntrcMapSqLatt.m."""
function zigzag_order(Lx::Int, Ly::Int)
    Lx > 0 && Ly > 0 || throw(ArgumentError("lattice dimensions must be positive"))
    # MATLAB reshape(1:L, [Ly,Lx]) is column-major.
    return reshape(collect(1:Lx*Ly), Ly, Lx)
end

function hubbard_interaction_map(Lx::Int, Ly::Int; BCX::Symbol=:OBC,
                                BCY::Symbol=:OBC, t::Real=1, tp::Real=0)
    BCX in (:OBC, :PBC) || throw(ArgumentError("BCX must be :OBC or :PBC"))
    BCY in (:OBC, :PBC) || throw(ArgumentError("BCY must be :OBC or :PBC"))
    ord = zigzag_order(Lx, Ly)
    weights = Dict{Tuple{Int,Int},Float64}()
    addedge(a,b,w) = a == b ? nothing : (weights[minmax(a,b)] = Float64(w))

    # Nearest-neighbor vertical and horizontal bonds.
    for x in 1:Lx, y in 1:Ly-1
        addedge(ord[y,x], ord[y+1,x], t)
    end
    for x in 1:Lx-1, y in 1:Ly
        addedge(ord[y,x], ord[y,x+1], t)
    end
    if BCX === :PBC
        for y in 1:Ly
            addedge(ord[y,Lx], ord[y,1], t)
        end
    end
    if BCY === :PBC
        for x in 1:Lx
            addedge(ord[Ly,x], ord[1,x], t)
        end
    end

    # Next-nearest-neighbor plaquette diagonals, including periodic images.
    if !iszero(tp)
        for x in 1:Lx-1, y in 1:Ly-1
            addedge(ord[y,x], ord[y+1,x+1], tp)
            addedge(ord[y+1,x], ord[y,x+1], tp)
        end
        if BCX === :PBC
            for y in 1:Ly-1
                addedge(ord[y,Lx], ord[y+1,1], tp)
                addedge(ord[y+1,Lx], ord[y,1], tp)
            end
        end
        if BCY === :PBC
            for x in 1:Lx-1
                addedge(ord[Ly,x], ord[1,x+1], tp)
                addedge(ord[1,x], ord[Ly,x+1], tp)
            end
        end
        if BCX === :PBC && BCY === :PBC
            addedge(ord[Ly,Lx], ord[1,1], tp)
            addedge(ord[1,Lx], ord[Ly,1], tp)
        end
    end
    return (; order=ord, bonds=sort!([(i,j,w) for ((i,j),w) in weights]))
end
