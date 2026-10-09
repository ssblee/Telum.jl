"""Build local `Acharge,SU2spin` operators used by the Hubbard Hamiltonian MPO.

The file is intentionally organized around a symmetry dispatch, matching the
MATLAB initializer's `switch Para.Symm` and leaving a clear place for later
branches. The caller must load Telum before including this file.
"""
function _hubbard_local_space(options::FermionSOptions)
    q = getLocalSpace(options, ("site", "site", "op"))
    symm(q.I) == (U1, SU{2}) || throw(ArgumentError(
        "the Hubbard MPO currently supports one U1-charge × SU2-spin channel; got $(options)"))
    local_dim = sum(dim * Telum.dimension(U1, labels[1]) *
                    Telum.dimension(SU{2}, labels[2])
                    for (labels, dim) in first(Telum.spaces(q.I)))
    local_dim == 4 || throw(ArgumentError(
        "the Hubbard MPO currently expects a four-state spinful site; got local dimension $local_dim"))
    return q
end

function prepare_hubbard_operators(p::HubbardDMRGParameters)
    validate(p)
    options = p.space_options
    q = _hubbard_local_space(options)
    n = lock(q.F', 2) * q.F
    doublon = lock(n - 1, 1) * n / 2
    return (; options, q, identity=q.I, parity=q.Z, annihilation=q.F,
            creation=q.F', number=n, doublon)
end
