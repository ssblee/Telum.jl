module Telum

using MKL
using HPTT_jll
using LinearAlgebra
using SparseArrays
using LoopVectorization
using Tullio

import Base: lock, unlock
import LurCGT
import LurCGT: AbelianSymm, NonabelianSymm, Symmetry, SO, SU, Sp, U1, Z
import LurCGT: add_qn, decompose_irop, decompose_space, detect_1j, dimension
import LurCGT: get_CGTom, get_IROP, get_conj_perm, get_dualq
import LurCGT: getNsave_CGTperm, getNsave_CGTSVD, getNsave_Xsymbol
import LurCGT: getNsave_omlist, getNsave_validout, isabelian
import LurCGT: nzops, remove_zeros, totxt, transf_basis!

⊗(a::AbstractMatrix, b::AbstractMatrix) = kron(b, a)
⊗(a::AbstractVector, b::AbstractVector) = kron(b, a)
comm(A, B) = A * B - B * A

include("TLArray.jl")

export TLArray, LurTensor, TLIndex, Itag
export ProductSymm, productsymm, product_symms, symm, nsymms
export LocalSpaceOptions, SpinOptions, FermionOptions, FermionSOptions
export Z, U1, SU, SO, Sp
export getLocalSpace, getSymmetryInfo, getIdentity, get1jtensor, legflip, contract
export discard_eigen
export svd_leg, svd_cgtsvd, get_new_cgp
export empty_qspace, qlabeltype, zero_qlabels, getvac
export addSingleton, deleteSingleton, ⊗
export oplus, getsub
export complete_oplus_matrix
export printmeta
export findlegs, findleg, matchings, matching, unmatchings, unmatching
export contractables, contractable, uncontractables, uncontractable
export prime, setprime, noprime
export lock, lockp, unlock
export additag, removeitag, replaceitag, setitag

end
