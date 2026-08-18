# Tensor operations

Tensor-network algorithms need more than tensor construction and contraction.
This page introduces Telum’s decomposition, structural, and leg-selection
operations.

```@repl tensor_operations
using Telum

zero_qlabels((U1, SU{2}))
```

```@repl tensor_operations
using LurCGT
using Telum
using LinearAlgebra

option = FermionSOptions(1, :U1, :SU2, nothing);
q = getLocalSpace(option);
```

# Space lists

Each tensor leg stores an explicit list of symmetry spaces and their
multiplicities. This plays a role similar to the QN block structure of an
ITensor, but the space list belongs to each Telum leg and is checked during
contraction. A contraction is allowed only when the corresponding space lists
match.

This extra metadata also makes operations such as eigendecomposition and
`oplus` unambiguous.

```@repl tensor_operations
q.I
```


Consider a single-channel spinful fermionic system. Its four-dimensional local
space is divided into three symmetry sectors. The following code displays the
space list of the first leg of `q.I`.

There are three symmetry sectors. In each tuple, the first entry is a q-label
and the second is its multiplicity, which gives the corresponding RMT size.

```@repl tensor_operations
q.I.spaces[1]
```


The spin IROP has nonzero matrix elements in only some sectors, but its space
lists still record all three sectors.

```@repl tensor_operations
println(q.S)
q.S.spaces[1]
```


The third leg has only one sector.

```@repl tensor_operations
q.S.spaces[3]
```


# Conj

Use Julia’s adjoint syntax (`'`) to construct a tensor with inverted leg
directions and complex-conjugated RMT data.

```@repl tensor_operations
ss = TLArray(q.S, ("site,asdf", "site,zxcv", "op"))
printmeta(ss)
printmeta(ss')
```


# get1jtensor

Use `get1jtensor` to construct the identity-like tensor needed when reversing a
leg direction.

In this example, the second leg of `ss` is selected. One result leg has
`dual=true`, shown in green by the display, so the two legs of `j` are treated
as distinct.

One difference is that the full space list of selected legs is considered. Even when the input tensor has nonzero entries in only some sectors, all sectors contribute to the resulting tensor.

```@repl tensor_operations
println(ss)
j = get1jtensor(ss, 2)
println(j)
println(j.inds[1])
println(j.inds[2])
```


Instead of a leg index, you can give a leg by its properties. The arrow direction of the resulting tensor is opposite to the selected leg.

```@repl tensor_operations
get1jtensor(ss; itag="zxcv")
```


```@repl tensor_operations
get1jtensor(ss; itag="asdf")
```


An error occurs when no legs are selected or when there is ambiguity.

```@repl tensor_operations
get1jtensor(ss; itag="q")
```


```@repl tensor_operations
get1jtensor(ss; itag="site")
```


# legflip

Flip the leg direction of selected leg(s) of a given tensor. Internally, it calls get1jtensor for each leg and contract.

```@repl tensor_operations
ss
```


```@repl tensor_operations
legflip(ss, 3)
legflip(ss; itag="op") # Those two do the same thing
```


```@repl tensor_operations
legflip(ss, (1, 2))
legflip(ss; itag="site")
```


# getIdentity

From the pairs of (tensor, leg index) tuples, the isometry between selected legs and the fused one is constructed. Similar to get1jtensor, the full space lists of selected legs are considered.

![getIdentity_2out.png](assets/getIdentity_2out.png)

```@repl tensor_operations
printmeta(ss)
ss2 = TLArray(q.S, ("aaa", "aaa", "bbb"))
printmeta(ss2)
```


```@repl tensor_operations
getIdentity((ss, 2), (ss2, 2))
```


You can set the property of the fused leg through the keyword arguments. However, the direction is fixed to outgoing.

```@repl tensor_operations
id = getIdentity((ss, 2), (ss2, 2); itag="out", lock=4)
printmeta(id)
```


A single `(tensor, Int)` input is also supported, in which case the outer
parentheses may be omitted.

```@repl tensor_operations
getIdentity((ss, 2))
```


```@repl tensor_operations
getIdentity(ss, 2)
```


If there is only one input tensor, you can call this function with leg indices. However, you cannot select legs by setting keyword arguments.

```@repl tensor_operations
getIdentity(ss, (1, 2); itag="out") # Keyword arguments are for the properties of the fused leg.
```


### What if incoming legs are selected?

When an incoming leg is selected, Telum inserts a 1j tensor to invert the
corresponding output leg.

![getIdentity_1in1out.png](assets/getIdentity_1in1out.png)

```@repl tensor_operations
getIdentity((ss, 2), (ss2, 1))
```


# Legs selection

The functions getIdentity and get1jtensor require leg indices of tensors. However, we don't need to remember the leg order of each tensor; use the functions below.

### findlegs

Return a vector of all leg indices that meet the condition from given keyword arguments. There are 5 available keyword arguments.

dir, plev, lock: Straightforward, select legs with given character or integer.

rev: True or false(default). If it is true, invert the selection.

itag: Select the legs from itag. You can give a tuple/vector of strings. If itag = ("aaa,bbb", "ccc,ddd") is given, legs that have itag with ("aaa" and "bbb") or ("ccc" and "ddd") are selected.

```@repl tensor_operations
printmeta(ss)
findlegs(ss; itag="site")
```


```@repl tensor_operations
findlegs(ss; dir='-')
```


```@repl tensor_operations
findlegs(ss; itag="site", dir='-')
```


```@repl tensor_operations
findlegs(ss; itag="site", dir='-', rev=true)
```


```@repl tensor_operations
findlegs(ss; itag=("site,zxcv", "asdf,op"))
```


```@repl tensor_operations
findlegs(ss; itag=("site,asdf", "op"))
```


```@repl tensor_operations
findlegs(ss; itag="lurlurlur") # Return an empty vector
```


There is a function findleg(without 's'), which returns only the lowest leg index among selected legs. If there is no leg, return nothing.

```@repl tensor_operations
findleg(ss; itag=("site,asdf", "op"))
```


```@repl tensor_operations
findleg(ss; itag="lurlurlur") # return nothing
```

### matchings

Given two tensors A and B, return the leg indices of A that have a matching leg in B. Matching leg means a leg with the same itag, plev, dual & opposite direction. Lock level is ignored.

Similar to findlegs, you can give keyword arguments to give additional conditions. There is also a function 'matching' without 's' that returns only the smallest index.

```@repl tensor_operations
printmeta(ss)
```


```@repl tensor_operations
matchings(ss, ss')
```


```@repl tensor_operations
matchings(ss, ss'; itag="site")
```


```@repl tensor_operations
iii = TLArray(q.I, ("site,asdf", "site,asdf"))
matchings(ss, iii)
```


### contractables

Similar to matchings, but also have a lock condition. The lock levels of the 'matching' legs from both sides should be 0.

Every contractable leg is a matching leg, but the converse does not generally
hold. Keyword arguments are also supported. The singular `contractable`
function returns only one matching index.

```@repl tensor_operations
printmeta(ss)
```


```@repl tensor_operations
matchings(ss, ss')
```


```@repl tensor_operations
matchings(ss, lock(ss', 2))
```


```@repl tensor_operations
contractables(ss, ss')
```


```@repl tensor_operations
contractables(ss, lock(ss', 2)) # 2nd leg: matching, but not contractable due to the lock
```


### unmatchings, uncontractables

For 4 functions matching(s) and contractable(s), there are inverse selection functions whose names are unmatching(s) and uncontractable(s).

```@repl tensor_operations
printmeta(ss)
printmeta(iii)
unmatchings(ss, iii)
```


```@repl tensor_operations
uncontractables(ss, lock(ss', 2))
```


# SVD

`svd` decomposes a tensor across a selected set of left legs, much like choosing
the left indices in an ITensor decomposition. Select any combination of
`1:(rank - 1)` legs; the selected legs belong to `U`, and the remaining legs
belong to `Vd`. There is no restriction on their directions.

Singular values smaller than (maximum value * tol) are automatically truncated. The value of tol is 1e-12 by default.

The result can be truncated by the keyword argument Nkeep.

!!! warning "Version note"
    This syntax may differ from earlier Telum releases.

```@repl tensor_operations
printmeta(ss)
res = svd(ss, (1, 2))
U, S, Vd = res.U, res.S, res.Vd
println("U: ")
println(U)
println("S: ")
println(S)
println("Vd: ")
println(Vd)
```


You can select legs by keyword arguments.

![settag_svd_default.png](assets/settag_svd_default.png)

```@repl tensor_operations
res = svd(ss; itag="site"); # same result as above
U, S, Vd = res.U, res.S, res.Vd
```


There are two more arguments of SVD, which set the tag of the resulting singular value tensor. The default values are 'svdL' and 'svdR', shown in the above result.

![settag_svd_custom.png](assets/settag_svd_custom.png)

```@repl tensor_operations
res = svd(ss, [1, 2], "lll", "rrr")
U, S, Vd = res.U, res.S, res.Vd
println("U: ")
printmeta(U)
println("S: ")
printmeta(S)
println("Vd: ")
printmeta(Vd)
```


```@repl tensor_operations
res = svd(ss, "lll", "rrr"; itag="site")
U, S, Vd = res.U, res.S, res.Vd
println("U: ")
printmeta(U)
println("S: ")
printmeta(S)
println("Vd: ")
printmeta(Vd)
```


To get the list of singular values, add a keyword argument 'get_lists' and access 'kept_list' and 'trunc_list' in the resulting struct.

Each entry consists of 

1. Singular value
2. Its degeneracy
3. q-label of the symmetry sector
4. What rank is the singular value in the symmetry sector?

```@repl tensor_operations
res = svd(ss, [1, 2]; get_lists=true);
```

```@repl tensor_operations
res.kept_list
```


```@repl tensor_operations
res.trunc_list # Empty vector
```


# QR decomposition

`qr` splits a tensor into an isometric left factor and a triangular right factor.
Select the legs that belong to the left factor; Telum creates a shared bond leg
with the supplied tag.

```@repl tensor_operations
Q, R = qr(q.S, (1, 2), "qr");
Q
R
```

The factors can be contracted through their shared `"qr"` leg to recover the
original tensor. As with other leg-selecting operations, the left legs can also
be selected by `itag`, `dir`, `plev`, or `lock` keyword arguments.

# Basic arithmetic

### Add

The addition operator '+' is overloaded. Just enter A + B for two tensors A and B.

If we can permute B so that the indices and space lists of them become identical, B is permuted. The resulting tensor has the same leg order as A. If we cannot determine a unique permutation, an error occurs.

Here is a simple example. You can see a more complex one in the DMRG tutorial.

```@repl tensor_operations
q.I
```


```@repl tensor_operations
q.Z
```


```@repl tensor_operations
q.I + q.Z
```


When adding multiple tensors A, B, C, and D, use sum([A, B, C, D]) that receives a vector or tuple of TLArrays instead of A+B+C+D. The latter generates 2 partial results, A+B and A+B+C, while the former does not.

```@repl tensor_operations
@time q.I + q.I + q.I + q.I # Slow since partial results are generated
```


```@repl tensor_operations
@time sum([q.I, q.I, q.I, q.I])
```


### Multiplication by scalar

```@repl tensor_operations
q.I * 7
```


```@repl tensor_operations
8 * q.I
```


### Special case, addition by scalar

If the given tensor is a block-diagonal square matrix, scalar n is treated as an n * identity matrix. A block-diagonal square matrix means

1. The tensor is rank-2, and has one incoming and one outgoing leg.
2. Two legs have the same space lists (q-labels and their multiplicities)

```@repl tensor_operations
q.I + q.Z # Square matrix on the local space
```


```@repl tensor_operations
q.I + q.Z + 3
```


# Eigendecomposition

This can be called for a block-diagonal square matrix. 

The result is stored in the EigenResult struct with 4 fields (V, D, V_inv, and eig_list).

```@repl tensor_operations
q.I + q.Z
```


The local space has three sectors, but `q.I + q.Z` has two nonzero blocks. As
shown by `eig_list`, Telum treats the omitted sector as having eigenvalue zero.
Because every leg retains its full space list, no artificial identity shift is
needed to retain that information.

```@repl tensor_operations
a = eigen(q.I + q.Z)
println("Eigenvectors: ")
println(a.V)
println("Eigenvalues: ")
println(a.D)
println("Inverse of eigenvectors:")
println(a.V_inv) # This is 'nothing' if the matrix is Hermitian.
```


V: Eigenvectors. Contains the incoming leg of the given tensor.

D: Diagonal tensor

V_inv: Inverse of V. It is nothing if the input tensor is Hermitian.

eig_list: Store the list of tuples consist of 

1. Eigenvalue
2. Its degeneracy
3. q-label of the symmetry sector
4. What rank the eigenvalue is in the symmetry sector.

Similar to SVD.

In the example below, the last element for each tuple is always 1 since there is only one eigenvalue for each sector.

```@repl tensor_operations
a.eig_list
```


The function starts with the Hermiticity check. If the input is Hermitian, Hermitian version is executed and V_inv field is left 'nothing'. 

If the input is Hermitian and you want to skip the check, give keyword argument 'hermitian=true' to the function.

In the Hermitian case, the two legs of the given tensor should have the same itag, lock, plev, and dual fields because V_inv is not returned.

```@repl tensor_operations
a = eigen(q.I + q.Z; hermitian=true); # Hermiticity check is skipped
```

The function 'discard_eigen' truncates the result of eigendecomposition. It is frequently used in iterative diagonalization, a key step in NRG and DMRG. The first argument is the EigenResult object, and the second one is Nkeep, the number of kept eigenvalues. 

For now, 'Nkeep' lowest eigenvalues survive. More options (keep the highest ones, etc.) will be available in the near future.

The next two arguments are the itags of the kept and discarded eigenvalues tensor, similar to svd.


![settag_eig.png](assets/settag_eig.png)

```@repl tensor_operations
k, d = discard_eigen(a, 4, "kkk", "ddd")
println("kept spaces: ")
println(k.V)
println("kept eigenvalues: ")
println(k.D) # Only keep zero eigenvalues -> zero tensor

println("discarded spaces: ")
println(d.V)
println("discarded eigenvalues: ")
println(d.D)
```


There is another argument 'tol', which helps keep degeneracy near the threshold. It looks at 'Nkeep * tol' more eigenvalues, then cuts where the maximal difference occurs.

![tol_eig.png](assets/tol_eig.png)

# oplus

This is the function for the direct sum. Widely used when building MPO.

The remaining examples use the same single-channel system. Here, `nothing` means no channel symmetry.

oplus(t1, t2, int/vector or tuple of ints): Perform direct sum on t1 and t2 through selected dimension(s). In other words, concatenate them.

Input tensors should have the same itag, arrow direction, plev, and lock level. Same 'dual' is not necessary. Moreover, unselected legs should have the same space lists.

If there exists a unique permutation of the second tensor so that the condition above is satisfied, it is permuted. This is generalized to the vector/matrix of tensors input that will be described later.

In two-tensors and vector of tensors versions, summed legs can be selected from keyword arguments(Not for matrix input).

```@repl tensor_operations
qf = TLArray(q.F, ("site", "site", "op"));
qs = TLArray(q.S, ("site", "site", "op"));
println("qf: ")
println(qf)
println("qs: ")
println(qs)
oplres = oplus(qf, qs, 3)
println("oplres: ")
println(oplres)
```


```@repl tensor_operations
qsp = permutedims(qs, (1, 3, 2))
oplres = oplus(qf, qsp; itag="op")
println("oplres: ")
println(oplres)
```


```@repl tensor_operations
qf.spaces[3]
```


```@repl tensor_operations
qs.spaces[3]
```


```@repl tensor_operations
oplres.spaces[3] # Two above spaces are added
```


We can select multiple legs. The result becomes a block-diagonal matrix. Here is an example.

```@repl tensor_operations
println("q.I: ")
println(q.I)
println("q.Z: ")
println(q.Z)

oplres = oplus(q.I, q.Z, (1, 2))
println("oplres: ")
println(oplres)
```


```@repl tensor_operations
oplres.RMTs[1] # q.Z have matrix element 1 => RMT is an identity. The 3rd & 4th dimensions of RMT are for outer multiplicity
```


```@repl tensor_operations
oplres.RMTs[2] # q.Z has matrix element -1
```


You can also provide a vector of tensors. All tensors are concatenated through
the selected legs, provided they meet conditions analogous to the two-tensor
case.

```@repl tensor_operations
t = oplus([qf, qf, qf], 3)
```


```@repl tensor_operations
qf.spaces[3] # Only one copy of <charge=-1, spin=1> space
```


```@repl tensor_operations
t.spaces[3] # Three copies of <charge=-1, spin=1> space
```


We can even give the matrix of tensors, which is used when constructing the MPO. I'll introduce this version in the DMRG tutorial.

# addSingleton, deleteSingleton

This function adds/deletes a singleton dimension of the given tensor. A singleton leg is a leg with only one copy of vacuum space.

```@repl tensor_operations
println("qf: ")
println(qf)
println("After addSingleton")
println(addSingleton(qf, 3)) # Add a singleton leg. The newly created leg becomes the 3rd leg.
println("======================================")
as = addSingleton(qf, 3; itag="new", dir='-') # Can specify itag and dir for the new leg. 
println(as)

print("\nSpace of newly created leg: ")
as.spaces[3] # One copy of vacuum space
```


```@repl tensor_operations
println(addSingleton(qf, (2, 4))) # Can add multiple singleton legs at once.
println(addSingleton(qf, (2, 4); itag=("new1", "new2"), dir=('-', '+')))
```


If the leg indices are not given, leg(s) are generated at the end of the leg list. You can set the number of created legs by the keyword argument 'nlegs'.

```@repl tensor_operations
println(addSingleton(qf)) # A leg is added at the end(4th).
println(addSingleton(qf; nlegs=2, itag=("new1", "new2"), dir=('-', '+'))) # Add two legs at the end.
```


There is a function to delete a singleton dimension.

```@repl tensor_operations
println(as)
println("After deleteSingleton")
println(deleteSingleton(as)) # Delete every singleton leg. In this case, only the 3rd leg is deleted.
println(deleteSingleton(as; itag="new")) # Singleton leg with itag "new" is deleted. In this case, only the 3rd leg is deleted.
```


```@repl tensor_operations
deleteSingleton(as, 3) # Can also specify the leg index. Same result as above
```


```@repl tensor_operations
deleteSingleton(as, 2) # Error if the specified leg is not a singleton leg.
```


# Subspace selection

getsub(t::TLArray, i::Int, f::Function; preserve_space=false) => Truncate the space of the ith leg according to the given function f.

f sets a criterion whether the space is kept or truncated. It receives a q-label, and possible returned values are

1. nothing: Completely truncate a space corresponding to the given qlabel

2. integer, vector/tuple of integers: Part of the spaces are truncated. RMT is sliced accordingly. A negative number means counting from the last element. (-1: last, -2: second last, ..., similar to the Python convention)

3. colon: Completely keep the given space

Here is a basic example. A more detailed example will come later.

```@repl tensor_operations
println(q.I)
# In the getsub function, the criteria function is called for x = ((-1,), (0,)) / ((0,), (1,)) / ((1,), (0,))
function criteria(x)
    # If the charge quantum number is zero, do not truncate the space.
    if x[1] == (0,) return (:)
    else return nothing end
end
getsub(q.I, 2, criteria)
```


The criteria function can be expressed as a ternary operator (Boolean expression) ? (value if true) : (value if false)

x->(expression including x) is the Julia syntax to define an anonymous function.

```@repl tensor_operations
# Equivalent to the above.
# 'x->x[1]==(0,) ? (:) : nothing' is another way to write the criteria function.
s = getsub(q.I, 2, x->x[1]==(0,) ? (:) : nothing)
println(s)
s.spaces[2] # Only the space with charge quantum number 0 survives.
```


```@repl tensor_operations
s.spaces[1] # Unselected leg is not truncated.
```


You can truncate multiple legs with the same criteria. If you want to use a different criterion, call this function twice or more.

```@repl tensor_operations
s = getsub(q.I, (1, 2), x->x[1]==(0,) ? (:) : nothing) # Both 1st and 2nd legs are truncated according to the same criteria. 
println(s.spaces[1])
println(s.spaces[2])
```


The keyword argument 'preserve_space' determines whether the space list is also truncated. Its default value is false, so the space list is also reduced by default.

```@repl tensor_operations
s2 = getsub(q.I, 2, x->x[1]==(0,) ? (:) : nothing; preserve_space=true)
println(s2) # RMTs with nonzero charge quantum numbers are truncated
s2.spaces[2] # The spaces are preserved.
```


```@repl tensor_operations
getsub(q.I, x->x[1]==(0,) ? (:) : nothing; dir='-') # The truncated leg can be chosen from keyword arguments.
```


# Leg selection in functions

Many functions of Telum.jl have function signatures (tensor, leg indices, other_args...). For those functions, keyword arguments itag/dir/plev/lock/rev can be used instead of leg indices.

There are three exceptional cases.

1. getIdentity: The keyword arguments itag, plev, and lock determine the properties of the fused leg.

2. addSingleton: The keyword arguments determine the properties of the newly created leg(s).

3. oplus (Only when the matrix of the tensor is given): Not supported.

For those functions, run findlegs/matchings/contractables and related functions to get the leg indices first.

You can merge the local database with the global one.

# HDF5 tensor I/O

`save_tlarray` writes a concrete tensor to an HDF5 file, and `load_tlarray`
returns an independent concrete tensor. The example below uses a temporary path
and removes it after loading.

```@repl tensor_operations
let path = tempname() * ".h5"
    save_tlarray(path, q.I)
    loaded = load_tlarray(path)
    rm(path)
    loaded
end
```

Use the `name` keyword to store or load a tensor under a group other than the
default `"tensor"` group. Only concrete `TLArray` values can be serialized;
materialize lazy results first when independent HDF5 storage is required.
