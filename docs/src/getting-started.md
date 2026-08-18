# Introduction and installation

## About Telum.jl

Telum.jl is a Julia tensor-network library with non-Abelian symmetries. It
currently supports fermionic systems with U(1) charge, SU(2) spin, and SU(N)
channel symmetries, as well as SU(2) spin systems. Support for Sp(2N), SO(N),
and SU(N≥3) spin systems is planned.

Telum is Latin for “weapon” and also stands for “TEnsor Library for Universal
Many-body simulation.” LurCGT.jl handles the Clebsch–Gordan algebra, so users
do not need to work with its internal representation directly.

Familiarity with ITensors.jl is helpful but not required. This tutorial is
self-contained. Readers migrating from QSpace can use the
[QSpace migration note](qspace-migration.md).

## Coming from ITensors.jl?

- `TLArray` is Telum’s tensor type, analogous to an `ITensor`.
- Every tensor leg has a `TLIndex`, which stores its tags (`itags`), direction,
  prime level (`plev`), lock level, and dual flag.
- `itags` serve a role similar to ITensor tags, and `plev` is analogous to an
  ITensor prime level.
- `A * B` contracts matching compatible legs. In addition to tags, Telum checks
  leg directions and symmetry space lists.
- Telum represents non-Abelian symmetry sectors explicitly through q-labels and
  per-leg space lists.

Caution

1. The interface of Telum.jl can be changed in the near future.

2. Its performance is not fully optimized yet.

## Installation

Install Telum with Julia’s package manager:

```julia
using Pkg
Pkg.add(url = "https://github.com/ssblee/Telum.jl")
```

Load Telum and the standard libraries used in the examples:

```@repl getting_started
using LurCGT
using Telum
using LinearAlgebra
```

# Defining tensor

To define a local tensor space:

1. Create an options object that describes the physical system and its symmetries.
2. Call `getLocalSpace` to obtain the local operators.

The code below creates local operators on a single-channel spinful fermionic site
with U(1) charge and SU(2) spin symmetry.

```@repl getting_started
# Arguments: number of channels, charge symmetry, spin symmetry, channel symmetry
option = FermionSOptions(1, :U1, :SU2, nothing);
q = getLocalSpace(option);
```

Four local operators are available: `I` (identity), `Z` (fermionic parity), `F`
(fermion annihilation), and `S` (spin). `F` and `S` are irreducible operators
with respect to the symmetry; Telum calls these IROPs.

```@repl getting_started
q.I
```


```@repl getting_started
q.S
```


`TLArray` (TeLum Array) is Telum’s basic tensor type.

This four-dimensional local space is split into three symmetry sectors. Its q-labels
give the charge relative to half filling and the SU(2) spin label `2S`.

```@repl getting_started
q.Z
```


For this one-channel system, `F` is the fermion annihilation IROP. Use `keys`
to list the local operators returned by `getLocalSpace`.

```@repl getting_started
keys(q)
```


```@repl getting_started
q.F
```


```@repl getting_started
q.S
```


You can attach tags to tensor legs when constructing a `TLArray`. The `itag`
field uses comma-separated words, similar to ITensor tags.

```@repl getting_started
ss = TLArray(q.S, ("site", "site", "op"))
```


Tags can also be assigned directly in the `getLocalSpace` call.

```@repl getting_started
qq = getLocalSpace(option, ("lur", "lur", "op"));
qq.S
```


Telum records a direction for every leg. In diagrams, an incoming leg is denoted
by `+` and an outgoing leg by `-`. This explicit direction helps distinguish
compatible bra and ket legs during contraction.

![arrow_convention.png](assets/arrow_convention.png)

An `itag` can contain multiple words. For example, the first leg below has the
two tags `site` and `asdf`, which can later be used to select it.

Tags are stored in a canonical order, so their written order does not matter.

```@repl getting_started
ss = TLArray(q.S, ("site,asdf", "site,zxcv", "op"))
```


# Contraction & leg manipulation

The multiplication operator `*` performs contraction. For `A * B`, Telum
contracts pairs of legs that have:

1. the same nonempty tags;
2. opposite directions;
3. lock level zero; and
4. matching prime level, dual flag, and space list.

This is similar to ITensor’s index-based contraction, with the additional
direction and symmetry-space checks made explicit.

For example, yellow and blue legs are contracted in the figure below.

![contract.png](assets/contract.png)

## Leg manipulation

Telum provides functions for changing leg metadata and thereby controlling which
legs contract. Each leg has one `TLIndex` object.

The five `TLIndex` fields are `itags`, direction, prime level, lock level, and
dual flag. The last three are introduced below.

```@repl getting_started
function print_index(i::TLIndex)
    println("(itag: $(i.itags), dir: $(i.dir), plev: $(i.plev), lock: $(i.lock), dual: $(i.dual))")
end

println(ss)
print_index(ss.inds[1])
print_index(ss.inds[2])
print_index(ss.inds[3])
```


### Lock

You can increase the lock level to avoid contraction. The leg is never contracted if its lock level is not 0.

lock(q::TLArray, i::Int): Increase lock level of ith leg by 1. The first leg is selected in the following example.

The lock level is printed next to the itag with a form "🔒\<lock level\>" if the lock level is nonzero.

```@repl getting_started
ss1 = lock(ss, 1)
println(ss1)
```


The tensors ss and ss1 refer to the same numerical data (w-matrices, RMT). The metadata of the argument ss is never changed.

```@repl getting_started
ss
```


It can also accept a vector or a tuple of leg indices to lock multiple legs.

```@repl getting_started
ss12 = lock(ss, (1, 2)) # Lock 1st & 2nd legs
printmeta(ss12)
```


Instead of leg indices, we can select legs by their fields. In this example, legs with itag 'site' (1st and 2nd legs) are locked.

```@repl getting_started
sst = lock(ss; itag="site")
printmeta(sst)
```


The argument behind the semicolon is referred to as a keyword argument. Detailed options can be specified through keyword arguments. A list of available keyword arguments will be given later.

By default, the lock level increases by 1. We can change the increment by giving the 'inc' keyword argument.

If you give a tensor only, every leg is locked.

```@repl getting_started
sst = lock(ss; itag="op", inc=3);
printmeta(sst)
sss = lock(ss)
printmeta(sss)
```


### Lockp & Unlock

You can use the lockp function for permanent locking. The lock levels of selected legs are fixed to -1, and an explicit unlock call is the only way to 'unlock' the leg.

The unlock function sets the lock level of selected legs to 0.

```@repl getting_started
ssp = lockp(ss, (1, 2));
printmeta(ssp)
ssp2 = lock(ssp, 2); # Nothing happens. Lock level is fixed to -1.
printmeta(ssp2)
sspp = unlock(ssp2, (2, 3))
printmeta(sspp)
```


### Lock level in contraction

Suppose we call A * B for two tensors A and B, and the leg l of A is locked(i.e., positive lock level). In the contraction function,

1. If the leg l has a matching leg in B (same itag/plev/dual, opposite direction), its lock level is reduced by 1.
2. Otherwise, its lock level is unchanged.

The same rules apply to the legs of `B`. A permanently locked leg (lock level
`-1`, created by `lockp`) is never changed during contraction.

Below is an example. Assume that the legs have the same plev and dual. The lock level of the 'bbb' leg remains the same since there is no matching leg in the tensor B.

![contraction_locklev.png](assets/contraction_locklev.png)

### Prime level

The prime level is a non-negative integer associated with each leg. Legs with
different prime levels do not contract. This is analogous to priming an ITensor
index: use it when otherwise-identical legs must be distinguished. Telum’s leg
direction already distinguishes many bra/ket pairs, so priming is often needed
less frequently.

As with locking, you can select legs by index or by keyword arguments.

The prime level is printed next to the itag with a form of "p\<prime level\>" if the prime level is nonzero.

```@repl getting_started
printmeta(ss)
printmeta(prime(ss, (1, 3)))
printmeta(prime(ss; itag="site"))
```


You can adjust the increment by the keyword argument 'inc'. Although the increment can be negative, the prime level never goes below 0.

```@repl getting_started
ssp = prime(ss, (1, 3); inc=3)
printmeta(ssp)
ssp2 = prime(ssp, (1, 2); inc=-2)
printmeta(ssp2)
```


There are two other prime-related functions, 'setprime' and 'noprime'. The setprime function sets the prime level to the given value, and noprime sets it to 0. Similarly, you can give keyword arguments instead of an explicit list of leg indices.

```@repl getting_started
ssp3 = setprime(ssp2, (2, 3), 2) # Set the prime level of the 2nd and 3rd legs to 2
printmeta(ssp3)
ssp4 = noprime(ssp3, 3) # Set the prime level of the 3rd leg to 0
printmeta(ssp4)
ssp5 = noprime(ssp4) # If no leg is specified, set all legs to 0
printmeta(ssp5)
```


If no leg is specified, apply the function to every leg.

```@repl getting_started
printmeta(prime(ss; inc=3))
printmeta(setprime(ss, 4))
```


### Change itag

There are many functions to add and remove itags of a given tensor. The first function is additag, which adds itag to the selected legs.

```@repl getting_started
printmeta(ss)
printmeta(additag(ss, (1, 2), "nitag"))
# Itag is not duplicated (only "bbb" is added for the 1st leg)
printmeta(additag(ss, "asdf,bbb"; itag="site")) 
printmeta(additag(ss, "ccc")) # Add itag to all legs
```


The next function is removeitag. Not only a single string, but this function can also accept a vector or a tuple of strings. If ("aaa,bbb", "ccc") is given, first "aaa" and "bbb" are removed for legs whose itag contains both "aaa" and "bbb". Then "ccc" is removed similarly for legs with itag "ccc".

You can give leg indices or keyword arguments to restrict the change to a part of the legs.

```@repl getting_started
printmeta(ss)
printmeta(removeitag(ss, (1, 2), "site"))
printmeta(removeitag(ss, (1, 2), ("asdf", "zxcv")))
printmeta(removeitag(ss, ("zzzz,asdf", "zxcv"); itag="site"))
```


The 'replaceitag' function replaces a given word with another word.

```@repl getting_started
printmeta(ss)
printmeta(replaceitag(ss, "site"=>"newtag"))
```


(...) => (...) is the constructor of the 'Pair' object in Julia. Their elements are accessible from 'first' and 'second'.

```@repl getting_started
a = 1 => 2
println(typeof(a))
println(a.first)
println(a.second)
```


If the string to be removed contains multiple words, only the legs that contains all are affected.

```@repl getting_started
printmeta(ss)
printmeta(replaceitag(ss, "site,asdf"=>"newtag")) # The second leg is not replaced
```


You can give multiple replacement rules. The code below swaps the itag "asdf" and "zxcv".

```@repl getting_started
printmeta(ss)
printmeta(replaceitag(ss, "asdf"=>"zxcv", "zxcv"=>"asdf"))
```


The detailed explanation of how the replacements act on the first leg "site,asdf" is:

1. Define new itag as an empty string. old="site,asdf", new=""

2. First replacement rule. Find "asdf" in the old itag. If so, remove it from the old itag and add "zxcv" to the new itag. old="site", new="zxcv"

3. Second replacement. Find "zxcv" and do a similar thing. Here, nothing happens.

4. Move the remaining 'old' itag to the new itag and return the new one. old="", new="site,zxcv"

The itag is processed in the order in which it comes in as an argument.

We can give a dictionary with String keys and values. In this case, the order of rule application is not defined. Here is an example of changing every number n <= 100 to n+1.

```@repl getting_started
dict = Dict("$i" => "$(i+1)" for i in 1:100);
ssn = TLArray(q.S, ("24,46", "site,25", "op,63"))
printmeta(replaceitag(ssn, dict))
```


For both versions, we can restrict the affected leg by keyword arguments.

```@repl getting_started
ssn = TLArray(q.S, ("24,46", "site,25", "op,63"))
printmeta(replaceitag(ssn, dict; dir='-')) # Only replace 2nd and 3rd legs
```


The last one is setitag. This sets the itag of selected legs to the given string.

```@repl getting_started
printmeta(ss)
printmeta(setitag(ss, (1, 2), "lur"))
printmeta(setitag(ss, "lur"; itag="site"))
```


### Tensor product

Sometimes you intend to perform a contraction, but the tensor product is performed due to your mistake when manipulating the legs. Then, the program can die silently.

```@repl getting_started
# Example: You intended to contract F and Z
qs = TLArray(q.S, ("bbbb", "bbbb", "op"))
qz = TLArray(q.Z, ("bbbbb", "cccc")); # But you made a typo in the first leg

# qs * qz in this situation will generate a rank-5 tensor. More severe things can happen for large computations.
```


For users to notice it quickly, the contraction function returns an error when the contracted pair is not found.

```@repl getting_started
qs * qz
```


If you intended a tensor product, use ⊗ operator instead. To enter the operator symbol, enter '\otimes' and the tab key in the REPL, code editor with Julia extension.

```@repl getting_started
printmeta(ss)
ii = TLArray(q.I, ("lur", "lur"))  
printmeta(ii)
ss ⊗ ii # It can take long time
```


Not to introduce ambiguity when contracting, an error occurs if the legs with the same properties (except lock level) appear as a result. The code below attempts to change the itag of the second leg to "op", resulting in an error.

```@repl getting_started
printmeta(ss)
setitag(ss, "op"; itag="zxcv")
```


Multiple legs with the same empty itags are allowed since they are never selected in contraction.

```@repl getting_started
q.S
```
