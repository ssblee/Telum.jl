# `AbstractTLArray` Accessors and Raw Field Access

This note inventories the accessors implemented for `AbstractTLArray` values and records the property-access rule: `q.field` exposes the raw stored field value. Logical or computed views belong to named accessors such as `inds(q)`, `spaces(q)`, and `logical_qlabels(q)`.

## Abstract type and concrete subtypes

`AbstractTLArray` is the common supertype for these five concrete lazy/eager implementations in `src/TLArray.jl`:

- `TLArray`: owns sector metadata, w-matrix storage, and RMT payload slots.
- `TLArrayContraction`: lazy contraction result; retains two source arrays and work metadata.
- `SubTLArray`: lazy sub-selection; maps output sectors and selectors back to its source.
- `AddSingletonTLArray`: lazy singleton-leg insertion; retains its source and leg maps.
- `DeleteSingletonTLArray`: lazy singleton-leg removal; retains its source and leg maps.

`SingletonTLArray` is a union alias for the last two, not another struct.

## Property access versus accessor functions

`Base.getproperty(::AbstractTLArray, name)` now delegates directly to `getfield`, so stored-field properties return raw values. `propertynames(q)` is derived from the concrete struct's fields. `AddSingletonTLArray` and `DeleteSingletonTLArray` no longer synthesize `.wmatdata` or `.wmatinfo`; use `sector_wmat` for effective w-matrices. Internal code that needs packed source storage uses `stored_wmat_storage`.

For the requested raw-property behavior, `q.name` should mean `getfield(q, :name)` for a field physically present on that concrete struct. Use explicit accessor functions when a logical-order or computed value is wanted. This also means a raw field only exists when the concrete struct actually stores it: singleton wrappers store neither `wmatdata` nor `wmatinfo`; callers needing their effective matrices should use `sector_wmat`.

| Property | Raw stored value | Logical view accessor |
| --- | --- | --- |
| `qlabels` | Stored sector q-label tuples | `logical_qlabels(q)` reorders each tuple when `perm` is nonidentity |
| `inds` | Stored-leg `TLIndex` tuple | `inds(q)` applies `perm` and deferred `conj` direction change |
| `spaces` | Stored-leg space-list tuple | `spaces(q)` applies `perm` |
| `wmatdata`, `wmatinfo` | Packed w-matrix storage and its descriptors, where physically stored | No logical field accessor; use `sector_wmat` for an effective matrix |
| all other real fields | Field value on that struct | No transformation; property access returns the field value |

Property names describe concrete fields, not computed accessor aliases. For singleton wrappers, `hasproperty(q, :wmatdata)` and `hasproperty(q, :wmatinfo)` are false because those fields are not stored on the wrapper.

## Accessor inventory

### Type and symmetry metadata

| Function | Result / meaning |
| --- | --- |
| `productsymm(q)` | Product-symmetry type parameter (`PS`) |
| `product_symms(q)` | Ordered tuple of component symmetries |
| `symm(q)` | Ordered component symmetry tuple; positions agree with q-label tuple positions |
| `nsymms(q)` | Number of symmetry components |
| `qlabeltype(q)` | Q-label element type (`QT`) |
| `Base.eltype(q)` / `eltype(typeof(q))` | Scalar element type (`T`) |
| `Base.ndims(q)` / `ndims(typeof(q))` | Logical tensor rank (`QD`) |

### Stored and logical leg metadata

| Function | Result / meaning |
| --- | --- |
| `stored_inds(q)` | Raw stored-leg index tuple |
| `stored_spaces(q)` | Raw stored-leg space lists |
| `stored_qlabels(q)` | Raw q-label tuples in stored-leg order |
| `stored_conj(q)` | Deferred conjugation flag |
| `stored_scale(q)` | Deferred scalar factor |
| `stored_perm(q)` | Logical-to-stored leg permutation |
| `stored_wmat_storage(q)` | Internal packed w-matrix data and descriptors; singleton wrappers delegate to their source |
| `logical_qlabels(q)` | Sector q-label tuples in logical-leg order, retaining sector-slot order |
| `inds(q)` | Indices in logical-leg order, including deferred conjugation direction change |
| `spaces(q)` | Space lists in logical-leg order |
| `stored_sector_qlabel(q, sector, leg)` | Q-label at a stored-leg position |
| `stored_sector_qlabel(QT, q, sector, leg)` | Same, asserted/cast to `QT` |
| `sector_qlabel(q, sector, leg)` | Q-label at a logical-leg position |
| `sector_qlabel(QT, q, sector, leg)` | Same, asserted/cast to `QT` |

### Array shape and slot iteration

| Function | Result / meaning |
| --- | --- |
| `Base.size(q)` | Logical dimensions, summed from `spaces(q)` |
| `Base.size(q, d)` | Dimension `d`, or `1` beyond rank |
| `Base.length(q)` | Stable sector-slot count (not number of nonzero sectors) |
| `Base.firstindex(q)` / `Base.lastindex(q)` | First slot (`1`) / final sector slot |
| `sector_count(q)` | Stable sector-slot count, including zero and unevaluated slots |
| `sector_slots(q)` | Iterable of stable sector-slot indices |
| `nsectors(q)` | Number of slots not marked zero |

### Sector state, w-matrices, and RMT blocks

| Function | Result / meaning |
| --- | --- |
| `is_sector_defined(q, sector)` | Whether an RMT payload is defined/materialized |
| `is_sector_zero(q, sector)` | Whether the sector is known zero |
| `is_sector_active(q, sector)` | Whether the sector is not known zero |
| `sector_wmat(q, sector, symmetry)` | CGT w-matrix for a product-symmetry index or `Val` index |
| `sector_wmat_slot(q, sector, slot)` | Stored non-Abelian w-matrix by compact slot |
| `stored_sector_rmt_dim(q, sector)` | RMT dimensions in stored order (or cached dimensions for lazy wrappers) |
| `sector_rmt_dim(q, sector)` | RMT dimensions in logical-leg order |
| `sector_rmt_axis_dim(q, sector, leg)` | One logical RMT leg dimension |
| `sector_rmt(q, sector)` | `(rmt, scale)` pair; scale must be applied for logical numeric values |
| `source_sector(q, sector)` | `SubTLArray` result slot's source slot |

`_defined_sector_rmt(q, sector)` is an internal helper that returns a materialized payload and errors if the sector is unevaluated. The `stored_*` functions intentionally expose storage representation; their leading underscore-free names do not make them logical views.

## Implementation and audit

### Intended policy

For every physically stored field, `q.field` should return the same raw stored value as `getfield(q, :field)`. This is especially important for `t.inds`, `t.spaces`, and `t.qlabels`: they represent stored-leg order and stored sector q-label tuples, regardless of deferred `perm` or `conj` state. Field access should not silently apply view transformations or proxy to another tensor.

Logical values remain available through explicitly named accessors:

- `inds(q)` for logical indices, including permutation and conjugation direction.
- `spaces(q)` for logical-leg space lists.
- `logical_qlabels(q)` for q-label tuples in logical-leg order.
- `sector_qlabel(q, sector, leg)` and `sector_rmt_dim(q, sector)` for logical-leg sector queries.
- `sector_rmt(q, sector)` and `sector_wmat(q, sector, ...)` for effective sector data.

The leg and q-label accessors are currently module-qualified in client code (for example, `Telum.inds(q)`, `Telum.spaces(q)`, and `Telum.logical_qlabels(q)`).

The implementation follows this sequence:

1. `Base.getproperty(::AbstractTLArray, name)` returns the actual concrete field via `getfield`; `.qlabels`, `.inds`, and `.spaces` no longer invoke logical accessors.
2. Singleton-wrapper property forwarding for `.wmatdata` and `.wmatinfo` is removed. Internal operations that require the underlying packed buffers use `stored_wmat_storage`; effective matrices remain available through `sector_wmat`.
3. `propertynames` is derived from `fieldnames(typeof(q))`, so it agrees with the fields physically present on each subtype, including singleton wrappers.
4. Internal package call sites that operate in logical leg order use `inds(q)` and `spaces(q)`. Direct property reads remaining in source are for unrelated structs' own fields (for example, CGT metadata and `getLocalSpace` constructor data), not `AbstractTLArray` logical views.
5. The permutation docstring now points to `logical_qlabels(q)` rather than the nonexistent `qlabels(q)` accessor.
6. Regression tests cover nonidentity permutations, raw field/property-name agreement, and singleton wrappers. The index metadata, TLArray structure, contraction, and lazy getsub tests exercise the explicit logical accessors and storage-alias helpers.

### Risks and migration impact

- **Silent semantic change:** Existing code using `q.inds`, `q.spaces`, or `q.qlabels` may currently receive logical values. After the change it receives stored values, which can produce wrong leg matching, shape interpretation, or q-label associations without raising an error.
- **Permuted and conjugated views:** The difference is observable only for nonidentity `perm` and/or `conj`; identity-view tests alone will miss regressions. `inds(q)` must remain the path that applies both transformations; `spaces(q)` and `logical_qlabels(q)` apply permutation but not conjugation.
- **Wrapper compatibility:** Code may rely on singleton wrappers exposing `.wmatdata` or `.wmatinfo` through their source. Raw-field policy means those property reads should fail because those are not wrapper fields. Callers should use `sector_wmat` or explicitly inspect the wrapped source when appropriate.
- **Reflection and introspection:** `propertynames` and `hasproperty` will change for singleton wrappers. Tools that discover properties dynamically may need updating.
- **Internal call-site regressions:** A broad replacement of computed properties with raw fields can alter algorithm behavior. Audit usages by intent and update only the callers that need logical order.
- **Mutability exposure:** Raw fields such as vectors and bit vectors are mutable storage. Returning them makes that fact explicit but does not protect invariants; document that direct mutation is unsupported unless the API intentionally permits it.
- **External downstream users:** This is a behavior change for package users. Consider a release note and migration guidance mapping old property usage to `inds(q)`, `spaces(q)`, or `logical_qlabels(q)` before shipping it.

### Acceptance criteria

- A stored-field property returns the raw concrete field value and never applies view state.
- A nonexistent field on a wrapper is not synthesized by `getproperty` and is not reported by `propertynames` / `hasproperty`.
- Logical accessors preserve their documented transformations for all `AbstractTLArray` subtypes.
- Package call sites that require logical values use the explicit accessors.

The Julia implementation has been changed. Downstream callers that treated `.inds`, `.spaces`, or `.qlabels` as logical accessors must migrate to the functions listed above. The risks below remain relevant for that migration. Tests that asserted logical-view behavior through fields were updated to use accessors, and regression coverage now verifies that properties expose stored values. The focused index-metadata test file passed (305 assertions); the full test suite has not been run.
