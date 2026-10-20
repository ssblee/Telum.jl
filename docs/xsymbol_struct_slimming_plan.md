# Xsymbol Struct Slimming Plan

## Goal

Reduce `LurCGT.Xsymbol` so it stores only:

- `xsym_arr::Array{Float64, 3}`
- `size_byte::Int`

The qlabel spaces, contracted-leg positions, and symmetry identity should no
longer be duplicated inside the stored object. They are already available in
the lookup key and at each `getNsave_Xsymbol` call site.

## Current Data Flow

`Telum/src/contract.jl` does not use the identity fields stored in
`Xsymbol`. It computes the X-symbol inputs from the two `TLArray` sectors:

- `_xsymbol_sector_args` builds `upsp`, `dnsp`, and contracted legs in CGT
  stored-leg order.
- `_load_nonabelian_xarr` calls `getNsave_Xsymbol(...)`.
- `_xarr_for_symmetry` caches only `Array{Float64, 3}` in memory, using a key
  made from contracted qlabels and free qlabels.

`LurCGT/src/Xsymbol.jl` currently duplicates identity data in `Xsymbol`:

- `up1sp`, `dn1sp`, `up2sp`, `dn2sp`
- `legs1`, `legs2`

`LurCGT/src/sqlite_io.jl` also uses those fields when saving:

- `save_Xsymbol_sqlite(xsym)` derives the SQLite key from `xsym.up1sp`, etc.
- `load_Xsymbol_sqlite(...)` already derives the key from caller-provided
  inputs.

This means the load path already has the right shape, but the save path must be
changed before the fields can be removed.

## Reviewed Plan

The proposed change is sound if the key remains the single source of identity
for database and cache lookup.

Implementation should follow this order:

1. Change `Xsymbol` in `LurCGT/src/Xsymbol.jl` to contain only `xsym_arr` and
   `size_byte`.
2. Keep the existing type parameters if they are useful for dispatch or
   serialized compatibility, but do not store the matching runtime fields. If
   no dispatch depends on `U1`, `D1`, `U2`, `D2`, `NZ`, or `M`, remove those type
   parameters too.
3. Replace `save_Xsymbol_sqlite(xsym)` usage with an API that receives the key
   data explicitly, for example:

   ```julia
   save_Xsymbol_sqlite(S, up1sp, dn1sp, up2sp, dn2sp, ctlegs1, ctlegs2, xsym)
   ```

4. In `getNsave_Xsymbol_zeroadded`, construct `Xsymbol(Xsym_arr)` and save it
   with the already-standardized spaces and legs from the caller frame.
5. Leave `load_Xsymbol_sqlite(S, up1, dn1, up2, dn2, legs1, legs2)` as the
   lookup API. It already builds the key from inputs.
6. Keep `Telum/src/contract.jl` caches as `Dict{K, Array{Float64, 3}}`; no
   identity fields are needed there.

## Compatibility Notes

Existing SQLite entries may deserialize to the old `Xsymbol` layout. Decide
whether to support old cache files or invalidate them:

- Preferred simple path: bump or clear X-symbol cache entries and regenerate.
- Compatibility path: load old entries, extract `xsym_arr`, wrap it in the new
  `Xsymbol`, and save under the same key.

The simple path is acceptable if X-symbols are treated as generated cache data.

## Validation

After implementation:

- Run focused `getNsave_Xsymbol` tests for a cached miss followed by a cached
  hit.
- Run a non-Abelian `contract` case that exercises `_xarr_for_symmetry`.
- Confirm `save_Xsymbol_sqlite` no longer reads removed fields from `Xsymbol`.
- Inspect `@code_warntype` for the updated `getNsave_Xsymbol` path if practical.

## Risks

- The main risk is accidentally saving with non-standardized spaces or legs.
  Save keys must use the same normalized values passed to
  `getNsave_Xsymbol_zeroadded`.
- Serialized old cache data may fail to load after the struct layout change.
- If external code constructs or inspects `Xsymbol.up1sp` and related fields,
  it will break. Internal Telum contraction code does not rely on those fields.
