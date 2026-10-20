# Final-only lazy contraction-chain evaluation plan

## Goal

Reduce allocation and peak live memory when a `TLArrayContraction` occurs only
as an intermediate in a lazy contraction chain whose final result is one
`TLArray` or scalar.

For example, in:

```julia
@lazy A * B * C * D
```

the results of `A * B` and `(A * B) * C` are consumed once and never observed
as tensor values.  Their dense RMT payloads therefore do not need to be
installed in the `RMTs` field of their `TLArrayContraction` nodes.

The optimization must preserve the existing lazy metadata model:

- every lazy node keeps its original sector slots and slot numbering;
- q-label, w-matrix, factor, index, and view-state metadata remain unchanged;
- a sector that is undefined in a final-only intermediate is not thereby known
  zero;
- the final materialized `TLArray` has the ordinary eager storage contract.

This is an execution optimization, not a sector-compaction or metadata
reconstruction change.

## Why `compute_sectors` is not the execution model

`compute_sectors` is a cache-population API.  It recursively finds source
sectors and writes each requested result payload to `q.RMTs[sector]`.  That is
the right behavior for an independently reusable lazy tensor, but it is not
the right execution model for a final-only contraction chain.

The final-only evaluator must not call `compute_sectors` on an intermediate
contraction.  It should use the sector dependency information already present
in the lazy nodes, allocate only evaluator-local buffers, and write only the
root tensor's final RMTs to persistent storage.

The current materialization route is:

```julia
compute_sectors(q, sector_slots(q))
```

For a contraction chain, this causes every required intermediate sector to be
allocated in persistent node storage before its parent reads it.  The planner
below replaces this route only for eligible final-only trees.

There are also smaller transient allocations for each recursive request:

- request/pending `BitVector`s in `_pending_contraction_work`;
- filtered work-item and used-source-sector vectors;
- result-slot-sized temporary vectors in
  `_compute_contraction_sectors_from_sources!`;
- prepared-RMT cache index vectors.

The dense intermediate RMT payloads are expected to dominate for useful
chains.  Measure that assumption before optimizing the smaller metadata
allocations.

## Build a sector dependency DAG

Planning begins with metadata only.  Assign each encountered lazy contraction
recipe a small **plan-local node ID** during a traversal from the final result.
Here, “node identity” means that node ID: one particular
`TLArrayContraction` computation returned by `_lazy_contract`, not its Julia
type, its q-labels, or a raw `objectid` value.  The ID exists only while
planning and executing the final result.

For example:

```julia
p = _lazy_contract(A, B)  # plan node 1
r = _lazy_contract(p, C)  # plan node 2, the final root
```

Sector slot `1` is only meaningful relative to its owner.  Thus `(node 1,
slot 1)` is a sector of `A * B`, while `(node 2, slot 1)` is a different
sector of `(A * B) * C`.  They must be different planner vertices even if
their q-labels happen to be equal.

Treat every pair:

```text
(plan-local contraction node ID, sector slot)
```

as one sector-value vertex.  A concrete `TLArray` sector is a leaf vertex.  A
lazy contraction output sector is an accumulator vertex.  Its `work_items`
provide contribution records.  For node `q`, each item
`(sector1, sector2, result_sector, factor_position)` means:

```text
input vertex (node(q.arr1), sector1) + input vertex (node(q.arr2), sector2)
    -- q.factors[result_sector][factor_position] --> output vertex (node(q), result_sector)
```

During the first implementation, the traversal can assign one node ID for
each source occurrence in the final contraction tree.  If the same lazy
computation is reachable through multiple source occurrences, use one shared
node ID only when it denotes the same canonical stored-RMT computation; this
is what exposes cross-parent sector reuse to the scheduler.  State-only
rewraps need separate handling of their `conj`, `scale`, and `perm` metadata,
so they should use the normal fallback until that sharing rule is explicit.

Expanding the requested final output slots therefore produces a directed
acyclic graph (DAG), not generally a tree.  Even though each tensor in the
syntactic contraction chain has one parent, one intermediate *sector* can be
used by several work items or by several final output sectors.

For every vertex, record in concrete, parallel arrays:

- its node and original sector slot;
- RMT dimensions and payload bytes;
- whether it is a concrete leaf or an accumulator;
- its incoming contribution range; and
- the number and locations of its downstream uses in the requested final
  result.

For every contribution, record the two input vertex IDs, target vertex ID,
factor position, and the existing contraction-kernel temporary size.  The
graph is only evaluator-local.  It must not alter any `TLArrayContraction`
slot list or state bit.

Sector IDs alone are not sufficient to choose an optimal schedule.  The
planner also needs the byte size of each RMT and the temporary size/cost of its
pair contraction.  Otherwise it cannot distinguish a sector that is cheap to
recompute from one that is expensive to retain.

## The scheduling problem

The planner chooses, for every non-root sector vertex, whether to:

- retain its buffer until all planned downstream uses complete; or
- recompute it for a later use and release the earlier buffer immediately.

It also chooses the order of contributions within each output sector.  This is
a weighted pebbling/scheduling problem on the sector DAG.  An exact
minimum-peak-memory schedule with optional recomputation is combinatorial; it
should not be the first implementation target.  The useful restricted problem
is a postorder schedule that retains a value only while it has imminent uses
in the current parent sector.

The key local rule is to group a target sector's work items by one input
sector.  For each group:

1. evaluate that input sector into one scratch buffer (the **held** input);
2. evaluate each matching opposite input sector one at a time;
3. accumulate the pair directly into the target RMT; and
4. release the held input after its final group use.

Choose the grouping side using byte and cost estimates.  Holding the smaller
input is normally best, but a large input that can be prepared in a streaming
buffer may be the better held side.  The existing prepared-RMT streaming
choice is one input to this decision, not the whole plan.

This reduces a wide set of same-level input sectors to at most two live input
sector buffers plus the target output buffer.  Apply the same rule recursively
while evaluating either input.

## Practical memory-cost heuristic

Use a deterministic, bottom-up estimate before emitting the executable
schedule.  For vertex `v`, let `B(v)` be its RMT bytes, and let `W(v)` be the
largest workspace needed to compute it.  Estimate the cost of an orientation
of a contribution group as the largest simultaneously live set:

```text
B(target) + B(held input) + B(streamed input)
+ local pair-contraction workspace
+ nested evaluation workspace not overlapped by a released sibling
```

For each target sector, evaluate candidate groupings by input 1 and input 2.
When groups have no cross-group cache, their order does not change the peak:
each held buffer is released before the next group.  If optional cross-group
caching is enabled later, choose an order with a deterministic live-byte/cost
heuristic and retain a value only when its saved recomputation cost justifies
the added live bytes.

Initially, use a simple conservative policy:

- reuse a sector within one contiguous work-item group;
- release it before moving to another group;
- recompute it for a later parent/root sector;
- process final root sectors one at a time.

This policy gives a predictable small live set and eliminates persistent
intermediate RMT allocation.  Later, a memory budget can permit cross-group
or cross-root caching when measurements show recomputation dominates.

## Integration point

No change to `_lazy_finalize` is required.

`_lazy_finalize(::AbstractTLArray)` already finalizes through `TLArray(q)`, and
`TLArray(::TLArrayContraction)` calls `materialize(q)` before aliasing the root
node's materialized storage into the public `TLArray`.  Therefore the natural
entry point is a specialized implementation of:

```julia
materialize(q::TLArrayContraction)
```

It should first construct the requested-sector DAG, then select final-only
evaluation for an eligible contraction tree and fall back to the existing
`compute_sectors(q, sector_slots(q))` path otherwise.
This also makes direct `TLArray(lazy_q)` use the same optimization, which is
appropriate because it is the same materialization boundary.

The generic `compute_sectors` API must keep its current memoizing behavior.
Callers may request arbitrary slots repeatedly or retain the lazy value, so it
cannot assume that its results have a single consumer.

## Final-only execution

Evaluate requested root sectors while treating intermediate RMTs as temporary
buffers.  The root still receives normal RMT storage because `materialize`
must return a concrete tensor.

For each requested root sector, execute the already planned contribution
groups.  A recursive sector evaluator receives a vertex ID, a supplied output
buffer, and the task-local workspace; it never assigns an intermediate
`q.RMTs[sector]` slot.

Conceptually:

```text
root output sector
  ├─ child sector 3  -> temporary RMT -> accumulate into root RMT
  ├─ child sector 8  -> temporary RMT -> accumulate into root RMT
  └─ child sector 11 -> temporary RMT -> accumulate into root RMT
```

The procedure recurses through a child `TLArrayContraction` until it reaches a
concrete input.  Slots are only identifiers in evaluator-local plan arrays;
they are never filtered, compacted, or renumbered in the lazy node itself.

For a fully contracted scalar, accumulate directly into the final scalar RMT
buffer.  No intermediate dense RMT array should survive beyond its immediate
parent accumulation.

## Kernel extraction and workspace

Extract the numerical part of
`_compute_contraction_sectors_from_sources!` into a helper that computes one
planned output sector into caller-provided storage.  The helper should reuse
the existing prepared-RMT and `_contract_RMT_pair_into!` kernels; this plan
does not change the non-Abelian factor equation or stored-leg permutations.

Pass concrete workspace buffers down the evaluator instead of adding a global
pool or a new lazy-wrapper type.  A workspace needs only concrete containers:

- RMT scratch storage sized to the largest live temporary sector;
- the existing contraction temporary and factor-scaling scratch buffers;
- compact prepared-RMT caches for concrete input sectors; and
- local slot-to-cache/request maps.

Use evaluator-local compact maps only.  In particular, do not replace a node's
`RMTs`, `isdefined`, or `iszero` vectors with compact storage.

When parallelizing final-sector work, give each task an independent workspace
and final-sector output buffer.  Do not share mutable prepared-RMT caches or a
global scratch buffer across tasks.

## Eligibility and fallback

Use final-only evaluation only when the reachable lazy graph is a pure,
single-consumer contraction tree.  It must fall back to regular materialization
when an intermediate can be observed or reused, or when a supported structural
node cannot forward the required sector request without materializing its
source.

The initial implementation can be intentionally narrow:

- accept `TLArrayContraction` nodes and concrete `TLArray` leaves;
- accept only child contractions with one parent use in the tree;
- retain the existing path for `SubTLArray`, singleton wrappers, or repeated
  lazy references until their request forwarding is explicitly implemented.

After correctness and measurements are established, extend the evaluator to
structural wrappers that already preserve sector-slot numbering.

## Low-risk preparatory changes

These can be implemented independently of temporary-RMT streaming:

1. Build the compact sector DAG and parent-to-child contribution adjacency once
   per final evaluation.  This avoids repeated scans of all child work items.
2. In final-only code, allocate contribution flags and temporary result handles
   only for planned root sectors, using evaluator-local vertex IDs.
3. Preserve the current prepared-RMT kernels and streaming policy as local
   choices inside the new schedule; replace only intermediate output-storage
   lifetime first.

Avoid a broad contraction-order optimizer in this change.  Reassociating
`A * B * C * D` could reduce scratch size further, but it changes planning,
index/lock handling, and floating-point accumulation order.  The final-only
evaluator should first preserve the existing left-associated contraction tree.

## Validation

Before changing the implementation, record allocations, elapsed time, and a
proxy for peak temporary RMT bytes for representative chains.  Include chains
with non-Abelian factors, a scalar final result, and multiple output sectors.

Add focused regression tests that verify:

- a final-only chain equals eager left-associated contraction;
- only root RMT slots become defined after final-only materialization;
- intermediate lazy contraction slots remain undefined and nonzero-unknown;
- source and root sector-slot numbering is unchanged; and
- a non-eligible/reused lazy value follows the regular memoizing path.

Run the relevant contraction tests, inspect the modified helpers with
`@code_warntype`, and run the required DMRG test.  Report allocation and peak
temporary-memory changes separately from runtime, since reducing retained RMT
storage may trade a small amount of recomputation for a substantially smaller
live set.
