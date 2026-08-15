```@example advanced_topics
using LurCGT
using Telum
using LinearAlgebra
```

Output:

```text
SYSTEM: caught exception of type :MethodError while trying to print a failed Task notice; giving up
```

```@example advanced_topics
option = FermionSOptions(1, :U1, :SU2, nothing)
q = getLocalSpace(option);
```

# Advanced topics

Simple transformations such as scalar multiplication, permutation, and conjugation retain shared tensor storage. In the example below, `T2` uses the storage of `T1` together with its own transformation metadata.

```@example advanced_topics
using Telum

set_accumul_costs!(false)
```

```@example advanced_topics
T1 = q.I
println(T1)
println(typeof(T1)) # Concrete TLArray
T2 = 2 * T1
println(T2)
println(typeof(T2)) # Shares storage with T1 through lazy transformation state
```

If the first RMT of T1 is mutated, T2 is also affected.

```@example advanced_topics
T1.RMTs[1] = [4;;;;] 
println(T1)
println(T2) # The first RMT of T2 is also changed.
```

Output:

```text
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	4.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	1.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	1.000000	√1
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	8.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	2.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	2.000000	√1
```

You can call `copy` on a concrete `TLArray` to get independently owned storage
while retaining its deferred logical state.

```@example advanced_topics
q = getLocalSpace(option)
T1 = q.I
T2 = 2 * T1
println(T1)
T2_ = copy(T2)
println(T2_)
```

Output:

```text
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	1.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	1.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	1.000000	√1
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	2.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	2.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	2.000000	√1
```

```@example advanced_topics
T1.RMTs[1] = [4;;;;] 
println(T1)
println(T2_) # Not changed
```

Output:

```text
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	4.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	1.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	1.000000	√1
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	2.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	2.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	2.000000	√1
```

Avoid mutation of tensor as far as possible. If it is inevitable, copy the original tensor first.

```@example advanced_topics
q = getLocalSpace(option)
T1 = q.I
T2 = 2 * T1
T1_ = copy(T1)
println(T1_)
println(T2)
```

Output:

```text
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	1.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	1.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	1.000000	√1
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	2.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	2.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	2.000000	√1
```

```@example advanced_topics
T1_.RMTs[1] = [4;;;;]
println(T1_)
println(T2) # Not changed
```

Output:

```text
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	4.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	1.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	1.000000	√1
2D TLArray, 2 symmetries [U1, SU2]  ["+", "-"]
  1.	1x1	| 1x1	[ -1 0 ; -1 0 ]	2.000000	√1
  2.	1x1	| 2x2	[  0 1 ;  0 1 ]	2.000000	√2
  3.	1x1	| 1x1	[  1 0 ;  1 0 ]	2.000000	√1
```

# Database management

LurCGT has a 3-level system to store CGT data.

1. Global database: Global storage so that every process can read. Only one process can write at a time.
2. Local database:  Local storage for a single process.
3. In-memory cache: Cache managed by a process for fast reuse.

LurCGT finds the CGT data (CG3, X-symbol, ...) from 3->2->1. If it does not exist, data is created and stored in 2 and 3.

By default, databases are placed in `\$HOME/.LurCGT_sqlite/`.


### In the local environment

The CGT data computed in LurCGT is stored on disk via SQlite. In the local environment, no additional settings are required. 

Databases are automatically created if they do not exist. Merging the local one with global and deleting local one occurs automatically when the process is terminated normally. Otherwise, local ones should be removed manually.

```@example advanced_topics
script = expanduser("~/.LurCGT_sqlite/")
run(`ls $script`)
```

Output:

```text
global
local
Process(`ls /home/lurlurlur/.LurCGT_sqlite/`, ProcessExited(0))
```

```@example advanced_topics
script = expanduser("~/.LurCGT_sqlite/local/")
run(`ls $script`)
```

Output:

```text
Sp4
Sp6
SU2
SU3
Process(`ls /home/lurlurlur/.LurCGT_sqlite/local/`, ProcessExited(0))
```

```@example advanced_topics
script = expanduser("~/.LurCGT_sqlite/local/SU2")
run(`ls $script`)
```

Output:

```text
lurlurlur-MS-7C94_pid17002.db
lurlurlur-MS-7C94_pid17002.db-shm
lurlurlur-MS-7C94_pid17002.db-wal
lurlurlur-MS-7C94_pid20625.db
lurlurlur-MS-7C94_pid20625.db-shm
lurlurlur-MS-7C94_pid20625.db-wal
lurlurlur-MS-7C94_pid26611.db
lurlurlur-MS-7C94_pid26611.db-shm
lurlurlur-MS-7C94_pid26611.db-wal
lurlurlur-MS-7C94_pid28478.db
lurlurlur-MS-7C94_pid28478.db-shm
lurlurlur-MS-7C94_pid28478.db-wal
lurlurlur-MS-7C94_pid32003.db
lurlurlur-MS-7C94_pid32003.db-shm
lurlurlur-MS-7C94_pid32003.db-wal
lurlurlur-MS-7C94_pid32174.db
lurlurlur-MS-7C94_pid32174.db-shm
lurlurlur-MS-7C94_pid32174.db-wal
lurlurlur-MS-7C94_pid32750.db
lurlurlur-MS-7C94_pid32750.db-shm
lurlurlur-MS-7C94_pid32750.db-wal
lurlurlur-MS-7C94_pid36229.db
lurlurlur-MS-7C94_pid36229.db-shm
lurlurlur-MS-7C94_pid36229.db-wal
lurlurlur-MS-7C94_pid36521.db
lurlurlur-MS-7C94_pid36521.db-shm
lurlurlur-MS-7C94_pid36521.db-wal
Process(`ls /home/lurlurlur/.LurCGT_sqlite/local/SU2`, ProcessExited(0))
```

```@example advanced_topics
script = expanduser("~/.LurCGT_sqlite/global")
run(`ls $script`)
```

Output:

```text
SU2.db
SU2.db-shm
SU2.db-wal
SU3.db
SU3.db-shm
SU3.db-wal
locks
Process(`ls '~/.LurCGT_sqlite/global'`, ProcessExited(0))
```

### In server

!!! warning "Version note"
    Cluster database configuration may change in a future LurCGT release.

To use the LurCGT database system efficiently in a cluster, 4 environment variables are required.

<span style="color: red;">LURCGT_RUN_MODE=server</span>:
The variables below are activated when this is set to 'server'.

<span style="color: red;">LURCGT_LOCALDB_DIR_NODE</span>:
The location of the local database inside the computing node.

<span style="color: red;">LURCGT_GLOBALDB_DIR_NODE</span>:
The location of the global database inside the computing node.

<span style="color: red;">LURCGT_GLOBALDB_DIR</span>:
The location of the global database inside the head node.

The variables ending with '_NODE' are recommended to be inside the local scratch of the computing node for performance.

When the computation is requested through the Slurm command,

1. The global database in the head node (in LURCGT_GLOBALDB_DIR) is copied to the LURCGT_GLOBALDB_DIR_NODE directory.

2. A local database is created inside the node, and computation takes place using the local and global DBs in the node. 

3. After finishing, the local DB is merged into the global one in the head node.

4. DBs in the computing node are deleted when terminated normally.

The reason for copying the global DB is for performance. Getting data from the head node via NFS(network file system) is extremely slow.
