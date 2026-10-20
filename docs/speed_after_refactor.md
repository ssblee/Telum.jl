** Never remove here **
Before refactoring:
DMRG test with Nkeep=20
35.022891 seconds (468.10 M allocations: 28.482 GiB, 8.99% gc time)
DMRG test with Nkeep=50
52.071013 seconds (656.60 M allocations: 42.579 GiB, 9.55% gc time)

contract (U1+SU2+SU3, 100 times)
0.317521 seconds (6.75 M allocations: 380.511 MiB, 14.01% gc time)
contract (U1+SU2 Nkeep=1000, 20 times)
1.205918 seconds (1.66 M allocations: 1.373 GiB, 49.80% gc time)



After refactoring:
DMRG test with Nkeep=20
51.359630 seconds (516.64 M allocations: 25.229 GiB, 32.35% gc time)
DMRG test with Nkeep=50
75.521979 seconds (729.36 M allocations: 37.642 GiB, 34.60% gc time)

contract (U1+SU2+SU3, 100 times)
0.467746 seconds (13.55 M allocations: 640.902 MiB, 12.98% gc time)
contract (U1+SU2 Nkeep=1000, 20 times)
1.520908 seconds (2.08 M allocations: 1.376 GiB, 46.15% gc time)
** 10% garbage collecting time for half cases


After CGR patch & q-index matrix transpose:
DMRG test with Nkeep=20
29.249116 seconds (434.99 M allocations: 21.852 GiB, 8.83% gc time)
DMRG test with Nkeep=50
40.257669 seconds (606.28 M allocations: 32.563 GiB, 8.49% gc time)


After WMAT patch (Matrix of w-matrices)
DMRG test with Nkeep=20
33.789782 seconds (426.92 M allocations: 21.536 GiB, 6.41% gc time)
DMRG test with Nkeep=50
46.795861 seconds (595.61 M allocations: 31.588 GiB, 7.26% gc time)


After contraction patch 1 (selecting spaces)
DMRG test with Nkeep=20
31.999989 seconds (410.04 M allocations: 20.025 GiB, 6.23% gc time)
DMRG test with Nkeep=50
45.462576 seconds (568.72 M allocations: 29.845 GiB, 7.27% gc time)


After WMAT patch (Vector of tuples of w-matrices)
DMRG test with Nkeep=20
32.581202 seconds (427.60 M allocations: 20.636 GiB, 6.12% gc time)
DMRG test with Nkeep=50
45.791973 seconds (590.19 M allocations: 30.601 GiB, 7.12% gc time)
contract (U1+SU2+SU3, 100 times)
0.440693 seconds (10.63 M allocations: 505.703 MiB, 10.30% gc time)
contract (U1+SU2 Nkeep=1000, 20 times)
1.308961 seconds (1.64 M allocations: 1.340 GiB, 50.23% gc time)


After contraction patch 2 (Contracting X-symbol, w-matrices, and RMTs)
DMRG test with Nkeep=20
27.140981 seconds (371.08 M allocations: 17.855 GiB, 6.48% gc time)
DMRG test with Nkeep=50
37.929842 seconds (505.56 M allocations: 26.475 GiB, 6.62% gc time)
contract (U1+SU2+SU3, 100 times)
0.248160 seconds (5.27 M allocations: 228.083 MiB, 23.28% gc time) (No garbage collection on about half of calling)
contract (U1+SU2 Nkeep=1000, 20 times)
1.083793 seconds (1.01 M allocations: 1.312 GiB, 57.84% gc time) (No garbage collection on about 1/3 of calling)

After sum patch
DMRG test with Nkeep=20
23.674097 seconds (321.23 M allocations: 14.637 GiB, 6.26% gc time)
DMRG test with Nkeep=50
32.163440 seconds (432.54 M allocations: 21.105 GiB, 7.83% gc time)

After diagonal RMT type
DMRG test with Nkeep=20
23.181192 seconds (315.87 M allocations: 14.575 GiB, 6.97% gc time)
DMRG test with Nkeep=50
30.981983 seconds (425.09 M allocations: 21.077 GiB, 8.81% gc time)
SVD test (Nkeep=50, run same svd 100 times, without truncation)
0.893499 seconds (10.95 M allocations: 533.433 MiB, 6.29% gc time)

After SVD patch 
DMRG test with Nkeep=20
22.582309 seconds (309.27 M allocations: 14.355 GiB, 6.53% gc time)
DMRG test with Nkeep=50
30.516837 seconds (415.18 M allocations: 20.740 GiB, 6.68% gc time)
SVD test (Nkeep=50, run same svd 100 times, without truncation)
0.571761 seconds (4.94 M allocations: 302.402 MiB, 7.73% gc time)

After view patch (no wasted memory for w-mat part of conj)
DMRG test with Nkeep=20
17.047193 seconds (255.02 M allocations: 14.041 GiB, 7.61% gc time)
DMRG test with Nkeep=50
25.115376 seconds (346.66 M allocations: 20.691 GiB, 7.50% gc time)

After view patch (more optimized summation)
DMRG test with Nkeep=20
16.796584 seconds (253.92 M allocations: 14.196 GiB, 7.68% gc time)
DMRG test with Nkeep=50
23.549438 seconds (345.02 M allocations: 20.791 GiB, 7.97% gc time)

After in-memory key update (string -> tuple)
DMRG test with Nkeep=20
15.321062 seconds (172.20 M allocations: 11.513 GiB, 7.09% gc time)
DMRG test with Nkeep=50
21.625024 seconds (241.16 M allocations: 17.384 GiB, 7.31% gc time)

After w-matrices arena (naive update)
DMRG test with Nkeep=20
15.884589 seconds (172.53 M allocations: 11.862 GiB, 6.94% gc time)
DMRG test with Nkeep=50
22.978654 seconds (259.42 M allocations: 19.247 GiB, 7.03% gc time)

DMRG test with Nkeep=200
42.792113 seconds (356.66 M allocations: 42.233 GiB, 13.57% gc time)
DMRG test with Nkeep=400
72.809350 seconds (410.29 M allocations: 91.122 GiB, 24.70% gc time)
DMRG test with Nkeep=800
180.076534 seconds (452.87 M allocations: 267.933 GiB, 35.44% gc time)

After streaming in contraction
DMRG test with Nkeep=20
15.971497 seconds (179.91 M allocations: 12.548 GiB, 6.80% gc time)
DMRG test with Nkeep=50
22.380263 seconds (252.85 M allocations: 18.548 GiB, 6.79% gc time)

DMRG test with Nkeep=200
38.752024 seconds (345.97 M allocations: 35.539 GiB, 11.60% gc time)
DMRG test with Nkeep=400
64.509287 seconds (397.46 M allocations: 67.499 GiB, 20.17% gc time)
DMRG test with Nkeep=800
146.136179 seconds (438.09 M allocations: 179.115 GiB, 29.83% gc time)


After lazy-evaluation on contraction patch
DMRG test with Nkeep=200
40.614229 seconds (334.09 M allocations: 27.938 GiB, 11.39% gc time)
DMRG test with Nkeep=400
62.773015 seconds (383.24 M allocations: 55.937 GiB, 19.61% gc time)
DMRG test with Nkeep=800
143.957918 seconds (421.97 M allocations: 155.790 GiB, 27.02% gc time)

After @lazy macro & lazy-evaluation on getsub
DMRG test with Nkeep=200
36.400808 seconds (347.79 M allocations: 28.672 GiB, 9.69% gc time)
DMRG test with Nkeep=400
58.233621 seconds (399.86 M allocations: 56.852 GiB, 18.29% gc time)
DMRG test with Nkeep=800
131.892585 seconds (441.30 M allocations: 157.120 GiB, 29.06% gc time)
