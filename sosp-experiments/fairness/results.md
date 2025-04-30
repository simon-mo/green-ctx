

MPS 100:

Single Server:
```
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  11.38
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              17.58
Output token throughput (tok/s):         1757.76
Total Token throughput (tok/s):          19335.39
---------------Time to First Token----------------
Mean TTFT (ms):                          42.76
Median TTFT (ms):                        36.79
P99 TTFT (ms):                           77.19
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.95
Median TPOT (ms):                        8.99
P99 TPOT (ms):                           11.25
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.00
Median ITL (ms):                         6.30
P99 ITL (ms):                            38.73
==================================================
```

Two Servers:
```
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  11.69
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              17.10
Output token throughput (tok/s):         1710.14
Total Token throughput (tok/s):          18811.50
---------------Time to First Token----------------
Mean TTFT (ms):                          79.27
Median TTFT (ms):                        70.32
P99 TTFT (ms):                           184.59
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.03
Median TPOT (ms):                        23.07
P99 TPOT (ms):                           25.27
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.03
Median ITL (ms):                         11.93
P99 ITL (ms):                            80.33
==================================================

============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  11.68
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              17.13
Output token throughput (tok/s):         1712.63
Total Token throughput (tok/s):          18838.94
---------------Time to First Token----------------
Mean TTFT (ms):                          81.93
Median TTFT (ms):                        74.07
P99 TTFT (ms):                           178.37
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.92
Median TPOT (ms):                        23.10
P99 TPOT (ms):                           25.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           20.92
Median ITL (ms):                         11.65
P99 ITL (ms):                            85.04
==================================================
```

```
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  11.71
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              17.07
Output token throughput (tok/s):         1707.22
Total Token throughput (tok/s):          18779.44
---------------Time to First Token----------------
Mean TTFT (ms):                          79.85
Median TTFT (ms):                        72.23
P99 TTFT (ms):                           162.35
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.36
Median TPOT (ms):                        23.88
P99 TPOT (ms):                           25.50
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.36
Median ITL (ms):                         11.98
P99 ITL (ms):                            77.15
==================================================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:11<00:00, 17.08it/s]
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  11.71
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              17.08
Output token throughput (tok/s):         1708.39
Total Token throughput (tok/s):          18792.33
---------------Time to First Token----------------
Mean TTFT (ms):                          79.80
Median TTFT (ms):                        75.24
P99 TTFT (ms):                           149.08
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.05
Median TPOT (ms):                        23.18
P99 TPOT (ms):                           25.18
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.05
Median ITL (ms):                         12.16
P99 ITL (ms):                            78.24
==================================================
```

GTX

* 64 SMs (0-3, 4-7)
```
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  12.54
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              15.95
Output token throughput (tok/s):         1594.73
Total Token throughput (tok/s):          17542.01
---------------Time to First Token----------------
Mean TTFT (ms):                          114.28
Median TTFT (ms):                        99.66
P99 TTFT (ms):                           249.22
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.02
Median TPOT (ms):                        27.06
P99 TPOT (ms):                           28.48
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.02
Median ITL (ms):                         11.77
P99 ITL (ms):                            112.71
==================================================

============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  12.47
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              16.04
Output token throughput (tok/s):         1604.49
Total Token throughput (tok/s):          17649.41
---------------Time to First Token----------------
Mean TTFT (ms):                          111.95
Median TTFT (ms):                        102.74
P99 TTFT (ms):                           217.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.85
Median TPOT (ms):                        26.89
P99 TPOT (ms):                           28.28
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.85
Median ITL (ms):                         11.67
P99 ITL (ms):                            109.79
==================================================
```

* 100 SMs (96 + 4 remaining)
```
============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  11.80
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              16.95
Output token throughput (tok/s):         1695.50
Total Token throughput (tok/s):          18650.45
---------------Time to First Token----------------
Mean TTFT (ms):                          86.32
Median TTFT (ms):                        77.55
P99 TTFT (ms):                           193.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.76
Median TPOT (ms):                        24.14
P99 TPOT (ms):                           25.72
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.76
Median ITL (ms):                         12.09
P99 ITL (ms):                            98.86
==================================================

============ Serving Benchmark Result ============
Successful requests:                     200
Benchmark duration (s):                  11.63
Total input tokens:                      200000
Total generated tokens:                  20000
Request throughput (req/s):              17.19
Output token throughput (tok/s):         1719.10
Total Token throughput (tok/s):          18910.14
---------------Time to First Token----------------
Mean TTFT (ms):                          86.17
Median TTFT (ms):                        75.41
P99 TTFT (ms):                           230.64
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.47
Median TPOT (ms):                        23.83
P99 TPOT (ms):                           25.66
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.47
Median ITL (ms):                         12.01
P99 ITL (ms):                            87.88
==================================================
```