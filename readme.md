This repository contains scripts used to generate the performance plots for
[rnull](https://rust-for-linux.com/null-block-driver) and
[rnvme](https://rust-for-linux.com/nvme-driver) that are published periodically
on [rust-for-linux.com](https://rust-for-linux.com/).

The benchmark is [fio](https://github.com/axboe/fio) executed by
[fio-matrix](https://github.com/metaspace/fio-matrix). For for `rnull` the
following configuration is used:

```toml
prep = true
samples = 40
runtime = 30
ramp = 10
cpufreq_governor_performance = true
use_hugepages = true
jobcounts = [ 1, 2, 6 ]
workloads = [ "randread", "randwrite" ]
block_sizes = [ "4KiB", "32KiB", "256KiB", "1MiB", "16MiB" ]
queue_depths = [ 1, 8, 32, 128 ]
device = "rnullb0"
module = "rnull_mod"
module_args = [ "memory_backed=1", "param_memory_backed=1" ]
modprobe = true
module_reload_policy = "Always"
```

For `rnvme` the following configuration is used:

```toml
prep = false
samples = 40
runtime = 30
ramp = 10
cpufreq_governor_performance = true
use_hugepages = true
jobcounts = [ 1 ]
hipri = true
workloads = [ "randread" ]
block_sizes = [ "512", "4KiB", "32KiB", "256KiB", "1MiB", "16MiB" ]
queue_depths = [ 1, 8, 32, 128 ]
device = "nvme0n1"
module = "rnvme"
module_args = [
  "poll_queue_count=1",
  "irq_queue_count=1",
]
modprobe = true
module_reload_policy = "Once"
```
