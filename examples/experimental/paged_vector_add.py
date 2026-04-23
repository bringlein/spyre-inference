# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from torch.profiler import profile, ProfilerActivity

# debug = True
debug = False

if debug:
    import debugpy
    host_addr = os.environ.get("TORCH_SPYRE_DEBUG_ADDR", "0.0.0.0")
    pdb_port = int(os.environ.get("TORCH_SPYRE_DEBUG_PORT", "5679"))
    debugpy.listen((host_addr, pdb_port))
    print(f"[debugpy] listening at {host_addr}:{pdb_port}; wait for client...\n")
    debugpy.wait_for_client()
    print("[debugpy] connected")

DEVICE = torch.device("spyre")

def create_paged_memory(
    page_size: int, number_of_pages: int, fill_value: float = 0.0, dtype=torch.float16
):
    list_of_pages = []
    for i in range(number_of_pages):
        new_page = torch.full([page_size], fill_value, dtype=dtype)
        new_page_device = new_page.to(DEVICE)
        list_of_pages.append(new_page_device)
    return list_of_pages


print("create paged memory...")
PAGE_SIZE = 16
# create paged memory 
a_pages = create_paged_memory(PAGE_SIZE, 512, 1.0)
b_pages = create_paged_memory(PAGE_SIZE, 512, 2.0)

# test output also as paged
out_pages = create_paged_memory(PAGE_SIZE, 256)


# manipulate individual pages 

a_pages[42].fill_(40.0)
b_pages[312].fill_(31.0)

# 0 page as padding index
a_pages[0].fill_(0.0)
b_pages[0].fill_(0.0)

print("prepare computation...")


def profile_and_print(func, name, num_runs=10):
    """Helper to profile a function and extract CPU total time"""
    # Warmup
    for _ in range(3):
        func()
    
    # Profile
    print(f"\nProfiling {name}...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
    ) as prof:
        for _ in range(num_runs):
            func()
    
    table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    table_cpu_display = table_cpu.replace("CUDA", "Spyre")
    
    print(f"\n{name} - Sorted by CPU time:")
    print(table_cpu_display)
    
    # Extract total CPU time from the profiler
    total_cpu_time = sum(evt.cpu_time_total for evt in prof.key_averages())
    return total_cpu_time / num_runs  # Average per run


def paged_vector_add(a_pages, b_pages, page_table, max_page_table_length, out_pages):
    # output pages are starting from 0, in this case
    for i in range(max_page_table_length):
        page_index = page_table[i]
        a_page = a_pages[page_index]
        b_page = b_pages[page_index]
        out_page = out_pages[i]
        sum_result = a_page + b_page
        out_page.copy_(sum_result)

# There are two ways to make paged_vector_add compilable:
#   1. Freeze only the length of page_table, and pass page_table at runtime. 
#        The page table remains on the CPU, and we have a sync after every iteration.
#   2. Freeze the entire page_table at compile time
#        To avoid the overhead of soultion 1, we can freeze the entire page_table at compile time. 
#   We try both to asses the overhead. 
#   The version with the page table on the device and using is not compilable (at the moment?).

def create_compilable_paged_vector_add_cpu(page_table_length):
    """For CPU list/tensor: freeze only the length, pass page_table at runtime"""
    def paged_vector_add_with_fixed_length(a_pages, b_pages, page_table, out_pages):
        # Assert length matches to help compiler and validate input
        assert len(page_table) == page_table_length, (
            f"Expected page_table length {page_table_length}, got {len(page_table)}"
        )
        return paged_vector_add(
            a_pages, b_pages, page_table, page_table_length, out_pages
        )
    return paged_vector_add_with_fixed_length


def create_compilable_paged_vector_add_frozen(page_table):
    """Freeze the entire page_table for full compile-time optimization"""
    page_table_length = len(page_table)
    
    def paged_vector_add_with_frozen_table(a_pages, b_pages, out_pages):
        return paged_vector_add(
            a_pages, b_pages, page_table, page_table_length, out_pages
        )
    return paged_vector_add_with_frozen_table


list_of_compiled_functions_per_request_length = {}

# Example 1: Using CPU list - freeze only length
print("\n" + "="*60)
print("Example 1: CPU list with frozen length")
print("="*60)
page_table_list = [13, 312, 42, 14, 15]
page_table_length = len(page_table_list)
compiled_dynamic = torch.compile(
    create_compilable_paged_vector_add_cpu(page_table_length)
)

# Execute once and print result
print("Executing once to verify correctness...")
compiled_dynamic(a_pages, b_pages, page_table_list, out_pages)

print("Result (dynamic page table):")
# Expected result:
#  0: 3.0
#  1: 32.0
#  2: 42.0
#  3: 3.0
#  4: 3.0

out_pages_cpu = [p.cpu() for p in out_pages]
for i in range(page_table_length):
    print(f"  out page {i}: {out_pages_cpu[i].tolist()}")

# Profile
avg_cpu_time_dynamic = profile_and_print(
    lambda: compiled_dynamic(a_pages, b_pages, page_table_list, out_pages),
    "Dynamic page table"
)


print("\n" + "="*60)
print("Example 2: Frozen page table (CPU list)")
print("="*60)

# Example 2: Freeze entire page table at compile time for maximum optimization
#   and to asses the overhead of Example 1
page_table_2 = [500, 501, 312, 0, 0]

# print(f"page_table_2: {page_table_2}")

# Compile with frozen table - compiler can inline/unroll loop with known indices
compiled_frozen = torch.compile(
    create_compilable_paged_vector_add_frozen(page_table_2)
)

# Execute once and print result
print("Executing once to verify correctness...")
compiled_frozen(a_pages, b_pages, out_pages)

print("Result (frozen page table):")
# Expected result:
#  0: 3.0
#  1: 3.0
#  2: 32.0
#  3: 0.0
#  4: 0.0

out_pages_cpu = [p.cpu() for p in out_pages]
for i in range(len(page_table_2)):
    print(f"  out page {i}: {out_pages_cpu[i].tolist()}")

# Profile
avg_cpu_time_frozen = profile_and_print(
    lambda: compiled_frozen(a_pages, b_pages, out_pages),
    "Frozen page table"
)

# Compare
print("\n" + "="*60)
print("Performance Comparison")
print("="*60)
print(f"Dynamic page table - Avg CPU time: {avg_cpu_time_dynamic/1000:.4f} ms")
print(f"Frozen page table  - Avg CPU time: {avg_cpu_time_frozen/1000:.4f} ms")
if avg_cpu_time_dynamic > avg_cpu_time_frozen:
    speedup = avg_cpu_time_dynamic / avg_cpu_time_frozen
    print(f"Speedup (frozen vs dynamic): {speedup:.2f}x")
else:
    slowdown = avg_cpu_time_frozen / avg_cpu_time_dynamic
    print(f"Slowdown (frozen vs dynamic): {slowdown:.2f}x")
