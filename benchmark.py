"""
RMSNorm Benchmark

Compares 3 implementations:
- PyTorch native (torch.nn.functional.rms_norm)
- PyTorch compiled (torch.compile)
- Triton optimized kernel
"""

import torch
import torch._dynamo
import triton
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from rmsnorm import (
    pytorch_native_rmsnorm,
    pytorch_compiled_rmsnorm,
    triton_rmsnorm,
    TORCH_COMPILE_AVAILABLE,
)

# Increase dynamo cache limit to handle multiple tensor sizes
torch._dynamo.config.cache_size_limit = 64


def get_gpu_info():
    """Get GPU information and theoretical peak bandwidth."""
    props = torch.cuda.get_device_properties(0)

    bandwidth_map = {
        "RTX 3050 Laptop": 192,  # 128-bit bus, 12 GT/s GDDR6
        "RTX 3050": 224,
        "RTX 3060": 360,
        "RTX 3070": 448,
        "RTX 3080": 760,
        "RTX 3090": 936,
        "RTX 4070": 504,
        "RTX 4080": 717,
        "RTX 4090": 1008,
        "A100": 2039,
        "H100": 3350,
    }

    gpu_name = props.name
    for key, bw in bandwidth_map.items():
        if key.lower() in gpu_name.lower():
            peak_bw = bw
            break
    else:
        peak_bw = 200

    return {
        "name": gpu_name,
        "compute_capability": props.major * 10 + props.minor,
        "peak_bandwidth": peak_bw,
    }


def do_bench(fn, warmup=25, rep=100):
    """Benchmark using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / rep


def get_bandwidth(M, N, time_ms):
    """Calculate memory bandwidth in GB/s."""
    bytes_per_elem = 4
    total_bytes = (2 * M * N + N) * bytes_per_elem
    return total_bytes / (time_ms * 1e-3) / 1e9


def get_implementations():
    """Get available implementations based on environment."""
    impls = {
        "PyTorch Native": pytorch_native_rmsnorm,
        "Triton": triton_rmsnorm,
    }
    if TORCH_COMPILE_AVAILABLE:
        impls["torch.compile"] = pytorch_compiled_rmsnorm
    return impls


def verify_correctness(M=1024, N=128, eps=1e-6):
    """Verify all implementations match PyTorch reference."""
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)

    ref = pytorch_native_rmsnorm(x, weight, eps)

    implementations = get_implementations()

    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    if not TORCH_COMPILE_AVAILABLE:
        print("  (torch.compile unavailable on Python 3.14+)")

    all_passed = True
    for name, fn in implementations.items():
        out = fn(x, weight, eps)
        max_err = (out - ref).abs().max().item()
        passed = max_err < 1e-4
        status = "PASS" if passed else "FAIL"
        print(f"  {name:15s}: max_error={max_err:.2e} [{status}]")
        if not passed:
            all_passed = False

    print()
    return all_passed


def benchmark_size(M, N=128, eps=1e-6):
    """Benchmark all implementations for a given size."""
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)

    implementations = get_implementations()

    results = {"M": M, "N": N}

    for name, fn in implementations.items():
        time_ms = do_bench(lambda fn=fn: fn(x, weight, eps))
        bandwidth = get_bandwidth(M, N, time_ms)
        results[name] = bandwidth

    return results


def print_results(all_results, peak_bw):
    """Print benchmark results in ASCII table."""
    print("\n" + "=" * 60)
    print(f"Results Summary (at M=1,048,576)")
    print("=" * 60)

    large_result = next((r for r in all_results if r["M"] == 1048576), None)
    if large_result is None:
        large_result = all_results[-1]

    print(f"\n{'Method':<20s} {'Bandwidth':<15s} {'% of Peak':<15s}")
    print("-" * 60)

    for name, bw in sorted(
        large_result.items(), key=lambda x: -x[1] if isinstance(x[1], float) else 0
    ):
        if name in ["M", "N"]:
            continue
        pct = 100 * bw / peak_bw
        print(f"{name:<20s} {bw:>10.1f} GB/s {pct:>10.1f}%")

    print(f"\nTheoretical Peak: {peak_bw:.0f} GB/s")


def create_plot(all_results, peak_bw, output_path):
    """Create line plot showing bandwidth vs problem size."""
    df = pd.DataFrame(all_results)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define all possible methods and their styles
    method_styles = {
        "PyTorch Native": {"color": "#ff7f0e", "marker": "s"},
        "torch.compile": {"color": "#2ca02c", "marker": "^"},
        "Triton": {"color": "#1f77b4", "marker": "o"},
    }

    # Only plot methods that exist in the results
    for method, style in method_styles.items():
        if method in df.columns:
            ax.plot(
                df["M"],
                df[method],
                marker=style["marker"],
                label=method,
                color=style["color"],
                linewidth=2,
                markersize=6,
            )

    ax.axhline(
        y=peak_bw,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Theoretical Peak ({peak_bw} GB/s)",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (number of rows)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12, fontweight="bold")
    ax.set_title(
        "RMSNorm Performance: Bandwidth vs Problem Size\n(RTX 3050 Laptop, N=128)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0, peak_bw * 1.15)

    # Format x-axis ticks
    ax.set_xticks(df["M"])
    ax.set_xticklabels(
        [f"{m // 1024}K" if m >= 1024 else str(m) for m in df["M"]],
        rotation=45,
        ha="right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def main():
    gpu_info = get_gpu_info()

    print("=" * 60)
    print("RMSNorm Triton Benchmark")
    print("=" * 60)
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Theoretical Peak Bandwidth: ~{gpu_info['peak_bandwidth']} GB/s")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")

    if not verify_correctness():
        print("\nERROR: Verification failed!")
        return

    print("=" * 60)
    print("Benchmarking")
    print("=" * 60)

    N = 128
    sizes = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
    ]

    all_results = []
    for M in sizes:
        results = benchmark_size(M, N)
        all_results.append(results)

        best = max(
            [(k, v) for k, v in results.items() if k not in ["M", "N"]],
            key=lambda x: x[1],
        )
        print(f"M={M:8d} | Best: {best[0]:15s} @ {best[1]:6.1f} GB/s")

    df = pd.DataFrame(all_results)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    plot_path = output_dir / "benchmark.png"
    create_plot(all_results, gpu_info["peak_bandwidth"], plot_path)

    print_results(all_results, gpu_info["peak_bandwidth"])


if __name__ == "__main__":
    main()
