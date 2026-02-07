#!/usr/bin/env python3
"""
Nsight Systems Trace Analyzer for MoE Expert GEMM Optimization

Processes .nsys-rep files to categorize CUDA kernels and calculate Amdahl's law
ceiling for expert GEMM optimization in Mixture-of-Experts models.

Usage:
    python analyze_nsys.py --input trace.nsys-rep --output results.json
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# Kernel categorization patterns (case-insensitive)
KERNEL_PATTERNS = {
    'expert_gemm': [
        r'.*cutlass.*moe.*',
        r'.*fused_moe.*',
        r'.*nvfp4_gemm.*',
        r'.*fp8.*gemm.*',
    ],
    'attention_gemm': [
        r'.*flash.*attn.*',
        r'.*fmha.*',
        r'.*gpt_attention.*',
    ],
    'gate_gemm': [
        r'.*cublas.*',  # Note: may need NVTX range filtering in future
    ],
    'routing_permute': [
        r'.*topk.*',
        r'.*sort.*',
        r'.*scatter.*',
        r'.*gather.*',
        r'.*permute.*',
        r'.*index.*',
    ],
    'quant_dequant': [
        r'.*quantize.*',
        r'.*dequant.*',
        r'.*cast.*',
        r'.*fp4.*quant.*',
        r'.*fp8.*quant.*',
    ],
    'norm': [
        r'.*rms_norm.*',
        r'.*layer_norm.*',
        r'.*rmsnorm.*',
    ],
    'allreduce': [
        r'.*nccl.*',
        r'.*allreduce.*',
        r'.*reduce_scatter.*',
    ],
}


def categorize_kernel(kernel_name: str) -> str:
    """Categorize a kernel by matching against patterns."""
    kernel_lower = kernel_name.lower()
    
    for category, patterns in KERNEL_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, kernel_lower):
                return category
    
    return 'other'


def parse_nsys_csv(csv_content: str) -> List[Dict[str, any]]:
    """Parse CSV output from nsys stats command."""
    kernels = []
    reader = csv.DictReader(csv_content.strip().split('\n'))
    
    for row in reader:
        # nsys stats CSV typically has columns like:
        # "Time(%)", "Total Time (ns)", "Instances", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)", "Name"
        # We need kernel name and total time
        
        # Handle different possible column names
        name = row.get('Name') or row.get('Kernel Name') or row.get('name')
        total_time_ns = row.get('Total Time (ns)') or row.get('Total Time(ns)') or row.get('total_time_ns')
        
        if name and total_time_ns:
            try:
                kernels.append({
                    'name': name.strip('"'),  # Remove quotes if present
                    'time_ns': float(total_time_ns.replace(',', ''))  # Remove commas
                })
            except (ValueError, AttributeError):
                continue
    
    return kernels


def extract_kernel_trace(nsys_file: Path) -> List[Dict[str, any]]:
    """Extract kernel trace from .nsys-rep file using nsys CLI."""
    if not nsys_file.exists():
        raise FileNotFoundError(f"Input file not found: {nsys_file}")
    
    # Run nsys stats to get GPU trace in CSV format
    cmd = [
        'nsys', 'stats',
        '--report', 'gputrace',
        '--format', 'csv',
        str(nsys_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # nsys stats outputs multiple sections; we need the kernel trace section
        # Look for the section that starts with kernel data
        csv_content = result.stdout
        
        # Find the start of the CSV data (after headers)
        # nsys typically outputs some metadata lines before the CSV
        lines = csv_content.split('\n')
        csv_start = 0
        
        for i, line in enumerate(lines):
            # Look for the header line with "Name" column
            if 'Name' in line and ('Time' in line or 'Duration' in line):
                csv_start = i
                break
        
        if csv_start > 0:
            csv_content = '\n'.join(lines[csv_start:])
        
        return parse_nsys_csv(csv_content)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running nsys stats: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("Error: 'nsys' command not found. Please ensure Nsight Systems is installed.", file=sys.stderr)
        raise


def analyze_kernels(kernels: List[Dict[str, any]]) -> Dict[str, Dict[str, any]]:
    """Categorize kernels and calculate statistics."""
    categories = defaultdict(lambda: {'time_ns': 0, 'count': 0, 'kernels': []})
    
    for kernel in kernels:
        category = categorize_kernel(kernel['name'])
        categories[category]['time_ns'] += kernel['time_ns']
        categories[category]['count'] += 1
        categories[category]['kernels'].append(kernel['name'])
    
    # Calculate total time
    total_time_ns = sum(cat['time_ns'] for cat in categories.values())
    
    # Add percentages
    results = {}
    for category, data in categories.items():
        time_ms = data['time_ns'] / 1_000_000
        percentage = (data['time_ns'] / total_time_ns * 100) if total_time_ns > 0 else 0
        
        results[category] = {
            'time_ms': time_ms,
            'time_ns': data['time_ns'],
            'percentage': percentage,
            'count': data['count'],
            'kernels': data['kernels']
        }
    
    # Add total
    results['TOTAL'] = {
        'time_ms': total_time_ns / 1_000_000,
        'time_ns': total_time_ns,
        'percentage': 100.0,
        'count': len(kernels),
        'kernels': []
    }
    
    return results


def print_summary_table(results: Dict[str, Dict[str, any]]):
    """Print formatted summary table to stdout."""
    # Define category order (excluding TOTAL which goes last)
    category_order = [
        'expert_gemm',
        'attention_gemm',
        'gate_gemm',
        'routing_permute',
        'quant_dequant',
        'norm',
        'allreduce',
        'other'
    ]
    
    # Print header
    print("\n" + "="*80)
    print("CUDA Kernel Analysis - Amdahl's Law Breakdown")
    print("="*80)
    print(f"{'Category':<20} | {'Time (ms)':>10} | {'% of Total':>10} | {'Kernel Count':>12}")
    print("-"*80)
    
    # Print categories in order
    for category in category_order:
        if category in results:
            data = results[category]
            print(f"{category:<20} | {data['time_ms']:>10.2f} | {data['percentage']:>9.1f}% | {data['count']:>12}")
    
    # Print total
    print("-"*80)
    total = results['TOTAL']
    print(f"{'TOTAL':<20} | {total['time_ms']:>10.2f} | {total['percentage']:>9.1f}% | {total['count']:>12}")
    print("="*80)
    
    # Print Amdahl's law note
    if 'expert_gemm' in results:
        expert_pct = results['expert_gemm']['percentage']
        max_speedup = 1 / (1 - expert_pct / 100) if expert_pct < 100 else float('inf')
        print(f"\nAmdahl's Law Analysis:")
        print(f"  Expert GEMM is {expert_pct:.1f}% of total execution time")
        print(f"  Maximum theoretical speedup from expert optimization: {max_speedup:.2f}x")
        print(f"  (Assumes expert GEMM can be made infinitely fast)")
    print()


def save_json_output(results: Dict[str, Dict[str, any]], output_path: Path):
    """Save results to JSON file with Amdahl's law note."""
    # Prepare output structure
    output = {
        'categories': {},
        'total': results['TOTAL'],
        'amdahls_law_note': None
    }
    
    # Add categories (excluding TOTAL)
    for category, data in results.items():
        if category != 'TOTAL':
            output['categories'][category] = {
                'time_ms': data['time_ms'],
                'percentage': data['percentage'],
                'count': data['count'],
                'sample_kernels': data['kernels'][:5]  # Include first 5 kernel names as samples
            }
    
    # Add Amdahl's law note
    if 'expert_gemm' in results:
        expert_pct = results['expert_gemm']['percentage']
        max_speedup = 1 / (1 - expert_pct / 100) if expert_pct < 100 else float('inf')
        output['amdahls_law_note'] = (
            f"If expert_gemm is {expert_pct:.1f}% of total, "
            f"max speedup from expert precision change is {max_speedup:.2f}x "
            f"(calculated as 1/(1-{expert_pct:.1f}/100))"
        )
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Nsight Systems traces for MoE expert GEMM optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input trace.nsys-rep --output results.json
  %(prog)s -i profile.nsys-rep -o analysis.json

This tool categorizes CUDA kernels to calculate Amdahl's law ceiling for
optimizing expert GEMM operations in Mixture-of-Experts models.
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to input .nsys-rep file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Path to output JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        # Extract kernel trace
        print(f"Extracting kernel trace from: {args.input}")
        kernels = extract_kernel_trace(args.input)
        print(f"Found {len(kernels)} kernels")
        
        # Analyze and categorize
        print("Categorizing kernels...")
        results = analyze_kernels(kernels)
        
        # Print summary table
        print_summary_table(results)
        
        # Save JSON output
        save_json_output(results, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
