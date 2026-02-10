cd ~ && python /home/jerry/Documents/TensorRT-LLM/benchmarks/qwen3_moe/ppl_bench.py \
  --model_path_bf16 Qwen/Qwen3-30B-A3B \
  --model_path_fp8 Qwen/Qwen3-30B-A3B-FP8 \
  --model_path_nvfp4 nvidia/Qwen3-30B-A3B-FP4 \
  --precisions nvfp4 \
  --input_len 512 \
  --num_chunks 100 \
  --output_dir ./benchmark_results



python benchmarks/qwen3_moe/ppl_bench.py \
  --model_path_bf16 Qwen/Qwen3-30B-A3B \
  --model_path_fp8 Qwen/Qwen3-30B-A3B-FP8 \
  --model_path_nvfp4 nvidia/Qwen3-30B-A3B-FP4 \
  --precisions nvfp4 fp8 \
  --input_len 512 \
  --num_chunks 10000 \
  --output_dir ./benchmark_results

python benchmarks/qwen3_moe/ppl_bench.py   --model_path_bf16 Qwen/Qwen3-30B-A3B   --model_path_nvfp4 nvidia/Qwen3-30B-A3B-FP4   --precisions bf16 nvfp4    --input_len 512   --num_chunks 10000   --output_dir ./benchmark_results