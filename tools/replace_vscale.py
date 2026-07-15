import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

FP8_MAX = 448.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--kv_statistics_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Load kv statistics
    stats_file = os.path.join(args.kv_statistics_path, "activation_stats_per_head.json")
    with open(stats_file, "r") as f:
        stats_data = json.load(f)

    # Load index
    index_file = os.path.join(args.input_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        index_data = json.load(f)

    # Find all v_scale keys and compute new scales
    weight_map = index_data["weight_map"]
    v_scale_keys = [k for k in weight_map if "v_scale" in k]

    new_v_scales = {}
    for key in v_scale_keys:
        # Extract layer_id from key like "model.layers.0.self_attn.v_scale"
        parts = key.split(".")
        layer_id = parts[2]

        stats_key = f"model.layers.{layer_id}.self_attn.attn.v_cache"
        mins = stats_data[stats_key]["min"]
        maxs = stats_data[stats_key]["max"]

        num_heads = len(mins)
        scales = []
        for head_id in range(num_heads):
            scale = max(abs(mins[head_id]), abs(maxs[head_id])) / FP8_MAX
            scales.append(scale)

        new_v_scales[key] = torch.tensor(scales, dtype=torch.float32)

    # Save new v_scales
    print(f"Computed new v_scales for {len(v_scale_keys)} layers, saving...")
    save_file(new_v_scales, os.path.join(args.output_path, "v_scales.safetensors"))
    print("v_scales.safetensors saved.")

    # Update index: point v_scale keys to new file
    for key in v_scale_keys:
        weight_map[key] = "v_scales.safetensors"

    with open(os.path.join(args.output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    # Copy all other files in parallel
    files_to_copy = []
    for fname in os.listdir(args.input_path):
        if fname == "model.safetensors.index.json":
            continue
        src = os.path.join(args.input_path, fname)
        dst = os.path.join(args.output_path, fname)
        if os.path.isfile(src):
            files_to_copy.append((src, dst))

    import time
    print(f"Copying {len(files_to_copy)} files in parallel...")
    copied_count = [0]
    start_time = time.time()
    last_report_time = [start_time]

    def copy_file(pair):
        shutil.copy2(pair[0], pair[1])
        copied_count[0] += 1
        now = time.time()
        if copied_count[0] % 10 == 0 or copied_count[0] == len(files_to_copy):
            elapsed_since_last = now - last_report_time[0]
            total_elapsed = now - start_time
            avg_per_file = total_elapsed / copied_count[0]
            remaining = avg_per_file * (len(files_to_copy) - copied_count[0])
            print(f"  Copied {copied_count[0]}/{len(files_to_copy)} files | last 10: {elapsed_since_last:.1f}s | ETA: {remaining:.0f}s")
            last_report_time[0] = now

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(copy_file, files_to_copy))

    print(f"Done. Output saved to {args.output_path}")
    print(f"Replaced {len(v_scale_keys)} v_scale entries from kv_statistics.")


if __name__ == "__main__":
    main()
