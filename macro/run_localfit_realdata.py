#!/usr/bin/env python3
"""
Run LocalFitBaseline on real data with artificial masking.

Generates the same artificial mask pattern used by
compute_inpainter_baselines.py, writes a per-event dead channel file,
and runs LocalFitBaseline.C via ROOT.

Usage:
    python macro/run_localfit_realdata.py \\
        --input val_data/data/DataGammaAngle_430026-430126.root \\
        --dead-channel-file dead_channels_run430000.txt \\
        --output localfit_realdata.root

    # Custom masking (must match compute_inpainter_baselines.py settings)
    python macro/run_localfit_realdata.py \\
        --input val_data/data/DataGammaAngle_430026-430126.root \\
        --dead-channel-file dead_channels_run430000.txt \\
        --n-artificial "inner:10,us:1,ds:1,outer:1,top:1,bot:1" \\
        --seed 42 \\
        --output localfit_realdata.root
"""

import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_n_artificial(value):
    try:
        return int(value)
    except ValueError:
        result = {}
        for part in value.split(','):
            face, count = part.strip().split(':')
            result[face.strip()] = int(count.strip())
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Run LocalFitBaseline on real data with artificial masking")
    parser.add_argument("--input", "-i", required=True,
                        help="Input ROOT file (from PrepareRealData)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output ROOT file for LocalFit predictions")

    dead_group = parser.add_mutually_exclusive_group(required=True)
    dead_group.add_argument("--run", type=int,
                            help="Run number for dead channels from database")
    dead_group.add_argument("--dead-channel-file", type=str,
                            help="Dead channel list file")

    parser.add_argument("--n-artificial", type=str,
                        default="inner:10,us:1,ds:1,outer:1,top:1,bot:1",
                        help="Artificial masks per event (default: stratified 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for artificial masking (default: 42)")
    parser.add_argument("--tree-name", type=str, default="tree")

    args = parser.parse_args()

    # --- Load data to generate artificial mask ---
    print("[INFO] Loading data to generate artificial mask...")
    from lib.dataset import expand_path
    from validate_inpainter import get_dead_channels

    import uproot
    file_list = expand_path(args.input)
    all_npho = []
    for fpath in file_list:
        with uproot.open(fpath) as f:
            tree = f[args.tree_name]
            npho_branch = "npho"
            if npho_branch not in tree.keys() and "relative_npho" in tree.keys():
                npho_branch = "relative_npho"
            all_npho.append(tree[npho_branch].array(library='np'))
    raw_npho = np.concatenate(all_npho).astype(np.float32)
    n_events = len(raw_npho)
    print(f"[INFO] {n_events} events")

    # --- Dead channels ---
    dead_channels = get_dead_channels(
        run_number=args.run,
        dead_channel_file=args.dead_channel_file,
    )
    dead_mask = np.zeros(4760, dtype=bool)
    dead_mask[dead_channels] = True
    print(f"[INFO] Dead channels: {len(dead_channels)}")

    # --- Generate artificial mask (same as compute_inpainter_baselines.py) ---
    n_artificial = parse_n_artificial(args.n_artificial)
    rng = np.random.default_rng(args.seed)
    N_CHANNELS = 4760

    from validate_inpainter import get_face_sensor_ids, FACE_NAME_TO_INT
    invalid = (raw_npho > 9e9) | np.isnan(raw_npho)

    # Write per-event dead channel file
    # Format: event_index sensor_id (one pair per line)
    tmpdir = tempfile.mkdtemp(prefix="localfit_")
    perevent_path = os.path.join(tmpdir, "artificial_mask_perevent.txt")

    n_total_masks = 0
    with open(perevent_path, 'w') as f:
        f.write("# event_idx sensor_id\n")
        if isinstance(n_artificial, dict):
            face_sensor_ids = {
                fname: get_face_sensor_ids(fname) for fname in n_artificial
            }
            for i in range(n_events):
                for fname, n_per_face in n_artificial.items():
                    if n_per_face <= 0:
                        continue
                    sids = face_sensor_ids[fname]
                    valid = ~dead_mask[sids] & ~invalid[i, sids]
                    valid_sids = sids[valid]
                    if len(valid_sids) >= n_per_face:
                        chosen = rng.choice(valid_sids, size=n_per_face,
                                            replace=False)
                        for s in chosen:
                            f.write(f"{i} {s}\n")
                        n_total_masks += n_per_face
        else:
            for i in range(n_events):
                valid = ~dead_mask & ~invalid[i]
                valid_indices = np.where(valid)[0]
                if len(valid_indices) > n_artificial:
                    chosen = rng.choice(valid_indices, size=n_artificial,
                                        replace=False)
                    for s in chosen:
                        f.write(f"{i} {s}\n")
                    n_total_masks += n_artificial

    print(f"[INFO] Generated {n_total_masks} artificial masks "
          f"({n_total_masks / n_events:.1f} per event)")
    print(f"[INFO] Per-event mask file: {perevent_path}")

    # --- Run LocalFitBaseline.C ---
    # Pass empty string for deadFile (no global dead channels — only per-event)
    macro_path = os.path.join(os.path.dirname(__file__), "..",
                              "others", "LocalFitBaseline.C")
    macro_path = os.path.abspath(macro_path)
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    cmd = (f'root -l -b -q \'{macro_path}'
           f'("{input_path}", "", "{output_path}", "{perevent_path}")\'')

    print(f"[INFO] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)

    # Cleanup temp file
    try:
        os.unlink(perevent_path)
        os.rmdir(tmpdir)
    except OSError:
        pass

    if result.returncode != 0:
        print(f"[ERROR] LocalFitBaseline failed (exit {result.returncode})")
        sys.exit(1)

    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Use with: python macro/compare_inpainter.py "
          f"--mode data --localfit {output_path}")


if __name__ == "__main__":
    main()
