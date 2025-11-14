#!/usr/bin/env python3
"""
run_pipeline.py
--------------------------------
CLI entrypoint for running the Mesh Processing Pipeline.

Usage examples:
  python run_pipeline.py
  python run_pipeline.py --data_root "C:/path/to/8samples" --out_root "./outputs"
  python run_pipeline.py --adaptive     # run adaptive quantization prototype (bonus)
--------------------------------
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Setup: ensure src/ is in path
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# ---------------------------------------------------------------------
# Imports from src
# ---------------------------------------------------------------------
from src.pipeline import process_all_meshes, process_mesh_adaptive
from src.loader import find_mesh_files

# ---------------------------------------------------------------------
# CLI Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Mesh Quantization & Reconstruction Pipeline")
    parser.add_argument("--data_root", type=str,
                        default=str(PROJECT_ROOT.parent / "8samples"),
                        help="Path to dataset folder containing meshes (.obj/.ply/.stl)")
    parser.add_argument("--out_root", type=str,
                        default=str(PROJECT_ROOT / "outputs"),
                        help="Output directory for results (will be created if not exists)")
    parser.add_argument("--normalizer", type=str,
                        default="minmax", choices=["minmax", "unitsphere"],
                        help="Normalization method (minmax or unitsphere)")
    parser.add_argument("--n_bins", type=int,
                        default=1024,
                        help="Number of quantization bins for uniform quantizer")
    parser.add_argument("--adaptive", action="store_true",
                        help="Run adaptive quantization prototype (bonus task)")
    parser.add_argument("--rotations", type=int, default=4,
                        help="Number of random rotations for adaptive mode (default 4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output (debug)")

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()

    print("\n" + "=" * 70)
    print(" Mesh Processing Pipeline — SeamGPT Assignment")
    print("=" * 70)
    print(f"Data root : {data_root}")
    print(f"Output dir: {out_root}")
    print(f"Mode      : {'ADAPTIVE (bonus)' if args.adaptive else 'UNIFORM (standard pipeline)'}")
    print(f"Normalizer: {args.normalizer}")
    print(f"Bins      : {args.n_bins}")
    print("=" * 70 + "\n")

    if not data_root.exists():
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------
    try:
        if args.adaptive:
            # BONUS: Adaptive quantization mode
            files = find_mesh_files(data_root)
            print(f"Found {len(files)} meshes for adaptive quantization.\n")
            for f in files:
                try:
                    process_mesh_adaptive(f, out_root,
                                          bins_low=256,
                                          bins_med=1024,
                                          bins_high=4096,
                                          k=16,
                                          n_rot=args.rotations)
                except Exception as e:
                    print(f"[ERROR] {f.name} failed: {e}")
            print("\n Adaptive quantization prototype completed.")
            print(f"Results saved to: {out_root / 'adaptive_bonus'}")

        else:
            # STANDARD: Uniform pipeline
            process_all_meshes(data_root,
                               out_root,
                               normalizer=args.normalizer,
                               n_bins=args.n_bins)
            print("\n Standard pipeline completed successfully.")
            print(f"Results saved to: {out_root}")

    except KeyboardInterrupt:
        print("\n[ABORTED] User interrupted execution.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline crashed: {e}")

    print("\n" + "=" * 70)
    print("Pipeline run finished.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
