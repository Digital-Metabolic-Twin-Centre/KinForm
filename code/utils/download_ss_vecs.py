#!/usr/bin/env python3
"""
Download Protein Embeddings and Compute Secondary-Structure-Weighted Vectors

This script processes per-residue protein embedding files and produces three
additional pooled vectors per sequence, each weighted by the per-residue
secondary-structure confidence scores predicted by S4PRED:

    • ss_C  — coil-weighted mean
    • ss_H  — helix-weighted mean
    • ss_E  — strand-weighted mean

Pipeline per batch:
  1. Download a batch of .npy embedding files from Google Drive via rclone.
  2. Build a temporary FASTA containing the corresponding sequences.
  3. Run S4PRED (predict.sh) on the FASTA to obtain per-residue C / H / E scores.
  4. Compute the three weighted vectors for each sequence.
  5. Save the vectors (float32 .npy) under results/protein_embeddings/<model>/:
         ss_C_vecs/<seq_id>.npy
         ss_H_vecs/<seq_id>.npy
         ss_E_vecs/<seq_id>.npy
  6. Clean up all temporary files before the next batch.

Requires:
  - rclone installed and configured with a "gdrive" remote
  - S4PRED predict.sh at S4PRED_SCRIPT (see configuration below)
  - unique_sequences.fasta at DATA_FASTA (used to look up raw sequences)
"""

import csv
import logging
import shutil
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repository root — this file lives at code/utils/, so go up three levels
ROOT = Path(__file__).resolve().parent.parent.parent

# FASTA file containing all unique sequences (headers use underscores: Sequence_1234)
DATA_FASTA = ROOT / "data" / "unique_sequences.fasta"

# S4PRED predict.sh path
S4PRED_SCRIPT = Path("/home/saleh/s4pred/predict.sh")

# Rclone remote path to the per-residue embedding directories
RCLONE_REMOTE = "gdrive:Recon4IMD/WP2_MetNetwork/T2.5_Kinetics/results/KinForm/embeddings"

# Target root for processed vectors
LOCAL_RESULTS_ROOT = ROOT / "results" / "protein_embeddings"

# Embedding models to process
EMBEDDING_MODELS = [
    "esm2_layer_26",
    "esm2_layer_29",
    "esmc_layer_24",
    "esmc_layer_32",
    "prot_t5_layer_19",
    "prot_t5_last",
]

# Subdirectory names for the three output vector types
SS_TYPES: List[str] = ["ss_C_vecs", "ss_H_vecs", "ss_E_vecs"]

# Default batch size
DEFAULT_BATCH_SIZE = 200

# S4PRED inference device
DEFAULT_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# FASTA utilities
# ---------------------------------------------------------------------------

def load_fasta_index(fasta_path: Path) -> Dict[str, str]:
    """
    Build a mapping  seq_id (with spaces, e.g. "Sequence 1234") -> amino-acid sequence.

    The FASTA file uses underscores in headers (e.g. ``>Sequence_1234``).
    This function normalises headers by replacing underscores with spaces so
    they match the .npy file stems used everywhere else in the project.

    Parameters
    ----------
    fasta_path : Path
        Path to the FASTA file.

    Returns
    -------
    Dict[str, str]
        seq_id -> sequence string (uppercase, no newlines).
    """
    index: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_seq_parts: List[str] = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    index[current_id] = "".join(current_seq_parts)
                # Normalise: "Sequence_1234" -> "Sequence 1234"
                current_id = line[1:].replace("_", " ")
                current_seq_parts = []
            else:
                current_seq_parts.append(line.upper())

    if current_id is not None:
        index[current_id] = "".join(current_seq_parts)

    return index


def write_temp_fasta(
    seq_ids: List[str],
    fasta_index: Dict[str, str],
    out_path: Path,
) -> List[str]:
    """
    Write a temporary FASTA file for the given sequence IDs.

    IDs that are missing from *fasta_index* are silently skipped and
    returned as a separate list.

    Parameters
    ----------
    seq_ids : List[str]
        Ordered list of sequence IDs (space-normalised).
    fasta_index : Dict[str, str]
        Full seq_id -> sequence mapping.
    out_path : Path
        Destination FASTA file path.

    Returns
    -------
    List[str]
        seq_ids that were successfully written (preserves order).
    """
    written: List[str] = []
    missing: List[str] = []

    with open(out_path, "w") as fh:
        for sid in seq_ids:
            seq = fasta_index.get(sid)
            if seq is None:
                missing.append(sid)
                continue
            # Use underscored header so S4PRED seq_id round-trips cleanly
            fh.write(f">{sid.replace(' ', '_')}\n{seq}\n")
            written.append(sid)

    if missing:
        logger.warning(
            f"No FASTA sequence found for {len(missing)} ID(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    return written


# ---------------------------------------------------------------------------
# S4PRED interface
# ---------------------------------------------------------------------------

def run_s4pred(
    fasta_path: Path,
    output_csv: Path,
    device: str = "cpu",
) -> bool:
    """
    Invoke ``predict.sh`` and wait for it to finish.

    Parameters
    ----------
    fasta_path : Path
        Input FASTA file.
    output_csv : Path
        Where S4PRED should write its CSV output.
    device : str
        ``"cpu"`` or ``"gpu"``.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    cmd = [
        "bash",
        str(S4PRED_SCRIPT),
        "--device", device,
        str(fasta_path),
        str(output_csv),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"S4PRED stdout: {result.stdout.strip()}")
        return True

    except subprocess.CalledProcessError as exc:
        logger.error(f"S4PRED failed (exit {exc.returncode}): {exc.stderr.strip()}")
        return False

    except FileNotFoundError:
        logger.error(f"predict.sh not found at: {S4PRED_SCRIPT}")
        return False


def parse_s4pred_csv(csv_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parse S4PRED output CSV and return per-residue probability arrays.

    The CSV has columns: seq_id, C, H, E — where each score column is a
    space-separated string of floats.

    Parameters
    ----------
    csv_path : Path
        Path to the S4PRED output CSV.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Maps normalised seq_id (spaces) -> (C_probs, H_probs, E_probs),
        each a float64 array of shape (L,).
    """
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Normalise ID back to spaces
            sid = row["seq_id"].replace("_", " ")
            c_arr = np.fromstring(row["C"], sep=" ", dtype=np.float64)
            h_arr = np.fromstring(row["H"], sep=" ", dtype=np.float64)
            e_arr = np.fromstring(row["E"], sep=" ", dtype=np.float64)
            results[sid] = (c_arr, h_arr, e_arr)

    return results


# ---------------------------------------------------------------------------
# Vector computation
# ---------------------------------------------------------------------------

def _weighted_mean(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute normalised weighted mean over axis-0.

    Parameters
    ----------
    arr : np.ndarray
        Shape (L, D).
    w : np.ndarray
        Shape (L,).  Will be normalised to sum to 1.

    Returns
    -------
    np.ndarray
        Shape (D,), float32.
    """
    w = w.astype(np.float64)
    total = w.sum()
    if total == 0.0:
        # Fallback to uniform mean if all weights are zero
        return arr.mean(axis=0).astype(np.float32)
    w = w / total
    return (arr * w[:, None]).sum(axis=0).astype(np.float32)


def compute_ss_vectors(
    emb: np.ndarray,
    c_probs: np.ndarray,
    h_probs: np.ndarray,
    e_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute secondary-structure-weighted mean vectors.

    The embedding may be longer or shorter than the SS probability arrays
    (e.g. when truncation was applied, or due to minor length discrepancies
    from tokenisation).  The arrays are aligned at the *shorter* length.

    Parameters
    ----------
    emb : np.ndarray
        Per-residue embeddings, shape (L_emb, D).
    c_probs, h_probs, e_probs : np.ndarray
        Per-residue SS probabilities from S4PRED, each shape (L_ss,).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (ss_C_vec, ss_H_vec, ss_E_vec), each shape (D,), dtype float32.
    """
    L_emb = emb.shape[0]
    L_ss = c_probs.shape[0]
    L = min(L_emb, L_ss)

    if L_emb != L_ss:
        logger.debug(
            f"Length mismatch: embedding has {L_emb} residues, "
            f"S4PRED has {L_ss}. Aligning to {L}."
        )

    emb_trim = emb[:L]
    c_trim = c_probs[:L]
    h_trim = h_probs[:L]
    e_trim = e_probs[:L]

    return (
        _weighted_mean(emb_trim, c_trim),
        _weighted_mean(emb_trim, h_trim),
        _weighted_mean(emb_trim, e_trim),
    )


# ---------------------------------------------------------------------------
# rclone download
# ---------------------------------------------------------------------------

def copy_batch_with_rclone(
    batch_files: List[Path],
    model_name: str,
    temp_dir: Path,
) -> List[Path]:
    """
    Download a batch of .npy files from Google Drive using rclone.

    Parameters
    ----------
    batch_files : List[Path]
        Source file paths on the *mounted* GDrive (used only for filenames).
    model_name : str
        Subdirectory name within the remote, e.g. ``"esm2_layer_26"``.
    temp_dir : Path
        Local temporary directory to receive the files.

    Returns
    -------
    List[Path]
        Successfully downloaded local file paths.
    """
    manifest_file = temp_dir / "manifest.txt"
    with open(manifest_file, "w") as fh:
        for src in batch_files:
            fh.write(f"{model_name}/{src.name}\n")

    cmd = [
        "rclone", "copy",
        RCLONE_REMOTE,
        str(temp_dir),
        "--files-from", str(manifest_file),
        "--transfers", "16",
        "--checkers", "64",
        "--fast-list",
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f"rclone failed: {exc.stderr.strip()}")
        return []
    except Exception as exc:
        logger.error(f"rclone error: {exc}")
        return []
    finally:
        try:
            manifest_file.unlink()
        except OSError:
            pass

    model_temp_dir = temp_dir / model_name
    if not model_temp_dir.exists():
        logger.error(f"rclone did not create expected directory: {model_temp_dir}")
        return []

    return list(model_temp_dir.glob("*.npy"))


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def create_output_dirs(model_name: str) -> Dict[str, Path]:
    """
    Create and return the three output subdirectories for SS vectors.

    Parameters
    ----------
    model_name : str
        Embedding model name.

    Returns
    -------
    Dict[str, Path]
        Maps SS type name (``"ss_C_vecs"``, etc.) to its Path.
    """
    dirs: Dict[str, Path] = {}
    for ss_type in SS_TYPES:
        d = LOCAL_RESULTS_ROOT / model_name / ss_type
        d.mkdir(parents=True, exist_ok=True)
        dirs[ss_type] = d
    return dirs


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    batch_files: List[Path],
    model_name: str,
    temp_dir: Path,
    fasta_index: Dict[str, str],
    out_dirs: Dict[str, Path],
    progress_bar: tqdm,
    device: str,
) -> Tuple[int, int]:
    """
    Download, S4PRED-predict, and save SS-weighted vectors for one batch.

    Parameters
    ----------
    batch_files : List[Path]
        Source .npy paths (used for filenames; actual transfer uses rclone).
    model_name : str
        Embedding model name.
    temp_dir : Path
        Temporary working directory (cleaned up per-batch by caller).
    fasta_index : Dict[str, str]
        Full seq_id -> sequence lookup.
    out_dirs : Dict[str, Path]
        Output directories for the three SS vector types.
    progress_bar : tqdm
        Progress bar to update.
    device : str
        S4PRED device (``"cpu"`` or ``"gpu"``).

    Returns
    -------
    Tuple[int, int]
        (success_count, error_count).
    """
    success = 0
    errors = 0

    # ── 1. Download embeddings ───────────────────────────────────────────────
    local_npy_files = copy_batch_with_rclone(batch_files, model_name, temp_dir)
    if not local_npy_files:
        logger.error(f"rclone returned no files for batch in {model_name}")
        progress_bar.update(len(batch_files))
        return 0, len(batch_files)

    if len(local_npy_files) < len(batch_files):
        errors += len(batch_files) - len(local_npy_files)
        logger.warning(
            f"{errors} file(s) failed to download in this batch"
        )

    # Build seq_id list from downloaded filenames
    seq_ids = [f.stem for f in local_npy_files]

    # ── 2. Write temporary FASTA ─────────────────────────────────────────────
    batch_fasta = temp_dir / "batch.fas"
    written_ids = write_temp_fasta(seq_ids, fasta_index, batch_fasta)

    if not written_ids:
        logger.error("No sequences could be written to batch FASTA — skipping batch")
        progress_bar.update(len(local_npy_files))
        return 0, len(local_npy_files)

    # Build a quick lookup: seq_id -> local npy path for written IDs
    npy_lookup: Dict[str, Path] = {f.stem: f for f in local_npy_files}

    # ── 3. Run S4PRED ────────────────────────────────────────────────────────
    ss_csv = temp_dir / "ss_predictions.csv"
    ok = run_s4pred(batch_fasta, ss_csv, device=device)

    if not ok or not ss_csv.exists():
        logger.error("S4PRED failed for this batch — skipping")
        progress_bar.update(len(written_ids))
        errors += len(written_ids)
        return success, errors

    # ── 4. Parse S4PRED output ───────────────────────────────────────────────
    ss_data = parse_s4pred_csv(ss_csv)

    # ── 5. Compute and save vectors ──────────────────────────────────────────
    for sid in written_ids:
        try:
            npy_path = npy_lookup.get(sid)
            if npy_path is None:
                raise FileNotFoundError(f"No local npy for seq_id: {sid!r}")

            emb = np.load(npy_path, allow_pickle=True)
            if emb.ndim != 2:
                raise ValueError(f"Expected 2-D embedding, got shape {emb.shape}")

            ss_entry = ss_data.get(sid)
            if ss_entry is None:
                raise KeyError(f"S4PRED output missing entry for: {sid!r}")

            c_probs, h_probs, e_probs = ss_entry

            ss_C_vec, ss_H_vec, ss_E_vec = compute_ss_vectors(
                emb, c_probs, h_probs, e_probs
            )

            np.save(out_dirs["ss_C_vecs"] / f"{sid}.npy", ss_C_vec)
            np.save(out_dirs["ss_H_vecs"] / f"{sid}.npy", ss_H_vec)
            np.save(out_dirs["ss_E_vecs"] / f"{sid}.npy", ss_E_vec)

            success += 1

        except Exception as exc:
            logger.error(f"Error processing {sid!r} ({model_name}): {exc}")
            errors += 1

        finally:
            progress_bar.update(1)

    # ── 6. Clean up downloaded embeddings ────────────────────────────────────
    model_temp_dir = temp_dir / model_name
    if model_temp_dir.exists():
        try:
            shutil.rmtree(model_temp_dir)
        except Exception as exc:
            logger.warning(f"Could not clean temp dir {model_temp_dir}: {exc}")

    return success, errors


# ---------------------------------------------------------------------------
# Model-level processing
# ---------------------------------------------------------------------------

def get_source_files(model_name: str) -> List[Path]:
    """
    List all .npy files for a model using rclone lsf (no local mount needed).

    Returns
    -------
    List[Path]
        Dummy Path objects whose .name matches the remote filenames.
        These are used only for filename resolution — actual data is
        transferred by rclone.
    """
    cmd = [
        "rclone", "lsf",
        f"{RCLONE_REMOTE}/{model_name}",
        "--include", "*.npy",
        "--fast-list",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
        # Return as sorted dummy Path objects (parent doesn't matter — only .name is used)
        return sorted(Path(n) for n in names)
    except subprocess.CalledProcessError as exc:
        logger.error(f"rclone lsf failed for {model_name}: {exc.stderr.strip()}")
        return []


def process_model(
    model_name: str,
    fasta_index: Dict[str, str],
    batch_size: int,
    skip_existing: bool,
    device: str,
) -> Dict[str, int]:
    """
    Process all embeddings for a single model.

    Parameters
    ----------
    model_name : str
        Name of the embedding model.
    fasta_index : Dict[str, str]
        seq_id -> sequence mapping.
    batch_size : int
        Number of files per rclone+S4PRED batch.
    skip_existing : bool
        Skip sequences whose three output vectors already exist.
    device : str
        S4PRED device.

    Returns
    -------
    Dict[str, int]
        Keys: ``"success"``, ``"error"``, ``"skipped"``, ``"total"``.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing model: {model_name}")
    logger.info(f"{'='*80}")

    out_dirs = create_output_dirs(model_name)

    # List all remote files
    all_files = get_source_files(model_name)
    total = len(all_files)

    if total == 0:
        logger.warning(f"No .npy files found for model {model_name}")
        return {"success": 0, "error": 0, "skipped": 0, "total": 0}

    logger.info(f"Found {total:,} embedding files")

    # Filter already-processed sequences
    files_to_process: List[Path] = []
    skipped = 0

    if skip_existing:
        for f in all_files:
            sid = f.stem
            all_done = all(
                (out_dirs[ss_type] / f"{sid}.npy").exists()
                for ss_type in SS_TYPES
            )
            if all_done:
                skipped += 1
            else:
                files_to_process.append(f)
        logger.info(f"Skipping {skipped:,} already-processed files")
    else:
        files_to_process = all_files

    if not files_to_process:
        logger.info("All files already processed!")
        return {"success": total - skipped, "error": 0, "skipped": skipped, "total": total}

    logger.info(
        f"Processing {len(files_to_process):,} files "
        f"in batches of {batch_size} (device: {device})"
    )

    total_success = 0
    total_errors = 0
    total_batches = (len(files_to_process) + batch_size - 1) // batch_size

    with tqdm(
        total=len(files_to_process),
        desc=model_name,
        unit="seq",
        ncols=100,
    ) as pbar:
        for batch_idx, i in enumerate(range(0, len(files_to_process), batch_size)):
            batch = files_to_process[i : i + batch_size]
            pbar.set_description(f"{model_name} [batch {batch_idx+1}/{total_batches}]")

            with tempfile.TemporaryDirectory(
                prefix=f"kinform_ss_{model_name}_"
            ) as tmp:
                tmp_path = Path(tmp)
                s, e = process_batch(
                    batch,
                    model_name,
                    tmp_path,
                    fasta_index,
                    out_dirs,
                    pbar,
                    device,
                )
                total_success += s
                total_errors += e

    logger.info(f"\n{model_name} Summary:")
    logger.info(f"  ✓ Success : {total_success:,}")
    logger.info(f"  ✗ Errors  : {total_errors:,}")
    logger.info(f"  ⊘ Skipped : {skipped:,}")
    logger.info(f"  Total     : {total:,}")

    return {
        "success": total_success,
        "error": total_errors,
        "skipped": skipped,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global RCLONE_REMOTE, S4PRED_SCRIPT  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description=(
            "Download per-residue protein embeddings from Google Drive and "
            "compute S4PRED secondary-structure-weighted vectors (C, H, E)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models with default settings
  python download_ss_vecs.py

  # Process specific models on GPU with larger batches
  python download_ss_vecs.py --models esm2_layer_26 esmc_layer_32 --batch-size 500 --device gpu

  # Force reprocessing of all files
  python download_ss_vecs.py --no-skip-existing

  # Use a custom rclone remote path
  python download_ss_vecs.py --rclone-remote "gdrive:MyPath/embeddings"
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=EMBEDDING_MODELS,
        choices=EMBEDDING_MODELS,
        help="Embedding models to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Sequences per rclone+S4PRED batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "gpu"],
        help=f"S4PRED inference device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute vectors even if outputs already exist",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=DATA_FASTA,
        help=f"Path to the FASTA file with all unique sequences (default: {DATA_FASTA})",
    )
    parser.add_argument(
        "--rclone-remote",
        type=str,
        default=None,
        help="Override the rclone remote path",
    )
    parser.add_argument(
        "--s4pred-script",
        type=Path,
        default=None,
        help=f"Path to predict.sh (default: {S4PRED_SCRIPT})",
    )

    args = parser.parse_args()

    # Apply overrides to globals used by helper functions
    if args.rclone_remote:
        RCLONE_REMOTE = args.rclone_remote
    if args.s4pred_script:
        S4PRED_SCRIPT = args.s4pred_script

    # Validate prerequisites
    if not args.fasta.exists():
        logger.error(f"FASTA file not found: {args.fasta}")
        return 1

    if not S4PRED_SCRIPT.exists():
        logger.error(f"predict.sh not found: {S4PRED_SCRIPT}")
        return 1

    # Load FASTA index once for all models
    logger.info(f"Loading FASTA index from {args.fasta} …")
    fasta_index = load_fasta_index(args.fasta)
    logger.info(f"Loaded {len(fasta_index):,} sequences")

    # Ensure output root exists
    LOCAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {len(args.models)} model(s): {', '.join(args.models)}")
    logger.info(f"{'='*80}")

    all_results: Dict[str, Dict[str, int]] = {}

    for model in args.models:
        all_results[model] = process_model(
            model_name=model,
            fasta_index=fasta_index,
            batch_size=args.batch_size,
            skip_existing=not args.no_skip_existing,
            device=args.device,
        )

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")

    g_success = sum(r["success"] for r in all_results.values())
    g_errors  = sum(r["error"]   for r in all_results.values())
    g_skipped = sum(r["skipped"] for r in all_results.values())
    g_total   = sum(r["total"]   for r in all_results.values())

    for model, stats in all_results.items():
        logger.info(
            f"  {model}: ✓{stats['success']:,}  ✗{stats['error']:,}  ⊘{stats['skipped']:,}"
        )

    logger.info("")
    logger.info(f"  Overall: ✓{g_success:,}  ✗{g_errors:,}  ⊘{g_skipped:,}  total={g_total:,}")
    rate = (g_success / g_total * 100) if g_total else 0.0
    logger.info(f"  Success rate: {rate:.2f}%")
    logger.info(f"\n  Output root: {LOCAL_RESULTS_ROOT}")
    logger.info(f"  Vector dirs : <model>/ss_C_vecs/, ss_H_vecs/, ss_E_vecs/")
    logger.info(f"{'='*80}\n")

    return 0 if g_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
