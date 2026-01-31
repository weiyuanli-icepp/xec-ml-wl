import os
import glob
import logging
import warnings
import time
import torch
import numpy as np
import uproot
from torch.utils.data import IterableDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Track if deprecation warning has been shown (avoid spamming)
_RELATIVE_NPHO_WARNING_SHOWN = False
from .utils import iterate_chunks
from .geom_defs import (
    DEFAULT_NPHO_SCALE,
    DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE,
    DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_VALUE,
    DEFAULT_NPHO_THRESHOLD
)

class XECStreamingDataset(IterableDataset):
    """
    Multi-target dictionaries for photon event from ROOT files.
    Supports multi-threaded CPU pre-processing before sending to GPU.
    1. Reads ROOT files in chunks using uproot.
    2. Normalizes Npho and Time branches in parallel using ThreadPoolExecutor.
    3. Yields individual samples for DataLoader batching.

    Args:
        root_files: List of ROOT file paths or single path.
        tree_name: Name of the TTree in ROOT files.
        batch_size: Batch size for yielded tensors.
        step_size: Number of events to load per chunk from ROOT file.
        npho_branch: Branch name for photon counts.
        time_branch: Branch name for timing.
        npho_scale, npho_scale2, time_scale, time_shift, sentinel_value: Normalization params.
        npho_threshold: Minimum npho for valid timing (sensors below have sentinel time).
        num_workers: Number of threads for parallel CPU preprocessing.
        log_invalid_npho: Log warning when invalid npho values detected.
        load_truth_branches: If True, load truth branches for regression. If False, only load inputs.
        shuffle: If True, shuffle samples within each chunk.
    """
    def __init__(self, root_files, tree_name="tree",
                 batch_size=1024, step_size=256000,
                 npho_branch="npho", time_branch="relative_time",
                 npho_scale=DEFAULT_NPHO_SCALE,
                 npho_scale2=DEFAULT_NPHO_SCALE2,
                 time_scale=DEFAULT_TIME_SCALE, time_shift=DEFAULT_TIME_SHIFT,
                 sentinel_value=DEFAULT_SENTINEL_VALUE,
                 npho_threshold=DEFAULT_NPHO_THRESHOLD,
                 num_workers=8,
                 log_invalid_npho=True,
                 load_truth_branches=True,
                 shuffle=False,
                 profile=False):
        super().__init__()

        # I/O profiling
        self.profile = profile
        self._reset_profile_stats()

        # Warn if using deprecated branch name
        global _RELATIVE_NPHO_WARNING_SHOWN
        if npho_branch == "relative_npho" and not _RELATIVE_NPHO_WARNING_SHOWN:
            warnings.warn(
                "Using npho_branch='relative_npho' which is deprecated. "
                "Most data now uses 'npho' branch. Is this intended? "
                "Set npho_branch='npho' if using newer data format.",
                UserWarning,
                stacklevel=2
            )
            _RELATIVE_NPHO_WARNING_SHOWN = True

        self.root_files = root_files if isinstance(root_files, list) else [root_files]
        self.tree_name = tree_name
        self.batch_size = batch_size
        self.step_size = step_size
        self.shuffle = shuffle
        self.load_truth_branches = load_truth_branches

        # Input branches (always loaded)
        self.input_branches = [npho_branch, time_branch]

        # Truth branches (optional, for regression tasks)
        if load_truth_branches:
            self.truth_branches = [
                "energyTruth",   # Energy target
                "timeTruth",     # Timing target
                "uvwTruth",      # Position target (uvw) - first interaction point
                "xyzTruth",      # Position target (xyz) - first interaction point
                "emiAng",        # Emission angle target (theta, phi)
                "emiVec",        # Emission vector (for metric calculation)
                "xyzVTX",        # Vertex position (gamma-ray shooting point)
                "run",           # Run number (for analysis)
                "event",         # Event number (for analysis)
            ]
        else:
            # Minimal branches for MAE/Inpainter (only need run/event for logging)
            self.truth_branches = ["run", "event"]

        self.all_branches = self.input_branches + self.truth_branches

        # Normalization parameters
        # npho_threshold: sensors with raw_npho < threshold have valid npho but invalid time
        self.scales = (npho_scale, npho_scale2, time_scale, time_shift, sentinel_value, npho_threshold)

        # Logging for invalid npho values (unexpected data issues)
        self.log_invalid_npho = log_invalid_npho

        # ThreadPool for CPU-bound normalization
        self.num_threads = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _reset_profile_stats(self):
        """Reset profiling statistics."""
        self._profile_stats = {
            "io_time": 0.0,        # Time waiting for uproot.iterate (file I/O)
            "process_time": 0.0,   # Time in CPU normalization
            "batch_time": 0.0,     # Time creating/yielding batches
            "chunk_count": 0,      # Number of chunks processed
            "event_count": 0,      # Total events processed
            "file_count": 0,       # Number of files (set at start)
        }

    def get_profile_stats(self):
        """Get profiling statistics dictionary."""
        return self._profile_stats.copy()

    def get_profile_report(self):
        """Get a formatted profiling report string."""
        stats = self._profile_stats
        total_time = stats["io_time"] + stats["process_time"] + stats["batch_time"]
        if total_time == 0:
            return "[Dataset Profile] No profiling data collected (profile=False or no data processed)"

        lines = ["[Dataset Profile] I/O breakdown:"]
        lines.append(f"  Files: {stats['file_count']}, Chunks: {stats['chunk_count']}, Events: {stats['event_count']}")

        for name, key in [("I/O (uproot)", "io_time"), ("CPU (normalize)", "process_time"), ("Batch (numpy→torch)", "batch_time")]:
            t = stats[key]
            pct = 100 * t / total_time if total_time > 0 else 0
            lines.append(f"  {name}: {t:.2f}s ({pct:.1f}%)")

        lines.append(f"  TOTAL: {total_time:.2f}s")

        if stats["event_count"] > 0 and total_time > 0:
            throughput = stats["event_count"] / total_time
            lines.append(f"  Throughput: {throughput:.0f} events/s")

        if stats["file_count"] > 0:
            avg_events_per_file = stats["event_count"] / stats["file_count"]
            lines.append(f"  Avg events/file: {avg_events_per_file:.0f}")

        return "\n".join(lines)

    def shutdown(self):
        """Explicitly shutdown the thread pool executor."""
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

    def __del__(self):
        """Cleanup executor on garbage collection."""
        self.shutdown()

    def _process_sub_chunk(self, arr_subset, log_invalid=True):
        """
        Normalize a subset of the chunk in a separate thread.

        Normalization scheme:
        - npho > 9e9 or npho < -npho_scale or isnan: invalid (sentinel for both)
        - npho < npho_threshold: npho valid, time set to sentinel (timing unreliable)
        - otherwise: normal normalization for both

        The log1p transform requires raw_npho/npho_scale > -1, i.e., raw_npho > -npho_scale.
        Sensors with npho < npho_threshold have unreliable timing (uncertainty ~ 1/sqrt(npho)).
        """
        n_sc, n_sc2, t_sc, t_sh, sent, npho_thresh = self.scales

        # Input Normalization
        raw_n = arr_subset[self.input_branches[0]].astype("float32")
        raw_t = arr_subset[self.input_branches[1]].astype("float32")

        # Identify invalid npho values:
        # - raw_npho > 9e9: sentinel in data (missing/dead sensor)
        # - raw_npho < -npho_scale: would break log1p (log of negative)
        # - isnan: corrupted data
        mask_npho_invalid = (raw_n > 9e9) | (raw_n < -n_sc) | np.isnan(raw_n)

        # Log unexpected invalid npho values (these indicate data issues)
        if log_invalid and np.any(mask_npho_invalid):
            run_arr = arr_subset["run"].flatten()
            event_arr = arr_subset["event"].flatten()
            event_idx, sensor_idx = np.where(mask_npho_invalid)
            # Limit logging to first 10 invalid entries to avoid log flooding
            n_invalid = len(event_idx)
            n_to_log = min(n_invalid, 10)
            for i in range(n_to_log):
                ev_i, sens_i = event_idx[i], sensor_idx[i]
                val = raw_n[ev_i, sens_i]
                logger.warning(
                    f"Invalid npho detected: run={run_arr[ev_i]}, event={event_arr[ev_i]}, "
                    f"sensor_idx={sens_i}, value={val:.2e}"
                )
            if n_invalid > n_to_log:
                logger.warning(f"  ... and {n_invalid - n_to_log} more invalid npho values in this chunk")

        # Identify invalid time values:
        # - npho is invalid: can't trust timing either
        # - raw_npho < threshold: timing unreliable (uncertainty ~ 1/sqrt(npho))
        # - raw_time > 9e9: sentinel in data
        # - isnan: corrupted data
        mask_time_invalid = mask_npho_invalid | (raw_n < npho_thresh) | (np.abs(raw_t) > 9e9) | np.isnan(raw_t)

        # Normalize npho: log1p transform
        # For invalid npho, use 0 temporarily to avoid log1p warnings, then set to sentinel
        raw_n_safe = np.where(mask_npho_invalid, 0.0, np.maximum(raw_n, 0.0))  # Also clamp small negatives
        n_norm = np.log1p(raw_n_safe / n_sc) / n_sc2
        n_norm[mask_npho_invalid] = sent

        # Normalize time: linear transform
        t_norm = (raw_t / t_sc) - t_sh
        t_norm[mask_time_invalid] = sent

        x_in = np.stack([n_norm, t_norm], axis=-1)  # (SubBatch, 4760, 2)

        # Build target dict based on which branches are loaded
        if self.load_truth_branches:
            target_dict = {
                # Targets for training
                "energy":  arr_subset["energyTruth"].astype("float32"),
                "timing":  arr_subset["timeTruth"].astype("float32"),
                "uvwFI":   arr_subset["uvwTruth"].astype("float32"),
                "angle":   arr_subset["emiAng"].astype("float32"),
                # For metric calculation (not prediction targets)
                "emiVec":  arr_subset["emiVec"].astype("float32"),
                "xyzTruth": arr_subset["xyzTruth"].astype("float32"),
                "xyzVTX":  arr_subset["xyzVTX"].astype("float32"),
                # For analysis / event identification
                "run":     arr_subset["run"].astype("int64"),
                "event":   arr_subset["event"].astype("int64"),
            }
        else:
            # Minimal target dict for MAE/Inpainter (only run/event for logging)
            target_dict = {
                "run":     arr_subset["run"].astype("int64"),
                "event":   arr_subset["event"].astype("int64"),
            }
        return x_in, target_dict

    def __iter__(self):
        """
        Iterates over chunks and yields pre-batched tensors.
        """
        # Reset profiling stats at start of iteration
        if self.profile:
            self._reset_profile_stats()

        # DistributedDataParallel (DDP)
        worker_info = torch.utils.data.get_worker_info()
        files = self.root_files
        if worker_info is not None:
            # Split files across workers
            per_worker = int(np.ceil(len(files) / float(worker_info.num_workers)))
            files = files[worker_info.id * per_worker : (worker_info.id + 1) * per_worker]
            if len(files) == 0:
                return

        if self.profile:
            self._profile_stats["file_count"] = len(files)

        # Start I/O timer for first chunk
        io_start = time.perf_counter() if self.profile else 0

        for chunk in iterate_chunks(files, self.tree_name, self.all_branches, self.step_size):
            # Record I/O time (time waiting for this chunk)
            if self.profile:
                self._profile_stats["io_time"] += time.perf_counter() - io_start
                self._profile_stats["chunk_count"] += 1
                process_start = time.perf_counter()

            num_events = len(chunk[self.input_branches[0]])

            indices = np.linspace(0, num_events, self.num_threads + 1, dtype=int)
            futures = []

            for i in range(self.num_threads):
                sub = {key: chunk[key][indices[i]:indices[i+1]] for key in self.all_branches}
                futures.append(self.executor.submit(self._process_sub_chunk, sub, self.log_invalid_npho))

            # Collect all processed sub-chunks
            x_parts = []
            # Determine target keys from first result
            first_result = futures[0].result()
            x_parts.append(first_result[0])
            target_keys = list(first_result[1].keys())
            t_parts = {k: [first_result[1][k]] for k in target_keys}

            for f in futures[1:]:
                x_sub, t_sub = f.result()
                x_parts.append(x_sub)
                for k in target_keys:
                    t_parts[k].append(t_sub[k])

            # Concatenate all sub-chunks
            x_all = np.concatenate(x_parts, axis=0)
            t_all = {k: np.concatenate(v, axis=0) for k, v in t_parts.items()}

            # Shuffle within chunk if enabled
            num_samples = len(x_all)
            if self.shuffle:
                perm = np.random.permutation(num_samples)
                x_all = x_all[perm]
                t_all = {k: v[perm] for k, v in t_all.items()}

            if self.profile:
                self._profile_stats["process_time"] += time.perf_counter() - process_start
                self._profile_stats["event_count"] += num_samples

            # Yield batches directly
            # Note: batch_time is measured per-batch to exclude yield suspension time
            # (otherwise training loop time would be incorrectly included)
            for start in range(0, num_samples, self.batch_size):
                if self.profile:
                    batch_start = time.perf_counter()
                end = min(start + self.batch_size, num_samples)
                x_batch = torch.from_numpy(x_all[start:end].copy())
                t_batch = {k: torch.from_numpy(v[start:end].copy()) for k, v in t_all.items()}
                if self.profile:
                    self._profile_stats["batch_time"] += time.perf_counter() - batch_start
                yield x_batch, t_batch

            if self.profile:
                # Start I/O timer for next chunk
                io_start = time.perf_counter()

def expand_path(path):
    """
    Expand path to list of ROOT files.
    - If path is a directory, return all .root files in it.
    - If path is a glob pattern, return matched files.
    - If path is a single file, return it as a list.
    """
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.root")))
        if not files:
            raise ValueError(f"No ROOT files found in directory: {path}")
        return files
    elif os.path.isfile(path):
        return [path]
    else:
        # Treat as glob pattern
        files = sorted(glob.glob(path))
        if not files:
            raise ValueError(f"No ROOT files match pattern: {path}")
        return files


def get_dataloader(file_path, batch_size=1024, num_workers=4, num_threads=4, **kwargs):
    """
    Helper to initialize the multi-threaded data pipeline.

    Args:
        file_path: Directory path, glob pattern, or single file path for ROOT files.
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader workers (for parallel file reading).
        num_threads: Number of threads for CPU preprocessing within each worker.
        **kwargs: Additional arguments passed to XECStreamingDataset
                  (e.g., npho_scale, time_scale, sentinel_value, etc.)

    Returns:
        DataLoader instance.
    """
    root_files = expand_path(file_path)
    print(f"[INFO] DataLoader: Found {len(root_files)} ROOT files from '{file_path}'")

    dataset = XECStreamingDataset(
        root_files,
        batch_size=batch_size,
        num_workers=num_threads,  # Thread pool for CPU preprocessing
        **kwargs
    )

    # Dataset yields pre-batched tensors, so batch_size=None to pass through
    # Using persistent_workers=True keeps workers alive between epochs
    return DataLoader(
        dataset,
        batch_size=None,  # Dataset yields pre-batched tensors
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
