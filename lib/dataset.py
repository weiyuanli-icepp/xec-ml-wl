import os
import glob
import torch
import numpy as np
import uproot
from torch.utils.data import IterableDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from .utils import iterate_chunks
from .geom_defs import (
    DEFAULT_NPHO_SCALE, 
    DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, 
    DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_VALUE
)

class XECStreamingDataset(IterableDataset):
    """
    Multi-target dictionaries for photon event from ROOT files.
    Supports multi-threaded CPU pre-processing before sending to GPU.
    1. Reads ROOT files in chunks using uproot.
    2. Normalizes Npho and Time branches in parallel using ThreadPoolExecutor.
    3. Yields individual samples for DataLoader batching.
    """
    def __init__(self, root_files, tree_name="tree", 
                 batch_size=1024, step_size=256000,
                 npho_branch="relative_npho", time_branch="relative_time",
                 npho_scale=DEFAULT_NPHO_SCALE, 
                 npho_scale2=DEFAULT_NPHO_SCALE2,
                 time_scale=DEFAULT_TIME_SCALE, time_shift=DEFAULT_TIME_SHIFT, 
                 sentinel_value=DEFAULT_SENTINEL_VALUE,
                 num_workers=8):
        super().__init__()
        self.root_files = root_files if isinstance(root_files, list) else [root_files]
        self.tree_name = tree_name
        self.batch_size = batch_size
        self.step_size = step_size
        
        # Input and Truth branches
        self.input_branches = [npho_branch, time_branch]
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
        self.all_branches = self.input_branches + self.truth_branches
        
        # Scales
        # self.npho_scale = npho_scale
        # self.npho_scale2 = npho_scale2
        # self.time_scale = time_scale
        # self.time_shift = time_shift
        # self.sentinel_value = sentinel_value
        self.scales = (npho_scale, npho_scale2, time_scale, time_shift, sentinel_value)
        
        # ThreadPool for CPU-bound normalization
        self.num_threads = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def shutdown(self):
        """Explicitly shutdown the thread pool executor."""
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

    def __del__(self):
        """Cleanup executor on garbage collection."""
        self.shutdown()

    def _process_sub_chunk(self, arr_subset):
        """
        Normalize a subset of the chunk in a separate thread.
        """
        n_sc, n_sc2, t_sc, t_sh, sent = self.scales
        
        # Input Normalization
        raw_n = arr_subset[self.input_branches[0]].astype("float32")
        raw_t = arr_subset[self.input_branches[1]].astype("float32")
        
        # Identify bad values
        mask_npho_bad = (raw_n <= 0.0) | (raw_n > 9e9) | np.isnan(raw_n)
        mask_time_bad = mask_npho_bad | (np.abs(raw_t) > 9e9) | np.isnan(raw_t)
        
        # Normalize
        n_norm = np.log1p(raw_n / n_sc) / n_sc2
        t_norm = (raw_t / t_sc) - t_sh
        n_norm[mask_npho_bad] = 0.0
        t_norm[mask_time_bad] = sent
        
        x_in = np.stack([n_norm, t_norm], axis=-1) # (SubBatch, 4760, 2)
        
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
        return x_in, target_dict

    def __iter__(self):
        """
        Iterates over chunks and yields batches.
        """
        # DistributedDataParallel (DDP)
        worker_info = torch.utils.data.get_worker_info()
        files = self.root_files
        if worker_info is not None:
            # Split files across workers
            per_worker = int(np.ceil(len(files) / float(worker_info.num_workers)))
            files = files[worker_info.id * per_worker : (worker_info.id + 1) * per_worker]

        for chunk in iterate_chunks(files, self.tree_name, self.all_branches, self.step_size):
            num_events = len(chunk[self.input_branches[0]])
            
            indices = np.linspace(0, num_events, self.num_threads + 1, dtype=int)
            futures = []
            
            for i in range(self.num_threads):
                sub = {key: chunk[key][indices[i]:indices[i+1]] for key in self.all_branches}
                futures.append(self.executor.submit(self._process_sub_chunk, sub))
                
            for f in futures:
                x_sub, t_sub = f.result()
                for i in range(len(x_sub)):
                    yield torch.from_numpy(x_sub[i]), {k: torch.as_tensor(v[i]) for k, v in t_sub.items()}

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

    # Using persistent_workers=True keeps workers alive between epochs
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
