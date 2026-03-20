"""
stage06_dataloader.py — GPU-optimised streaming data loaders for training.

Three loader implementations match the three storage backends:
    LMDBGraphDataset       — random-access LMDB; ideal for single-node DDP
    HDF5GraphDataset       — chunked HDF5; efficient sequential reads
    WebDatasetGraphLoader  — streaming tar shards; ideal for cloud object storage

Public API:
    from pipeline.stage06_dataloader import get_loaders
    train_loader, val_loader, test_loader = get_loaders(backend="lmdb")
"""

import io
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import h5py
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CFG, GRAPH_DIR, LMDB_PATH, HDF5_PATH

log = logging.getLogger("stage06_dataloader")


# ---------------------------------------------------------------------------
# LMDB Dataset
# ---------------------------------------------------------------------------
class LMDBGraphDataset(Dataset):
    """
    Random-access PyTorch Dataset backed by LMDB.
    Opens a separate read transaction per DataLoader worker process
    (lazy initialisation inside _get_env).
    """

    def __init__(self, path: str, indices: Optional[List[int]] = None):
        self.path    = path
        self._env    = None
        self.indices = indices

        tmp = lmdb.open(path, subdir=False, readonly=True,
                        lock=False, readahead=False, meminit=False)
        with tmp.begin() as txn:
            raw  = txn.get(b"metadata")
            meta = json.loads(raw) if raw else {}
            n    = meta.get("n_graphs", tmp.stat()["entries"] - 1)
        tmp.close()

        self.n_total = n
        if self.indices is None:
            self.indices = list(range(n))

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.path, subdir=False, readonly=True,
                lock=False, readahead=True, meminit=False,
            )
        return self._env

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Data:
        key = f"graph_{self.indices[pos]:08d}".encode()
        with self._get_env().begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"Key {key} not found in LMDB at {self.path}")
        return pickle.loads(raw)

    def __del__(self):
        if self._env is not None:
            self._env.close()


# ---------------------------------------------------------------------------
# HDF5 Dataset
# ---------------------------------------------------------------------------
class HDF5GraphDataset(Dataset):
    """
    Sequential-access Dataset backed by a chunked HDF5 file.
    Re-constructs torch_geometric Data on-the-fly from dense padded arrays.
    Use num_workers=0 or SWMR mode for multiprocessing.
    """

    def __init__(self, path: str, indices: Optional[List[int]] = None):
        self.path = path
        self._f   = None

        with h5py.File(path, "r") as f:
            n                   = int(f["metadata"].attrs["n_graphs"])
            self.node_feat_dim  = int(f["metadata"].attrs["node_feat_dim"])
            self.edge_feat_dim  = int(f["metadata"].attrs["edge_feat_dim"])
            self.edge_offsets   = f["edges/offsets"][:]   # (N+1,)

        self.n_total = n
        self.indices = indices if indices is not None else list(range(n))

    def _get_file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r", swmr=True)
        return self._f

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Data:
        idx = self.indices[pos]
        f   = self._get_file()
        g   = f["graphs"]

        n_nodes = int(g["n_nodes"][idx])
        x       = torch.from_numpy(g["node_feats"][idx, :n_nodes])
        z       = torch.from_numpy(g["atomic_nums"][idx, :n_nodes].astype(np.int64))
        pos_arr = torch.from_numpy(g["positions"][idx, :n_nodes])
        forces  = torch.from_numpy(g["forces"][idx, :n_nodes])
        energy  = torch.tensor([g["energy"][idx]], dtype=torch.float32)
        form_e  = torch.tensor([g["formation_e"][idx]], dtype=torch.float32)

        e_start = int(self.edge_offsets[idx])
        e_end   = int(self.edge_offsets[idx + 1])
        src     = torch.from_numpy(f["edges/src"][e_start:e_end].astype(np.int64))
        dst     = torch.from_numpy(f["edges/dst"][e_start:e_end].astype(np.int64))
        e_attr  = torch.from_numpy(f["edges/attr"][e_start:e_end])

        return Data(
            x=x, z=z, pos=pos_arr, forces=forces,
            edge_index=torch.stack([src, dst], dim=0),
            edge_attr=e_attr,
            y=energy, formation_energy=form_e,
            num_nodes=n_nodes,
        )

    def __del__(self):
        if self._f is not None:
            self._f.close()


# ---------------------------------------------------------------------------
# WebDataset streaming loader
# ---------------------------------------------------------------------------
def _wds_graph_decoder(sample: dict) -> Data:
    graph_bytes = sample.get("graph.pkl")
    if graph_bytes is None:
        raise ValueError("WebDataset sample missing 'graph.pkl'")
    return pickle.loads(graph_bytes)


def get_webdataset_loader(
    shard_dir: str,
    split: str = "train",
    batch_size: int = CFG.loader.batch_size,
    num_workers: int = CFG.loader.num_workers,
    shuffle_buffer: int = 2000,
):
    """Build a streaming WebDataset DataLoader yielding PyG Batch objects."""
    try:
        import webdataset as wds
    except ImportError:
        raise ImportError("Install webdataset: pip install webdataset")

    with open(os.path.join(shard_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    all_shards = [s["shard"] for s in manifest["shards"]]
    n          = len(all_shards)

    if split == "train":
        shards = all_shards[:int(n * CFG.loader.train_split)]
    elif split == "val":
        s = int(n * CFG.loader.train_split)
        e = s + max(1, int(n * CFG.loader.val_split))
        shards = all_shards[s:e]
    else:
        shards = all_shards[int(n * (CFG.loader.train_split + CFG.loader.val_split)):]

    dataset = (
        wds.WebDataset(shards,
                       shardshuffle=(split == "train"),
                       nodesplitter=wds.split_by_node)
        .shuffle(shuffle_buffer if split == "train" else 0)
        .decode()
        .map(_wds_graph_decoder)
        .batched(batch_size, collation_fn=Batch.from_data_list)
    )

    return wds.WebLoader(
        dataset, batch_size=None, num_workers=num_workers,
        pin_memory=CFG.loader.pin_memory,
        prefetch_factor=CFG.loader.prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Unified loader factory
# ---------------------------------------------------------------------------
def get_loaders(
    backend: Optional[str] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).

    Args:
        backend:     "lmdb" | "hdf5" | "webdataset"  (default: CFG value)
        distributed: Enable DistributedSampler for multi-GPU DDP
        rank:        Local DDP process rank
        world_size:  Total number of DDP processes
    """
    backend = (backend or CFG.storage.backend).lower()
    splits_path = os.path.join(GRAPH_DIR, "splits.json")

    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"splits.json not found at {splits_path} — run stage05 first."
        )

    with open(splits_path) as fh:
        splits = json.load(fh)

    if backend == "webdataset":
        return (
            get_webdataset_loader(CFG.storage.webdataset_output_dir, "train"),
            get_webdataset_loader(CFG.storage.webdataset_output_dir, "val"),
            get_webdataset_loader(CFG.storage.webdataset_output_dir, "test"),
        )

    DatasetClass = {"lmdb": LMDBGraphDataset, "hdf5": HDF5GraphDataset}.get(backend)
    if DatasetClass is None:
        raise ValueError(f"Unsupported backend: {backend}")

    path     = LMDB_PATH if backend == "lmdb" else HDF5_PATH
    train_ds = DatasetClass(path, indices=splits["train"])
    val_ds   = DatasetClass(path, indices=splits["val"])
    test_ds  = DatasetClass(path, indices=splits["test"])

    def _make(ds, is_train: bool) -> PyGDataLoader:
        sampler = None
        if distributed and is_train:
            sampler = DistributedSampler(ds, num_replicas=world_size,
                                         rank=rank, shuffle=True)
        return PyGDataLoader(
            ds,
            batch_size=CFG.loader.batch_size,
            shuffle=(is_train and sampler is None),
            sampler=sampler,
            num_workers=CFG.loader.num_workers,
            pin_memory=CFG.loader.pin_memory,
            prefetch_factor=(
                CFG.loader.prefetch_factor if CFG.loader.num_workers > 0 else None
            ),
            persistent_workers=(CFG.loader.num_workers > 0),
            follow_batch=["x", "z"],
        )

    return _make(train_ds, True), _make(val_ds, False), _make(test_ds, False)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    log.info("Loading loaders (backend=%s)", CFG.storage.backend)
    train_loader, val_loader, test_loader = get_loaders()
    log.info("Train: %d  Val: %d  Test: %d",
             len(train_loader), len(val_loader), len(test_loader))
    batch = next(iter(train_loader))
    log.info("Batch — nodes: %d  edges: %d  y: %s  forces: %s",
             batch.num_nodes, batch.num_edges,
             tuple(batch.y.shape), tuple(batch.forces.shape))
