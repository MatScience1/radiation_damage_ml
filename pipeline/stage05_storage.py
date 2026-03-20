"""
stage05_storage.py — Serialise graph dataset into cloud-optimised binary formats.

Supported backends (set STORAGE_BACKEND env var or CFG.storage.backend):
    lmdb       — Key-value store.  Best for random-access on NFS/network storage
                 and single-node DDP training.
    hdf5       — Chunked + LZF-compressed arrays.  Best for sequential reads
                 and distributed training on Lustre / GPFS.
    webdataset — Sharded tar files.  Best for streaming to GPUs from S3/GCS/Blob.

Output  →  data/
    dataset.lmdb              — LMDB file (lmdb backend)
    dataset.h5                — HDF5 file (hdf5 backend)
    shards/shard_NNNNN.tar    — tar shards (webdataset backend)
    shards/manifest.json      — shard inventory
    graphs/splits.json        — train/val/test index split (all backends)
"""

import io
import json
import logging
import os
import pickle
import sys
import tarfile
from pathlib import Path
from typing import List

import h5py
import lmdb
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CFG, GRAPH_DIR, LMDB_PATH, HDF5_PATH

log = logging.getLogger("stage05_storage")


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
def split_indices(n: int) -> dict:
    train_end = int(n * CFG.loader.train_split)
    val_end   = train_end + int(n * CFG.loader.val_split)
    indices   = np.random.default_rng(42).permutation(n)
    return {
        "train": indices[:train_end].tolist(),
        "val":   indices[train_end:val_end].tolist(),
        "test":  indices[val_end:].tolist(),
    }


# ---------------------------------------------------------------------------
# LMDB backend
# ---------------------------------------------------------------------------
class LMDBWriter:
    """
    Writes torch_geometric Data objects into an LMDB database.
    Key format  : b"graph_{idx:08d}"
    Metadata key: b"metadata" → JSON with counts and feature dims.
    """

    def __init__(self, path: str, map_size: int = CFG.storage.lmdb_map_size):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.env = lmdb.open(
            path, map_size=map_size, subdir=False,
            meminit=False, map_async=True, writemap=True,
        )
        self.count = 0

    def write_graph(self, graph: Data, idx: int) -> None:
        key   = f"graph_{idx:08d}".encode()
        value = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
        with self.env.begin(write=True) as txn:
            txn.put(key, value)
        self.count += 1

    def write_metadata(self, meta: dict) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(b"metadata", json.dumps(meta).encode())

    def close(self):
        self.env.sync()
        self.env.close()


def save_lmdb(graphs: List[Data], path: str) -> None:
    log.info("Writing %d graphs to LMDB at %s", len(graphs), path)
    writer = LMDBWriter(path)
    for idx, graph in enumerate(graphs):
        writer.write_graph(graph, idx)
        if (idx + 1) % 1000 == 0:
            log.info("  Written %d / %d", idx + 1, len(graphs))
    writer.write_metadata({
        "n_graphs":      len(graphs),
        "node_feat_dim": graphs[0].x.shape[1] if graphs else 0,
        "edge_feat_dim": graphs[0].edge_attr.shape[1] if graphs else 0,
        "cutoff_radius": CFG.graph.cutoff_radius,
        "backend":       "lmdb",
    })
    writer.close()
    log.info("LMDB written: %d graphs", len(graphs))


# ---------------------------------------------------------------------------
# HDF5 backend
# ---------------------------------------------------------------------------
def _pad_ragged(arrays: List[np.ndarray], fill_value: float = 0.0) -> np.ndarray:
    """Pad list of variable-length 2D arrays to a common first dimension."""
    max_n = max(a.shape[0] for a in arrays)
    rest  = arrays[0].shape[1:]
    out   = np.full((len(arrays), max_n) + rest, fill_value, dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return out


def save_hdf5(graphs: List[Data], path: str) -> None:
    """
    Store graphs in HDF5 with ragged-array support via padding.

    Layout:
        /graphs/{positions, node_feats, atomic_nums, n_nodes, forces,
                 energy, formation_e}
        /edges/{src, dst, attr, offsets}    — flat + CSR-style offsets
        /metadata/                          — group-level attrs
    """
    log.info("Writing %d graphs to HDF5 at %s", len(graphs), path)
    chunk = min(CFG.storage.hdf5_chunk_size, len(graphs))
    compr = CFG.storage.hdf5_compression

    positions_list   = [g.pos.numpy()        for g in graphs]
    node_feats_list  = [g.x.numpy()          for g in graphs]
    atomic_nums_list = [g.z.numpy().reshape(-1, 1).astype(np.float32)
                        for g in graphs]   # pad_ragged needs 2D
    forces_list      = [g.forces.numpy()     for g in graphs]
    n_nodes          = np.array([g.num_nodes for g in graphs], dtype=np.int32)
    energies         = np.array([g.y.item()  for g in graphs], dtype=np.float32)
    formation_es     = np.array(
        [g.formation_energy.item() for g in graphs], dtype=np.float32
    )

    # Flatten edges into CSR-style storage
    edge_src_flat, edge_dst_flat, edge_attr_flat = [], [], []
    edge_offsets = [0]
    for g in graphs:
        src, dst = g.edge_index[0].numpy(), g.edge_index[1].numpy()
        edge_src_flat.append(src)
        edge_dst_flat.append(dst)
        edge_attr_flat.append(g.edge_attr.numpy())
        edge_offsets.append(edge_offsets[-1] + len(src))

    with h5py.File(path, "w") as f:
        grp = f.create_group("graphs")

        def _ds(name, data, **kw):
            grp.create_dataset(
                name, data=data,
                chunks=(chunk,) + data.shape[1:],
                compression=compr, **kw
            )

        _ds("positions",   _pad_ragged(positions_list))
        _ds("node_feats",  _pad_ragged(node_feats_list))
        # Store atomic nums as int32 (padded as float32, then cast)
        _ds("atomic_nums",
            _pad_ragged(atomic_nums_list).squeeze(-1).astype(np.int32))
        _ds("forces",      _pad_ragged(forces_list))
        _ds("n_nodes",     n_nodes)
        _ds("energy",      energies)
        _ds("formation_e", formation_es)

        egrp = f.create_group("edges")
        egrp.create_dataset("src",
                            data=np.concatenate(edge_src_flat),
                            compression=compr)
        egrp.create_dataset("dst",
                            data=np.concatenate(edge_dst_flat),
                            compression=compr)
        egrp.create_dataset("attr",
                            data=np.concatenate(edge_attr_flat),
                            compression=compr)
        egrp.create_dataset("offsets",
                            data=np.array(edge_offsets, dtype=np.int64))

        mgrp = f.create_group("metadata")
        mgrp.attrs["n_graphs"]      = len(graphs)
        mgrp.attrs["node_feat_dim"] = graphs[0].x.shape[1]
        mgrp.attrs["edge_feat_dim"] = graphs[0].edge_attr.shape[1]
        mgrp.attrs["cutoff_radius"] = CFG.graph.cutoff_radius
        mgrp.attrs["backend"]       = "hdf5"

    log.info("HDF5 written: %d graphs  (%.1f MB)",
             len(graphs), os.path.getsize(path) / 1e6)


# ---------------------------------------------------------------------------
# WebDataset backend
# ---------------------------------------------------------------------------
def save_webdataset(
    graphs: List[Data],
    output_dir: str,
    shard_size: int = CFG.storage.webdataset_shard_size,
) -> None:
    """
    Write sharded .tar files compatible with the WebDataset streaming loader.
    Each sample contains:
        {key}.graph.pkl  — pickled torch_geometric Data
        {key}.meta.json  — human-readable scalar metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    n_shards = (len(graphs) + shard_size - 1) // shard_size
    log.info("Writing %d graphs to %d shards at %s",
             len(graphs), n_shards, output_dir)

    shard_manifest = []
    for shard_idx in range(n_shards):
        start      = shard_idx * shard_size
        end        = min(start + shard_size, len(graphs))
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.tar")

        with tarfile.open(shard_path, "w") as tar:
            for local_idx, graph in enumerate(graphs[start:end]):
                global_idx  = start + local_idx
                key         = f"{global_idx:08d}"
                graph_bytes = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)

                info      = tarfile.TarInfo(name=f"{key}.graph.pkl")
                info.size = len(graph_bytes)
                tar.addfile(info, io.BytesIO(graph_bytes))

                meta_bytes = json.dumps({
                    "global_idx":   global_idx,
                    "defect_id":    graph.defect_id,
                    "defect_type":  graph.defect_type,
                    "n_vacancies":  graph.n_vacancies,
                    "n_nodes":      graph.num_nodes,
                    "n_edges":      graph.num_edges,
                    "energy_eV":    graph.y.item(),
                    "formation_eV": graph.formation_energy.item(),
                    "backend":      graph.backend,
                }).encode()
                info2      = tarfile.TarInfo(name=f"{key}.meta.json")
                info2.size = len(meta_bytes)
                tar.addfile(info2, io.BytesIO(meta_bytes))

        shard_manifest.append({"shard": shard_path, "n_graphs": end - start})
        log.info("  Shard %d/%d  [%d–%d]  (%.1f MB)",
                 shard_idx + 1, n_shards, start, end - 1,
                 os.path.getsize(shard_path) / 1e6)

    with open(os.path.join(output_dir, "manifest.json"), "w") as fh:
        json.dump({"n_total": len(graphs), "shards": shard_manifest}, fh, indent=2)
    log.info("WebDataset manifest written")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    log.info("=== Stage 5: Cloud Storage ===")

    raw_pkl = os.path.join(GRAPH_DIR, "graphs_raw.pkl")
    if not os.path.exists(raw_pkl):
        log.error("graphs_raw.pkl not found — run stage04 first")
        sys.exit(1)

    with open(raw_pkl, "rb") as fh:
        graphs: List[Data] = pickle.load(fh)
    log.info("Loaded %d graphs", len(graphs))

    splits = split_indices(len(graphs))
    with open(os.path.join(GRAPH_DIR, "splits.json"), "w") as fh:
        json.dump(splits, fh)
    log.info("Splits — train: %d  val: %d  test: %d",
             len(splits["train"]), len(splits["val"]), len(splits["test"]))

    backend = os.environ.get("STORAGE_BACKEND", CFG.storage.backend).lower()

    if backend == "lmdb":
        save_lmdb(graphs, LMDB_PATH)
    elif backend == "hdf5":
        save_hdf5(graphs, HDF5_PATH)
    elif backend == "webdataset":
        save_webdataset(graphs, CFG.storage.webdataset_output_dir)
    else:
        log.warning("Unknown backend '%s' — writing all three formats", backend)
        save_lmdb(graphs, LMDB_PATH)
        save_hdf5(graphs, HDF5_PATH)
        save_webdataset(graphs, CFG.storage.webdataset_output_dir)

    log.info("Storage complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    main()
