#!/usr/bin/env python3
"""Convert OGBench manipulation NPZ datasets into stable-worldmodel HDF5.

The LeWM training path consumes HDF5 files with flat per-step datasets plus
episode metadata (`ep_len` and `ep_offset`). OGBench distributes manipulation
datasets as `.npz`, with `terminals` marking episode boundaries. This converter
keeps the visual observations as `pixels`, maps `actions` to `action`, and adds
the state/goal fields needed by the existing OGB cube eval config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin
except Exception:  # pragma: no cover - compression is optional at write time.
    hdf5plugin = None


def _dataset_kwargs(name: str, array: np.ndarray, compression: str | None) -> dict:
    kwargs: dict = {
        "chunks": (min(1000, len(array)), *array.shape[1:]),
        "maxshape": (None, *array.shape[1:]),
    }
    if name == "pixels":
        kwargs["chunks"] = (min(100, len(array)), *array.shape[1:])

    if compression == "zstd":
        if hdf5plugin is None:
            raise RuntimeError("hdf5plugin is required for zstd compression")
        kwargs.update(hdf5plugin.Zstd(clevel=3))
    elif compression == "gzip":
        kwargs.update({"compression": "gzip", "compression_opts": 4})
    elif compression not in {None, "none"}:
        raise ValueError(f"Unsupported compression={compression!r}")

    return kwargs


def _episode_metadata(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    terminals = np.asarray(terminals, dtype=bool)
    end_indices = np.nonzero(terminals)[0]
    if len(end_indices) == 0 or end_indices[-1] != len(terminals) - 1:
        end_indices = np.concatenate([end_indices, [len(terminals) - 1]])

    lengths = np.diff(np.concatenate([[-1], end_indices])).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    ep_idx = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
    step_idx = np.concatenate(
        [np.arange(length, dtype=np.int64) for length in lengths]
    )
    return lengths, offsets, ep_idx, step_idx


def _cube_goal_fields(qpos: np.ndarray, num_cubes: int) -> dict[str, np.ndarray]:
    fields = {}
    base = qpos.shape[1] - 7 * num_cubes
    if base < 0:
        raise ValueError(
            f"Cannot infer {num_cubes} cube poses from qpos shape {qpos.shape}"
        )
    for cube_id in range(num_cubes):
        pose = qpos[:, base + 7 * cube_id : base + 7 * (cube_id + 1)]
        fields[f"privileged_block_{cube_id}_pos"] = pose[:, :3].astype(np.float32)
        fields[f"privileged_block_{cube_id}_quat"] = pose[:, 3:7].astype(np.float32)
        fields[f"goal_privileged_block_{cube_id}_pos"] = pose[:, :3].astype(np.float32)
        fields[f"goal_privileged_block_{cube_id}_quat"] = pose[:, 3:7].astype(np.float32)
    return fields


def convert(
    npz_path: Path,
    h5_path: Path,
    *,
    num_cubes: int,
    compression: str | None,
    limit: int | None,
) -> None:
    raw = np.load(npz_path, allow_pickle=False)
    observations = raw["observations"]
    actions = raw["actions"]
    terminals = raw["terminals"]
    qpos = raw["qpos"]
    qvel = raw["qvel"]

    if limit is not None:
        observations = observations[:limit]
        actions = actions[:limit]
        terminals = terminals[:limit].copy()
        qpos = qpos[:limit]
        qvel = qvel[:limit]
        terminals[-1] = True

    ep_len, ep_offset, ep_idx, step_idx = _episode_metadata(terminals)
    # OGBCube's step-time observation grows with the number of cubes:
    # single=28, double=37, triple=46. JEPA consumes pixels/action and uses
    # qpos/qvel for env reset, but World.evaluate mirrors dataset columns into
    # the env info buffer before stepping. Keep this column shape-compatible
    # with the env to avoid EnvPool write-shape mismatches.
    observation_dim = 19 + 9 * num_cubes
    observation = np.zeros((len(actions), observation_dim), dtype=np.float32)

    columns: dict[str, np.ndarray] = {
        "pixels": observations.astype(np.uint8),
        "action": actions.astype(np.float32),
        "observation": observation,
        "qpos": qpos.astype(np.float32),
        "qvel": qvel.astype(np.float32),
        "ep_idx": ep_idx,
        "step_idx": step_idx,
        "id": np.arange(len(actions), dtype=np.int64),
        "terminated": terminals.astype(bool),
        "success": terminals.astype(bool),
    }
    columns.update(_cube_goal_fields(qpos, num_cubes))

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w", libver="latest") as f:
        for name, array in columns.items():
            f.create_dataset(
                name,
                data=array,
                **_dataset_kwargs(name, array, compression),
            )
        f.create_dataset("ep_len", data=ep_len, maxshape=(None,), chunks=True)
        f.create_dataset("ep_offset", data=ep_offset, maxshape=(None,), chunks=True)

    print(f"Wrote {h5_path}")
    print(f"steps={len(actions)} episodes={len(ep_len)}")
    print("columns=" + ", ".join(sorted(columns)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--num-cubes", type=int, default=2)
    parser.add_argument(
        "--compression",
        choices=["zstd", "gzip", "none"],
        default="zstd",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    convert(
        args.npz_path,
        args.h5_path,
        num_cubes=args.num_cubes,
        compression=None if args.compression == "none" else args.compression,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
