# Copyright 2026 Norma Core contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset reader for the norma-core custom parquet format.

Schema per frame:
    episode_start_ns                 uint64   — episode id
    timestamp_ns_since_episode_start uint64
    joints                           list<struct{position_norm: f32, goal_norm: f32, ...}>
    images                           list<struct{jpeg: binary}>
    task                             string

Supports multiple parquet files concatenated together. Every distinct
`(parquet_path, episode_start_ns)` pair gets a unique global episode index so
episode-level train/eval splitting is leak-free even across files.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


QUARTER_TASK_RE = re.compile(r"^put the (\w+) on the [\w-]+ quarter of the pizza$")


def _mmap_table(path: str | Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        reader = pq.ParquetFile(source)
        return reader.read()


def split_episodes(
    n_episodes: int,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Return (train_ep, eval_ep) — deterministic episode-level split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes).tolist()
    n_train = round(n_episodes * train_frac)
    return sorted(perm[:n_train]), sorted(perm[n_train:])


class PickAndPlaceDataset(Dataset):
    """Per-frame samples with a future-action chunk of length `chunk_size`.

    Args:
        parquet_paths: one or more parquet files. Episodes from different files
            get disjoint global episode indices.
        episode_indices: if set, only frames belonging to these global episodes
            are visible. Use with `split_episodes()` for train/eval splits.
        chunk_size: how many future actions each sample carries.
        image_keys: what to name the per-frame JPEGs; index order matches
            the `images` list in the parquet.
        generic_prompt_prob: chance of swapping a quarter prompt ("put the
            cheese on the top-left quarter of the pizza") for the generic
            "cover the pizza with cheese", so one checkpoint learns both.

    Each __getitem__ returns a flat dict keyed for SmolVLAPolicy.forward:
        <img_key>:                      (3, H, W) float in [0, 1]
        observation.state:              (state_dim,) float
        action:                         (chunk_size, action_dim) float
        action_is_pad:                  (chunk_size,) bool — True past episode end
        task:                           str (tokenized later by the policy)
    """

    def __init__(
        self,
        parquet_paths: str | Path | list[str | Path],
        episode_indices: list[int] | None = None,
        chunk_size: int = 50,
        image_keys: tuple[str, ...] = ("observation.images.cam0", "observation.images.cam1"),
        generic_prompt_prob: float = 0.0,
    ):
        if isinstance(parquet_paths, (str, Path)):
            parquet_paths = [parquet_paths]
        self.paths = [Path(p) for p in parquet_paths]
        self.chunk_size = chunk_size
        self.image_keys = tuple(image_keys)
        self.generic_prompt_prob = generic_prompt_prob

        # Load all tables mmapped, compute global episode indices, merge into
        # a single flat view.
        tables = [_mmap_table(p) for p in self.paths]

        state_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        tasks_parts: list[list[str]] = []
        images_cols: list[pa.Array] = []  # per-table images column, kept separate
        row_to_table: list[np.ndarray] = []    # per-row: which table it came from
        row_to_local: list[np.ndarray] = []    # per-row: local index within that table
        row_to_ep_end: list[np.ndarray] = []   # per-row: exclusive upper bound of its episode (local)
        row_to_ep_global: list[np.ndarray] = []

        next_ep_global = 0
        episode_start_ns: dict[int, int] = {}   # global_ep -> episode_start_ns
        for t_idx, table in enumerate(tables):
            n = table.num_rows
            if n == 0:
                continue
            ep_col = table.column("episode_start_ns").to_numpy()
            # Local episode index within this table (0..k-1).
            uniq, inverse = np.unique(ep_col, return_inverse=True)
            global_ep = inverse + next_ep_global
            for local_i, start_ns in enumerate(uniq):
                episode_start_ns[next_ep_global + local_i] = int(start_ns)
            next_ep_global += int(inverse.max()) + 1

            # Episode bounds within this table.
            boundaries = np.flatnonzero(np.diff(ep_col) != 0) + 1
            ep_ends = np.concatenate([boundaries, [n]])
            row_ep_end = np.empty(n, dtype=np.int64)
            start = 0
            for end in ep_ends:
                row_ep_end[start:end] = end
                start = end

            # State / action — tiny, materialize now.
            joints_col = table.column("joints").to_pylist()
            state_parts.append(
                np.asarray(
                    [[j["position_norm"] for j in row] for row in joints_col], dtype=np.float32
                )
            )
            action_parts.append(
                np.asarray(
                    [[j["goal_norm"] for j in row] for row in joints_col], dtype=np.float32
                )
            )
            tasks_parts.append(table.column("task").to_pylist())

            # Keep the images column as-is (lazy decode path keeps JPEGs mmapped).
            images_cols.append(table.column("images"))

            row_to_table.append(np.full(n, t_idx, dtype=np.int32))
            row_to_local.append(np.arange(n, dtype=np.int64))
            row_to_ep_end.append(row_ep_end)
            row_to_ep_global.append(global_ep.astype(np.int64))

        if not state_parts:
            raise ValueError(f"No rows across {[str(p) for p in self.paths]}")

        self._tables = tables
        self._images_cols = images_cols
        self._state_all = np.concatenate(state_parts, axis=0)
        self._action_all = np.concatenate(action_parts, axis=0)
        self._tasks_all: list[str] = [t for chunk in tasks_parts for t in chunk]
        self._row_table = np.concatenate(row_to_table, axis=0)
        self._row_local = np.concatenate(row_to_local, axis=0)
        self._row_ep_end = np.concatenate(row_to_ep_end, axis=0)
        self._row_ep_global = np.concatenate(row_to_ep_global, axis=0)

        self.n_joints = self._state_all.shape[1]
        self.total_episodes = int(self._row_ep_global.max()) + 1
        self.episode_start_ns = episode_start_ns

        # Apply episode filter if given.
        if episode_indices is None:
            self._visible_rows = np.arange(len(self._row_ep_global), dtype=np.int64)
            self._episode_indices = list(range(self.total_episodes))
        else:
            ep_set = np.asarray(sorted(set(episode_indices)), dtype=np.int64)
            mask = np.isin(self._row_ep_global, ep_set)
            self._visible_rows = np.flatnonzero(mask)
            self._episode_indices = ep_set.tolist()

        # Convenience: state/action views over visible rows only (used for stats).
        self._state = self._state_all[self._visible_rows]
        self._action = self._action_all[self._visible_rows]

    def __len__(self) -> int:
        return len(self._visible_rows)

    @property
    def episode_indices(self) -> list[int]:
        return self._episode_indices

    def __getitem__(self, idx: int) -> dict:
        row = int(self._visible_rows[idx])
        ep_end = int(self._row_ep_end[row])
        local_end = ep_end  # ep_end is already local to the source table
        t_idx = int(self._row_table[row])
        local_row = int(self._row_local[row])

        chunk_end_local = local_row + self.chunk_size
        valid = min(chunk_end_local, local_end) - local_row

        # Action chunk — look up from the source table's local range.
        # Each table's local range maps 1:1 into _action_all via the same ordering
        # used at construction time, so we can slice _action_all via the global row.
        action = np.empty((self.chunk_size, self.n_joints), dtype=np.float32)
        action[:valid] = self._action_all[row : row + valid]
        if valid < self.chunk_size:
            action[valid:] = self._action_all[row + valid - 1]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[valid:] = True

        images = {}
        image_row = self._images_cols[t_idx][local_row].as_py()
        for i, key in enumerate(self.image_keys):
            img_bytes = image_row[i]["jpeg"]
            with Image.open(io.BytesIO(img_bytes)) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.uint8)
            images[key] = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0

        task = self._tasks_all[row]
        if self.generic_prompt_prob > 0 and torch.rand(()).item() < self.generic_prompt_prob:
            m = QUARTER_TASK_RE.match(task)
            if m:
                task = f"cover the pizza with {m.group(1)}"

        sample = {
            "observation.state": torch.from_numpy(self._state_all[row].copy()),
            "action": torch.from_numpy(action),
            "action_is_pad": torch.from_numpy(is_pad),
            "task": task,
        }
        sample.update(images)
        return sample


def collate_samples(samples: list[dict], image_keys: tuple[str, ...]) -> dict:
    batch: dict = {}
    for key in ("observation.state", "action", "action_is_pad"):
        batch[key] = torch.stack([s[key] for s in samples], dim=0)
    for key in image_keys:
        batch[key] = torch.stack([s[key] for s in samples], dim=0)
    batch["task"] = [s["task"] for s in samples]
    return batch
