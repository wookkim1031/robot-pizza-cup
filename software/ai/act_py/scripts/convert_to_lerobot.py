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

"""Convert norma-core parquet datasets into a local LeRobotDataset.

Same feature layout as smolvla_py so checkpoints line up with the station:
    observation.state        joints[*].position_norm   (n_joints,)
    action                   joints[*].goal_norm       (n_joints,)
    observation.images.camN  images[N].jpeg            (H, W, 3)

Images are stored as frames (use_videos=False): at 224px this is small and
skips the ffmpeg encode.
"""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquets", nargs="+", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path, help="Output dataset directory (replaced if it exists)")
    p.add_argument("--repo-id", default="local/norma", help="Dataset name recorded in metadata; nothing is pushed")
    p.add_argument("--task", default=None, help="Override the per-frame task string for every frame")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--camera-names", default="cam0,cam1",
                   help="feature name per camera in parquet order (cam0 = scene, cam1 = wrist); "
                        "the hackathon handbook wants 'top,front'")
    return p.parse_args()


def decode(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def main() -> None:
    args = parse_args()

    first = pq.read_table(args.parquets[0]).slice(0, 1).to_pylist()[0]
    n_joints = len(first["joints"])
    joint_names = [f"m{i + 1}" for i in range(n_joints)]
    cams = [decode(img["jpeg"]) for img in first["images"]]

    features = {
        "observation.state": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
    }
    names = args.camera_names.split(",")
    for i, img in enumerate(cams):
        features[f"observation.images.{names[i]}"] = {
            "dtype": "image", "shape": img.shape, "names": ["height", "width", "channel"],
        }

    if args.root.exists():
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, root=args.root,
        robot_type="so101", use_videos=False,
    )

    n_episodes = 0
    for path in args.parquets:
        table = pq.read_table(path)
        ep_col = table.column("episode_start_ns").to_numpy()
        for ep in dict.fromkeys(ep_col):
            rows = table.filter(ep_col == ep).to_pylist()
            for row in rows:
                frame = {
                    "observation.state": np.array([j["position_norm"] for j in row["joints"]], dtype=np.float32),
                    "action": np.array([j["goal_norm"] for j in row["joints"]], dtype=np.float32),
                    "task": args.task or row["task"],
                }
                for i, img in enumerate(row["images"]):
                    frame[f"observation.images.{names[i]}"] = decode(img["jpeg"])
                ds.add_frame(frame)
            ds.save_episode()
            n_episodes += 1
            print(f"{path.name}: episode {n_episodes} ({len(rows)} frames)")

    ds.finalize()
    print(f"Wrote {n_episodes} episodes to {args.root}")


if __name__ == "__main__":
    main()
