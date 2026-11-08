#!/usr/bin/env bash
# Upload the Robot Pizza Cup submission (8 datasets + 1 model) to the Hugging Face org.
#
#   HF_ORG=<org> TEAM_N=<n> TEAM_NAME=<name> ./upload_to_hub.sh            # dry run: prints repo ids
#   HF_ORG=<org> TEAM_N=<n> TEAM_NAME=<name> ./upload_to_hub.sh --go       # creates repos and uploads
#   HF_ORG=RoboIT1234 ./upload_to_hub.sh --suffix --go   # no team id: <dataset>-lerobot and <model>
#
# Needs `hf auth login` with a WRITE token for an account that is a member of HF_ORG.
# Repos are created private; add --public to make them public.
set -euo pipefail

SUB=/home/johan/datasets/submission
HF=/home/johan/norma-core/software/ai/act_py/.venv/bin/hf
MODEL_DIR=$SUB/model/smolvla-grasp5-2-2026-10-25-step003000
MODEL_NAME=smolvla-grasp5-2-2026-10-25
DATASETS=(
  cheese-r1r10-2026-10-24
  pepperoni-r1r10-2026-10-24
  pepperoni-on-cheese-r1r5-2026-10-24
  pepperoni-grasp-r1r3-2026-10-25
  pepperoni-grasp-on-cheese-r1r2-2026-10-25
  cheese-grasp-r1r3-2026-10-25
  cheese-grasp-new-2026-10-25
  pepperoni-grasp-new-2026-10-25
)

GO=0; PRIVATE=--private; SUFFIX=0
for a in "$@"; do
  case $a in
    --go) GO=1 ;;
    --public) PRIVATE=--public ;;
    --suffix) SUFFIX=1 ;;
    *) echo "unknown arg $a" >&2; exit 2 ;;
  esac
done
: "${HF_ORG:?set HF_ORG}"
if [[ $SUFFIX == 1 ]]; then
  ds_repo() { echo "$HF_ORG/$1-lerobot"; }
  MODEL_REPO="$HF_ORG/$MODEL_NAME"
else
  : "${TEAM_N:?set TEAM_N}" "${TEAM_NAME:?set TEAM_NAME}"
  PREFIX="$TEAM_N-$TEAM_NAME"
  ds_repo() { echo "$HF_ORG/$PREFIX-$1"; }
  MODEL_REPO="$HF_ORG/$PREFIX-$MODEL_NAME"
fi

for d in "${DATASETS[@]}"; do
  [[ -f $SUB/$d/meta/info.json && -f $SUB/$d/README.md ]] || { echo "missing $SUB/$d (info.json or README.md)" >&2; exit 1; }
done
[[ -f $MODEL_DIR/model.safetensors && -f $MODEL_DIR/README.md ]] || { echo "missing model files in $MODEL_DIR" >&2; exit 1; }

# The cards carry [TEAM-N]-[TEAM-NAME] placeholders; fill them in a temp copy of each README at upload time.
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fill() {
  if [[ $SUFFIX == 1 ]]; then
    sed -E -e 's/\[TEAM-N\]-\[TEAM-NAME\]-(smolvla[a-z0-9-]*)/\1/g' \
           -e 's/\[TEAM-N\]-\[TEAM-NAME\]-([a-z0-9-]+-2026-10-2[45])/\1-lerobot/g' "$1"
  else
    sed -e "s/\[TEAM-N\]-\[TEAM-NAME\]/$PREFIX/g" "$1"
  fi > "$TMP/README.md"
  echo "$TMP/README.md"
}

if [[ $GO == 1 ]]; then
  $HF auth whoami
fi

for d in "${DATASETS[@]}"; do
  repo=$(ds_repo "$d")
  echo "dataset  $repo  <- $SUB/$d"
  [[ $GO == 1 ]] || continue
  $HF repos create "$repo" --repo-type dataset $PRIVATE --exist-ok
  $HF upload "$repo" "$SUB/$d" . --repo-type dataset --exclude README.md --delete '*' --commit-message "Upload $d"
  $HF upload "$repo" "$(fill "$SUB/$d/README.md")" README.md --repo-type dataset --commit-message "Dataset card"
done

repo=$MODEL_REPO
echo "model    $repo  <- $MODEL_DIR"
if [[ $GO == 1 ]]; then
  $HF repos create "$repo" --repo-type model $PRIVATE --exist-ok
  $HF upload "$repo" "$MODEL_DIR" . --exclude README.md --delete '*' --commit-message "Upload $MODEL_NAME step 3000"
  $HF upload "$repo" "$(fill "$MODEL_DIR/README.md")" README.md --commit-message "Model card"
fi
