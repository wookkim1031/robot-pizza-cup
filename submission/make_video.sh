#!/usr/bin/env bash
# Turn a phone recording into the demo video: under 60 s, H.264 + AAC, 1080p max, named [TEAM-N]-[TEAM-NAME].mp4.
#
#   TEAM_N=<n> TEAM_NAME=<name> ./make_video.sh input.mp4 [START] [DURATION]
#     START     where to cut from, e.g. 00:00:12 (default 0)
#     DURATION  seconds to keep (default 58, capped at 59)
#   SPEED=2 ...  plays the clip 2x faster (audio dropped), handy for squeezing a full Ch2 run under a minute
set -euo pipefail
: "${TEAM_N:?set TEAM_N}" "${TEAM_NAME:?set TEAM_NAME}"
IN=${1:?input video}; START=${2:-0}; DUR=${3:-58}; SPEED=${SPEED:-1}
(( $(printf '%.0f' "$DUR") <= 59 )) || DUR=59
OUT="$(dirname "$0")/$TEAM_N-$TEAM_NAME.mp4"

# DURATION is output length, so read DUR*SPEED seconds of input.
IN_DUR=$(awk "BEGIN{print $DUR*$SPEED}")
VF="scale='min(1920,iw)':-2,setpts=PTS/$SPEED"
if [[ $SPEED == 1 ]]; then AUDIO=(-c:a aac -b:a 128k); else AUDIO=(-an); fi

ffmpeg -y -ss "$START" -t "$IN_DUR" -i "$IN" -vf "$VF" -c:v libx264 -preset medium -crf 23 \
  -pix_fmt yuv420p "${AUDIO[@]}" -movflags +faststart "$OUT"
ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUT"
echo "wrote $OUT"
