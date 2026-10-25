#!/usr/bin/env bash
# Fixed picture settings for the two KD-USB arm cameras.
#
# The station never configures the cameras. Left alone they run auto exposure
# with 60 Hz flicker settings, which washed the picture out on Oct 23 and
# makes brightness drift between takes. This pins it to fixed values.
#
# Run it with the station STOPPED: while the station runs it takes the cameras
# away from Linux and there is no /dev/video node to talk to. The settings live
# in the camera until it loses power, so re-run after any replug or reboot,
# then start the station.
#
# Cameras are matched by USB port, never by /dev/videoN (those numbers swap):
#   scene camera = port 1-8.1, wrist camera = port 1-8.2
#
# The resolution the station records at comes from station.yaml (usb-video
# formats), not from the camera, so this script also checks that file.
#
#   ./camera_setup.sh              apply and print the result
#   ./camera_setup.sh --snap DIR   also save one test frame per camera to DIR
set -euo pipefail

STATION_YAML=${STATION_YAML:-$HOME/Downloads/station.yaml}
WIDTH=640
HEIGHT=480
FORMAT="${WIDTH}x${HEIGHT}@30fps,mjpeg"

# Values tuned on the hackathon table at 640x480 (Oct 24, evening light).
# exposure_time_absolute is in 100 us units. The hall lights flicker at 100 Hz
# (50 Hz mains), so keep it a multiple of 100 (10 ms): 110 showed rolling
# bands. Set brightness with gain instead.
# White balance stays automatic: this sensor's manual mode gives a strong
# green cast at every temperature, while auto renders the table neutral.
SCENE_PORT=1-8.1
SCENE_CTRLS="exposure_time_absolute=100,gain=40,saturation=100"
WRIST_PORT=1-8.2
WRIST_CTRLS="exposure_time_absolute=100,gain=40,saturation=100"

# Same on both cameras; written explicitly so a replugged camera can't keep
# a stray value.
COMMON_CTRLS="brightness=0,contrast=40,gamma=100,hue=0,sharpness=3,backlight_compensation=1,power_line_frequency=1"

snap_dir=""
if [[ "${1:-}" == "--snap" ]]; then
    snap_dir=${2:?"--snap needs a directory"}
    mkdir -p "$snap_dir"
fi

if pgrep -x station >/dev/null; then
    echo "The station is running. Stop it first (it holds the cameras)." >&2
    exit 1
fi

if ! grep -q "$FORMAT" "$STATION_YAML"; then
    echo "WARNING: $STATION_YAML does not record at $FORMAT." >&2
    echo "         The data would not match the other 640x480 takes." >&2
fi

# Capture node (index 0) of the KD-USB camera on a given USB port.
find_node() {
    local port=$1 sys
    for sys in /sys/class/video4linux/video*; do
        [[ "$(cat "$sys/name")" == *"KD-USB Cameras"* ]] || continue
        [[ "$(cat "$sys/index")" == 0 ]] || continue
        [[ "$(readlink -f "$sys/device")" == */"$port":* ]] || continue
        echo "/dev/$(basename "$sys")"
        return
    done
}

setup() {
    local label=$1 port=$2 ctrls=$3 dev
    dev=$(find_node "$port")
    if [[ -z "$dev" ]]; then
        echo "$label camera (USB port $port) not found. Unplug and replug it, then run this again." >&2
        return 1
    fi
    v4l2-ctl -d "$dev" --set-fmt-video=width=$WIDTH,height=$HEIGHT,pixelformat=MJPG
    # Manual exposure first: the absolute value is ignored while auto is on.
    v4l2-ctl -d "$dev" --set-ctrl=auto_exposure=1,exposure_dynamic_framerate=0,white_balance_automatic=1
    v4l2-ctl -d "$dev" --set-ctrl="$COMMON_CTRLS,$ctrls"
    echo "$label ($dev, port $port): $(v4l2-ctl -d "$dev" --get-fmt-video | grep -o 'Width/Height *: [0-9/]*' | tr -s ' ')"
    v4l2-ctl -d "$dev" --get-ctrl=auto_exposure,exposure_time_absolute,gain,white_balance_automatic,saturation,power_line_frequency \
        | sed 's/^/    /'
    if [[ -n "$snap_dir" ]]; then
        # Skip the first frames while the sensor settles on the new values.
        v4l2-ctl -d "$dev" --stream-mmap --stream-skip=20 --stream-count=1 --stream-to="$snap_dir/$label.jpg"
        echo "    test frame: $snap_dir/$label.jpg"
    fi
}

setup scene "$SCENE_PORT" "$SCENE_CTRLS"
setup wrist "$WRIST_PORT" "$WRIST_CTRLS"
