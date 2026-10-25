#!/usr/bin/env bash
# Lab container entrypoint: bring up the virtual display the screenshot arms are measured on, then
# exec the command.
#
# run_browser.py launches headful Chromium at --window-position=0,0 --window-size=1280,960, and
# browser/capture.py grabs exactly that window rectangle (S1/S2), re-verifying the geometry on every
# capture. The X screen is ONE pixel wider and taller than the window (1281x961, 24-bit) because
# Chromium shrinks a window that would exactly fill the screen to 1279x959 (its X11 anti-maximize
# rule; CDP setWindowBounds cannot override it); the extra root-window row/column is never
# captured. There is no window manager (no decorations, no placement policy), and
# ORIGIN_BLIND_IN_LAB=1 is exported only once the display answers, so capture code can never run
# against a missing display.
set -euo pipefail

DISPLAY_NUM=99
SCREEN=1281x961x24

# a container restarted with `docker start` keeps /tmp: clear a stale server lock first
rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}"
Xvfb ":${DISPLAY_NUM}" -screen 0 "${SCREEN}" -dpi 96 -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!

ready=0
for _ in $(seq 1 100); do
  if xdpyinfo -display ":${DISPLAY_NUM}" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "${XVFB_PID}" 2>/dev/null; then break; fi
  sleep 0.1
done
if [ "${ready}" != 1 ]; then
  echo "entrypoint: Xvfb :${DISPLAY_NUM} did not come up (10 s)" >&2
  cat /tmp/xvfb.log >&2 || true
  exit 1
fi

export DISPLAY=":${DISPLAY_NUM}" ORIGIN_BLIND_IN_LAB=1
exec "$@"
