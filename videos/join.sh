#!/usr/bin/bash
set -e
ffmpeg -hide_banner -loglevel error -f concat -i inputs.txt -c copy ../joined.mp4
echo Created ../joined.mp4
