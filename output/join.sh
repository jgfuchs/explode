#!/usr/bin/bash
set -e
ls *.mp4 | sort -n | sed -e 's/^/file /' > inputs.txt
ffmpeg -hide_banner -loglevel error -f concat -i inputs.txt -c copy ../joined.mp4
echo Created ../joined.mp4
rm inputs.txt
