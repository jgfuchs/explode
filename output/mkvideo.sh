#!/usr/bin/bash
set -e

if [ -z "$1" ]; then
    FPS=25
else
    FPS=$1
fi

ffmpeg -hide_banner -loglevel error -r $FPS -i frame-%04d.png ../output.mp4
echo Created ../output.mp4
