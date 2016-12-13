#!/usr/bin/bash
set -e

if [ -z "$1" ]; then
    FNAME=output
else
    FNAME=$1
fi

ffmpeg -hide_banner -loglevel error -r 25 -i frame-%04d.png ../$FNAME.mp4
echo Created ../$FNAME.mp4
