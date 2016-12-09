#!/usr/bin/bash
set -e
ffmpeg -hide_banner -loglevel error -i frame-%04d.png ../output.mp4
echo Created ../output.mp4
