#!/usr/bin/bash
ffmpeg -hide_banner -loglevel error -i frame-%04d.png output.mp4
mv output.mp4 ..
echo Created ../output.mp4
