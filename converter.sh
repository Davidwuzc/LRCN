#!/bin/sh

for folders in "./UCF-101/*"; do
  for files in ${folders}; do
    mkdir -p mp4/${files}
  done
done

for folders in "./UCF-101/*/*"; do
  for files in ${folders}; do
    for video in ${files}; do
      echo ${video}
      ffmpeg -i ${video} -movflags faststart -vcodec libx264 ./mp4/${video}.mp4
    done
  done
done
