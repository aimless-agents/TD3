#!/bin/bash
DIR="recordings/$1"
find $DIR/*.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > list.txt; ffmpeg -f concat -i list.txt -c copy $DIR/$1.mp4 ; rm list.txt; rm $DIR/openaigym*; echo "output video stored at $DIR/$1.mp4"