#!/bin/bash

INPUT_IMAGE=$1

# generate all dihedral transformations with imagemagick

function output_image() {
    magick convert $INPUT_IMAGE $@ -resize 512x512! -colorspace gray -depth 8 gray:- 
}

output_image
output_image -flop
output_image -rotate 180
output_image -flip
output_image -rotate 90 -flop
output_image -rotate 270
output_image -flop -rotate 90
output_image -rotate 90