#!/bin/bash

make
for i in {1..16}
do
    size=$((16*i))
    ./matrix_mult $size $size $size
done