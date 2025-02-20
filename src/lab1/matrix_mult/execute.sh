#!/bin/bash

rm values.csv
make
title="Size,Ttx1,Ttx2,Tkrnl,Ttx3,BWtx1,BWtx2,BWkrnl,BWtx3"
echo "$title" >> values.csv
for j in {0..1}
do
    for i in {1..16}
    do
        size=$((16*i))
        value=$(./matrix_mult $size $size $size $j)
        echo "$value"
        echo "$value" >> values.csv
    done
done