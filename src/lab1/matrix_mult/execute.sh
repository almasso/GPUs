#!/bin/bash

rm values.csv
make
title="Size,Ttx1,Ttx2,Tkrnl,Ttx3,BWtx1,BWtx2,BWkrnl,BWtx3"
echo "$title" >> values.csv
for j in {1..2}
do
    for i in {1..64}
    do
        size=$((128*i))
        value=$(./matrix_mult $size $size $size $j)
        echo "$value"
        echo "$value" >> values.csv
    done
done