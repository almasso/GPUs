#!/bin/bash

make
title="ExecutionTimeCPU,ExecutionTimeGPUDefault"
echo "$title" >> ./data/data.csv
for j in {1..50}
do
    valueCPU=$(./image img0.png c t)
    valueGPU=$(./image img0.png g t)
    echo "$valueCPU"
    echo "$valueGPU"
    echo "$valueCPU,$valueGPU" >> ./data/data.csv
done