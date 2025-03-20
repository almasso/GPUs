#!/bin/bash

# Este script bash sirve para modificar el .csv con los datos de la ejecución.

make
temp_file="./data/modified_data.csv"

title="ExecutionTimeCPU,ExecutionTimeGPUDefault,ExecutionTimeGPUMultithread,ExecutionTimeGPUOptimized"
echo "$title" > "$temp_file"

while IFS=, read -r valueCPU valueGPU valueGPUMT
do
    if [[ "$valueCPU" == "ExecutionTimeCPU" ]];
    then
        continue
    fi

    new_value=$(./image img0.png g t)

    echo "$new_value"

    echo "$valueCPU,$valueGPU,$valueGPUMT,$new_value" >> "$temp_file"
done < ./data/data.csv

mv "$temp_file" ./data/data.csv