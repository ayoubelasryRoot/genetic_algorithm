#!/bin/bash

OPTIONS=""
while [[ $# -gt 0 ]]
do
EX_PATH="$1"
shift
done
echo " $EX_PATH"
python3 algo.py $EX_PATH
