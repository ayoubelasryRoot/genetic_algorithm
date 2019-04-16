#!/bin/bash

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e)
    EX_PATH="$2"
    shift
    ;;
    *)
        echo "Argument inconnu: ${1}"
        exit
    ;;
esac
shift
done

python3 algo.py $EX_PATH