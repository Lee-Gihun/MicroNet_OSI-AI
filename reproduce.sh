#!/bin/bash

echo "baseline training and pruning to 50%"
python ./main.py ./Config/reproduce1.json

echo "pruning from 50% to 60%"
python ./main.py ./Config/reproduce2.json

echo "pruning from 60% to 64%"
python ./main.py ./Config/reproduce3.json

echo "early exiting module training"
python ./main.py ./Config/reproduce4.json

echo "pruning early exiting module to 50%"
python ./main.py ./Config/reproduce5.json