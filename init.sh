#!/bin/sh

if [ -e "ml-1m.zip" ]; then
    echo 'File already exists' >&2
else
    curl http://files.grouplens.org/datasets/movielens/ml-1m.zip -o ml-1m.zip
    unzip ml-1m -d ml-1m
fi
python3 train_test_split.py