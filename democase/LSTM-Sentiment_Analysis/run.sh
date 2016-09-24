#!/bin/sh

[ ! -d data ] && mkdir data
python code/lstm.py
