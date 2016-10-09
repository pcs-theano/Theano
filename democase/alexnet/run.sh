#!/bin/sh

LABEL_FOLDER=`cat config.yaml | grep -E "label_folder" | awk -F ':' '{print $NF}'`
MEAN_FILE=`cat config.yaml | grep -E "mean_file" | awk -F ':' '{print $NF}'`
TRAIN_FOLDER=`cat spec.yaml | grep -E "train_folder" | awk -F ':' '{print $NF}'`
VAL_FOLDER=`cat spec.yaml | grep -E "val_folder" | awk -F ':' '{print $NF}'`

help()
{
    echo "-------------------------------------------------------------------------"
    echo "Warning: Make sure you've specified the correct path to ImageNet for"
    echo "  below varaibles listed in config.yaml and spec.yaml file before running"
    echo "  this workload !!!"
    echo "      [config.yaml] label_folder/mean_file"
    echo "      [spec.yaml]   train_folder/val_folder"
    echo "-------------------------------------------------------------------------"
}

### check if the path or file specified is existed
if [ ! -d $LABEL_FOLDER ]; then
    echo "ERROR: \"$LABEL_FOLDER\" is not existed!"
    help
    exit
fi
if [ ! -f $MEAN_FILE ]; then
    echo "ERROR: \"$MEAN_FILE\" is not existed!"
    help
    exit
fi
if [ ! -d $TRAIN_FOLDER ]; then
    echo "ERROR: \"$TRAIN_FOLDER\" is not existed!"
    help
    exit
fi
if [ ! -d $VAL_FOLDER ]; then
    echo "ERROR: \"$VAL_FOLDER\" is not existed!"
    help
    exit
fi

[ ! -d temp ] && mkdir temp

### run workload
python train.py
