#!/bin/sh

DATE=`date +%Y%m%d`
LOG_FILE=./regression_result_$DATE.log

if [ -f "$LOG_FILE" ]; then
    rm $LOG_FILE
fi

################################
# Check the coding style
################################
PY_FILE_DIR=../
PY_FILES=`find $PY_FILE_DIR -maxdepth 1 -name "*.py"`

echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "Start coding style check..." 2>&1 | tee -a $LOG_FILE
which flake8 >/dev/null 2>&1
FLAKE8=$?
if [ "$FLAKE8" -eq 0 ]; then
    for file in $PY_FILES
    do
        echo "Running flake8 on $file" 2>&1 | tee -a $LOG_FILE
        flake8 $file  2>&1 | tee -a $LOG_FILE       
        echo "" 2>&1 | tee -a $LOG_FILE
    done
else
    echo "Skip coding style check since flake8 is noe present" 2>&1 | tee -a $LOG_FILE
fi
echo "Coding style check done" 2>&1 | tee -a $LOG_FILE
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE


################################
# Graph generating validation
################################
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "Start graph generating validation..." 2>&1 | tee -a $LOG_FILE
python gen_combination_graph.py 2>&1 | tee -a $LOG_FILE
python gen_lrn_graph.py 2>&1 | tee -a $LOG_FILE
python gen_pool_graph.py 2>&1 | tee -a $LOG_FILE
python gen_relu_graph.py 2>&1 | tee -a $LOG_FILE
echo "Graph generating validation done" 2>&1 | tee -a $LOG_FILE
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE


################################
# Unit Test
################################
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "Start unit test..." 2>&1 | tee -a $LOG_FILE
python test_mkl_dummy.py 2>&1 | tee -a $LOG_FILE

# opt
python test_opt.py 2>&1 | tee -a $LOG_FILE

# Pooling
nosetests -s test_pool.py:TestDownsampleFactorMax.test_DownsampleFactorMax 2>&1 | tee -a $LOG_FILE
nosetests -s test_pool.py:TestDownsampleFactorMax.test_DownsampleFactorMaxStride 2>&1 | tee -a $LOG_FILE
nosetests -s test_pool.py:TestDownsampleFactorMax.test_DownsampleFactorMax_grad 2>&1 | tee -a $LOG_FILE

# Relu

# LRN
python test_lrn.py 2>&1 | tee -a $LOG_FILE

# Conv
python test_conv.py 2>&1 | tee -a $LOG_FILE

# BN

# Elemwise
python test_elemwise.py 2>&1 | tee -a $LOG_FILE

# others...
echo "Unit test done" 2>&1 | tee -a $LOG_FILE
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

################################
# Clean up
################################
echo "Clean up..." 2>&1 | tee -a $LOG_FILE
rm *.png
