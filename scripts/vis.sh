#!/bin/bash
WORK_DIR=${PWD}
cd ${TBSLAS_DIR}/examples
# make clean
make -j
cd ${WORK_DIR}

NODES=$1
MPI_NUM_PROCESSES=$2
OMP_NUM_THREADS=$3
TOTAL_TIME=$4

JOB_LIST=(zalesak)

for job in "${JOB_LIST[@]}"
do
./.submit_python_job.sh vis-${job}.py $NODES $MPI_NUM_PROCESSES $OMP_NUM_THREADS $TOTAL_TIME
done