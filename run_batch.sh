#!/bin/bash
# usage: run_batch.sh script outdir seeds...
S=$1; O=$2; shift 2
for s in "$@"; do python3 $S $s $O > $O/log_$(basename $S .py)_$s.txt 2>&1; done
