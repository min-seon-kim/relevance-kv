#!/bin/bash
# usage: run_queue.sh <jobs_file> <parallel>; each line: "<outname> <eval_ruler args...>"
export HF_HOME=/mnt/beegfs/mkim54/caches/hf
cd /home/mkim54/relevance-kv
R=/mnt/beegfs/mkim54/relevance-kv
run_one() { name=$1; shift; [ -f $name ] && { echo "skip $name"; return; }; $R/venv/bin/python eval_ruler.py "$@" --out $name > ${name%.json}.log 2>&1; echo "done $name $(grep AVG ${name%.json}.log)"; }
export -f run_one; export R
grep -v '^#' $1 | grep -v '^$' | xargs -P $2 -L 1 bash -c 'run_one "$@"' _
