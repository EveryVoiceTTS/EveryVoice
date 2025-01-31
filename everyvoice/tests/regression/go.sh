#!/bin/bash

#SBATCH --job-name=EV-r-main
#SBATCH --partition=standard
#SBATCH --account=nrc_ict
#SBATCH --qos=low
#SBATCH --time=10080
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --output=./%x.o%j
#SBATCH --error=./%x.e%j

# Automated application of the instructions in README.md

set -o errexit

TOP_LEVEL_DIR=$(mktemp --directory regress-$(date +'%Y%m%d')-XXX)
cd "$TOP_LEVEL_DIR"

if sbatch -h >& /dev/null; then
    SUBMIT_COMMAND=sbatch
else
    SUBMIT_COMMAND=bash
fi

../prep-datasets.sh
for DIR in regress-*; do
    pushd "$DIR"
    $SUBMIT_COMMAND ../../regression-test.sh
    popd
done

coverage run -p -m everyvoice test

JOB_COUNT=$(find . -maxdepth 1 -name regress-\* | wc -l)
while true; do
    DONE_COUNT=$(find . -maxdepth 2 -name DONE | wc -l)
    if (( DONE_COUNT + 2 >= JOB_COUNT )); then
        break
    fi
    echo "$DONE_COUNT/$JOB_COUNT regression job(s) done. Still waiting."
    date
    sleep $(( 60 * 5 ))
done

echo "$DONE_COUNT regression jobs done. Calculating coverage now, but some jobs may still be running."
../combine-coverage.sh
cat coverage.txt

while true; do
    DONE_COUNT=$(find . -maxdepth 2 -name DONE | wc -l)
    if (( DONE_COUNT >= JOB_COUNT )); then
        break
    fi
    echo "$DONE_COUNT/$JOB_COUNT regression job(s) done. Still waiting."
    date
    sleep $(( 60 * 5 ))
done

echo "All $DONE_COUNT regression jobs done. Calculating final coverage."
rm .coverage
../combine-coverage.sh
cat coverage.txt
