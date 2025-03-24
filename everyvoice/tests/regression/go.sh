#!/bin/bash

# Automated application of the instructions in README.md

set -o errexit

TOP_LEVEL_DIR=$(mktemp --directory regress-$(date +'%Y%m%d')-XXX)
cd "$TOP_LEVEL_DIR"

# Cluster-specific calling script should set SUBMIT_COMMAND appropriately
# to request a partition where GPU jobs can run.
# For non-cluster contexts, default to running subjobs sequentially with bash.
if [[ ! $SUBMIT_COMMAND ]]; then
    SUBMIT_COMMAND=bash
fi

# Save some version info so we know what this was run on
(
    echo git describe: "$(git describe)"
    echo git status:
    git status
) | tee version.txt

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

# Calculate some helpful coverage diffs: from origin/main and from the last published version.
(
    REGRESS_DIR=$(pwd)
    cd ../../../..
    diff-cover --compare-branch origin/main "$REGRESS_DIR"/coverage.xml --html-report "$REGRESS_DIR"/coverage-diff-main.html
    LAST_VERSION=$(git describe | sed 's/-.*//')
    diff-cover --compare-branch "$LAST_VERSION" "$REGRESS_DIR"/coverage.xml --html-report "$REGRESS_DIR"/coverage-diff-v0.2.0a1.html
)
