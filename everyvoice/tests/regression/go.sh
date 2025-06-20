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

if ! python -c $'from playwright.sync_api import sync_playwright\nwith sync_playwright() as playwright: playwright.chromium.launch(headless=True)'; then
    echo '
ERROR: playwright headless chromium browser not found.
Please install playwright by installing test dependencies with
    pip install -e .[test]
and then the headless chromium brownser with:
    playwright install --only-shell chromium'
    exit 1
fi

# Save some version info so we know what this was run on
(
    echo git describe: "$(git describe)"
    echo git status:
    git status
) | tee version.txt

../prep-datasets.sh
SUBMIT_OPTIONS=""
for DIR in regress-*; do
    pushd "$DIR"
    if [[ $SUBMIT_COMMAND =~ sbatch ]]; then
        SHORT_DIR=EVr-${DIR#regress-}
        SUBMIT_OPTIONS="--job-name $SHORT_DIR"
    fi
    $SUBMIT_COMMAND $SUBMIT_OPTIONS ../../regression-test.sh
    popd
done

coverage run -p -m everyvoice test 2>&1

JOB_COUNT=$(find . -maxdepth 1 -name regress-\* | wc -l)
while true; do
    DONE_COUNT=$(find . -maxdepth 2 -name DONE | wc -l)
    if (( DONE_COUNT >= JOB_COUNT )); then
        break
    fi
    echo "$DONE_COUNT/$JOB_COUNT regression job(s) done. Still waiting."
    date
    # Update coverage reports every polling period in case we want to follow it.
    rm -f .coverage
    ../combine-coverage.sh > /dev/null
    sleep 60
done

echo "All $DONE_COUNT regression jobs done. Calculating final coverage."
# Calculate coverage
rm -f .coverage
../combine-coverage.sh
cat coverage.txt

# Calculate some helpful coverage diffs: from origin/main and from the last published version.
(
    REGRESS_DIR=$(pwd)
    cd ../../../..
    diff-cover --compare-branch origin/main "$REGRESS_DIR"/coverage.xml --html-report "$REGRESS_DIR"/coverage-diff-main.html
    LAST_VERSION=$(git describe | sed 's/-.*//')
    diff-cover --compare-branch "$LAST_VERSION" "$REGRESS_DIR"/coverage.xml --html-report "$REGRESS_DIR/coverage-diff-$LAST_VERSION.html"
) | tee coverage-diff.txt
