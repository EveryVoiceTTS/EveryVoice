#!/bin/bash

find . -name .coverage.\* | xargs coverage combine --keep

COVERAGE_DIR=$(pwd -P)
EVERYVOICE_ROOT=$(python -c 'import everyvoice; print(everyvoice.__path__[0])')/..
COVERAGE_OPTIONS=(--data-file="$COVERAGE_DIR/.coverage" --include='*/everyvoice/*')

cd "$EVERYVOICE_ROOT" || exit 1
coverage report "${COVERAGE_OPTIONS[@]}" > "$COVERAGE_DIR/coverage.txt"
coverage html "${COVERAGE_OPTIONS[@]}" --directory="$COVERAGE_DIR/htmlcov"
coverage xml "${COVERAGE_OPTIONS[@]}" -o "$COVERAGE_DIR/coverage.xml"
