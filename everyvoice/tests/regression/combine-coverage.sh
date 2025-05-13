#!/bin/bash

find . -name .coverage\* | coverage combine --keep
SCRIPT_DIR=$(dirname "$0")
coverage report --include='*/everyvoice/*' | python "$SCRIPT_DIR"/format-coverage.py > coverage.txt
coverage html --include='*/everyvoice/*'
coverage xml --include='*/everyvoice/*'
sed -i 's/"[^"]*EveryVoice.everyvoice/"everyvoice/g' coverage.xml
