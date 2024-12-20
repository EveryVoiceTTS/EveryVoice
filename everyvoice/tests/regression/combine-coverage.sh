#!/bin/bash

find . -name .coverage\* | coverage combine --keep
coverage report --include='*/everyvoice/*' | sed 's/.*EveryVoice\/everyvoice/everyvoice/' > coverage.txt
coverage html --include='*/everyvoice/*'
coverage xml --include='*/everyvoice/*'
sed -i 's/"[^"]*EveryVoice.everyvoice/"everyvoice/g' coverage.xml
