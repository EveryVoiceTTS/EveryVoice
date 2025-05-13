#!/usr/bin/env python

"""
For some reason, the coverage reports in regression tend to include the full paths.
This script filters them to just start with everyvoice/... and realigns the header
and footer accordingly.

Usage: coverage report --include='*/everyvoice/*' | python format-coverage.py
"""

import re
import sys

lines = sys.stdin.readlines()
old_len = len(lines[2])
lines = [re.sub(r".*EveryVoice/everyvoice", "everyvoice", line) for line in lines]
new_len = len(lines[2])
len_diff = old_len - new_len
for i in (0, 1, -2, -1):
    lines[i] = lines[i][:5] + lines[i][5 + len_diff :]
print("".join(lines), end=None)
