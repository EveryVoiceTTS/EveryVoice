#!/bin/bash

# Usage:
# - make the `everyvoice` command available on the path (e.g., activate your uv venv)
# - run: ./profile-help-ci.sh "${{ github.event.pull_request.head.sha }}"
# - display `import-message.txt` as a PR comment (see .github/workflows/text.yml)

PYTHONPROFILEIMPORTTIME=1 everyvoice -h 2> importtime.txt > /dev/null
CLI_LOAD_TIME="$( (/usr/bin/time --format=%E everyvoice -h > /dev/null) 2>&1 )"

{
    echo "CLI load time: $CLI_LOAD_TIME"
    PR_HEAD="$1"
    [[ $PR_HEAD ]] && echo "Pull Request HEAD: $PR_HEAD"
    echo ""
    echo "Imports that take more than 0.1 s:"
    grep -E 'cumulative|[0-9]{6} ' importtime.txt
} > import-message.txt
cat import-message.txt

echo ""
echo "Full import time log:"
cat importtime.txt

EXIT_CODE=
if [[ "$CLI_LOAD_TIME" > "0:01.00" ]]; then
    {
        echo ""
        echo "ERROR: everyvoice --help is too slow."
        echo "Please run 'PYTHONPROFILEIMPORTTIME=1 everyvoice -h 2> importtime.txt; tuna importtime.txt' and tuck away expensive imports so that the CLI doesn't load them until it uses them."
    } | tee /dev/stderr >> import-message.txt
    EXIT_CODE=1
fi
if grep -E -q "shared_types|pydantic" importtime.txt; then
    {
        echo ""
        echo "ERROR: please be careful not to cause shared_types or pydantic to be imported when the CLI just loads. They are expensive imports."
    } | tee /dev/stderr >> import-message.txt
    EXIT_CODE=1
fi

exit $EXIT_CODE
