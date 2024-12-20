#!/bin/bash

# Prepare the datasets and directories for our regression test cases

set -o errexit

# Usage: cat my_file | get_slice lines_to_keep > out
# Use a number of lines or full to get all lines
get_slice() {
    lines=$1
    if [[ $lines == full ]]; then
        cat
    else
        head -"$lines"
    fi
}

EVERYVOICE_REGRESS_ROOT=$(python -c 'import everyvoice; print(everyvoice.__path__[0])')/tests/regression

LJ_SPEECH_DATASET=$HOME/sgile/data/LJSpeech-1.1
LJSLICES="150 600 1600 full"
for slice in $LJSLICES; do
    dir=regress-lj-$slice
    mkdir "$dir"
    ln -s "$LJ_SPEECH_DATASET/wavs" "$dir"/
    get_slice "$slice" < "$LJ_SPEECH_DATASET/metadata.csv" > "$dir"/metadata.csv
    cp "$EVERYVOICE_REGRESS_ROOT"/wizard-resume-lj "$dir"/wizard-resume
    cat <<'==EOF==' > "$dir"/test.txt
This is a test.
I am an anvil.
I have no idea what to write here, but it has to be synthesizable text; so here is something!
Boo!
==EOF==
    echo spec > "$dir"/test2.txt
done

SinhalaTTS=$HOME/sgile/data/SinhalaTTS
dir=regress-si
mkdir $dir
ln -s "$SinhalaTTS/wavs" $dir/
cp "$SinhalaTTS/si_lk.lines.txt" $dir/
cp "$EVERYVOICE_REGRESS_ROOT"/wizard-resume-si "$dir"/wizard-resume
# Source of this sample text: https://en.wikipedia.org/wiki/Sinhala_script CC BY-SA-4.0
#  - the first line means Sinhala script, found at the top of the page
#  - the rest is the first verse from the Pali Dhammapada lower on the same page
cat <<'==EOF==' > "$dir"/test.txt
සිංහල අක්ෂර මාලාව
මනොපුබ්‌බඞ්‌ගමා ධම්‌මා, මනොසෙට්‌ඨා මනොමයා;
මනසා චෙ පදුට්‌ඨෙන, භාසති වා කරොති වා;
තතො නං දුක්‌ඛමන්‌වෙති, චක්‌කංව වහතො පදං.
==EOF==
echo "අක-ෂර" > "$dir"/test2.txt

isiXhosa=$HOME/sgile/data/OpenSLR32-four-South-Afican-languages/xh_za/za/xho
dir=regress-xh
mkdir $dir
ln -s "$isiXhosa/wavs" $dir/
cp "$isiXhosa/line_index.tsv" $dir/
cp "$EVERYVOICE_REGRESS_ROOT"/wizard-resume-xh "$dir"/wizard-resume
# Source of this sample text: individual words copied from
# https://en.wikipedia.org/wiki/Xhosa_language CC BY-SA-4.0
cat <<'==EOF==' > "$dir"/test.txt
ukukrwentshwa
uqeqesho
iimpumlo
==EOF==
echo isiXhosa > "$dir"/test2.txt
