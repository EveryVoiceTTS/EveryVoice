#!/bin/bash

# Prepare the datasets and directories for our regression test cases

set -o errexit

EVERYVOICE_REGRESS_ROOT=$(python -c 'import everyvoice; print(everyvoice.__path__[0])')/tests/regression

SGILE_DATASET_ROOT=${SGILE_DATASET_ROOT:-$HOME/sgile/data}

make_symlink () {
    # Usage: make_symlink LINK_TARGET LINK_NAME
    # is equivalent to ln -s LINK_TARGET LINK_NAME
    if [[ $(uname) =~ MINGW ]]; then
        # Windows via mingw, probably Git Bash: create a "junction" using mklink /j
        cmd //c "mklink /j $(cygpath -w "$2") $(cygpath -w "$1")"
    else
        ln -s "$1" "$2"
    fi
}


DURATIONS="20 60 180 full"

# Source: https://keithito.com/LJ-Speech-Dataset
LJ_SPEECH_DATASET=$SGILE_DATASET_ROOT/LJSpeech-1.1

for duration in $DURATIONS; do
    dir=regress-lj-$duration
    echo "$dir"
    mkdir "$dir"
    make_symlink "$LJ_SPEECH_DATASET/wavs" "$dir"/wavs
    if [[ $duration == 'full' ]]; then
        cat "$LJ_SPEECH_DATASET/metadata.csv" > "$dir"/metadata.csv
    else
        duration_seconds=$(( $duration * 60 )) # Convert the duration from minutes to seconds
        python ../subsample.py "$LJ_SPEECH_DATASET/metadata.csv" "$LJ_SPEECH_DATASET/wavs" -d $duration_seconds -f psv > "$dir"/metadata.csv
    fi
    cp "$EVERYVOICE_REGRESS_ROOT"/wizard-resume-lj "$dir"/wizard-resume
    cat <<'==EOF==' > "$dir"/test.txt
This is a test.
I am an anvil.
I have no idea what to write here, but it has to be synthesizable text; so here is something!
Boo!
==EOF==
    echo spec > "$dir"/test2.txt
done
cp "$EVERYVOICE_REGRESS_ROOT"/run-demo-app-lj-full.sh regress-lj-full/run-demo-app.sh
cp "$EVERYVOICE_REGRESS_ROOT"/test-demo-app-lj-full.py regress-lj-full/test-demo-app.py
cp "$EVERYVOICE_REGRESS_ROOT"/wait-for-demo-app.py "$dir"/wait-for-demo-app.py

# Source: https://openslr.org/30
SinhalaTTS=$SGILE_DATASET_ROOT/SinhalaTTS
for duration in $DURATIONS; do
    dir=regress-si-$duration
    echo "$dir"
    mkdir $dir
    make_symlink "$SinhalaTTS/wavs" "$dir"/wavs
    if [[ $duration == 'full' ]]; then
        cat "$SinhalaTTS/si_lk.lines.txt" > "$dir"/si_lk.lines.txt
    else
        duration_seconds=$(( $duration * 60 )) # Convert the duration from minutes to seconds
        python ../subsample.py "$SinhalaTTS/si_lk.lines.txt" "$SinhalaTTS/wavs" -d "$duration_seconds" -f festival > "$dir"/si_lk.lines.txt
    fi
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
done

# Source: https://openslr.org/32
isiXhosa=$SGILE_DATASET_ROOT/OpenSLR32-four-South-Afican-languages/xh_za/za/xho
for duration in $DURATIONS; do
    dir=regress-xh-$duration
    echo "$dir"
    mkdir $dir
    make_symlink "$isiXhosa/wavs" "$dir"/wavs
    if [[ $duration == 'full' ]]; then
        cat "$isiXhosa/line_index.tsv" > "$dir"/line_index.tsv
    else
        duration_seconds=$(( $duration * 60 )) # Convert the duration from minutes to seconds
        python ../subsample.py "$isiXhosa/line_index.tsv" "$isiXhosa/wavs" -d "$duration_seconds" -f tsv > "$dir"/line_index.tsv
    fi
    cp "$EVERYVOICE_REGRESS_ROOT"/wizard-resume-xh "$dir"/wizard-resume
    # Source of this sample text: individual words copied from
    # https://en.wikipedia.org/wiki/Xhosa_language CC BY-SA-4.0
    cat <<'==EOF==' > "$dir"/test.txt
ukukrwentshwa
uqeqesho
iimpumlo
==EOF==
    echo isiXhosa > "$dir"/test2.txt
done

dir=regress-mix
mkdir $dir
echo "$dir"
cp "$EVERYVOICE_REGRESS_ROOT"/wizard-resume-mix "$dir"/wizard-resume
cat <<'==EOF==' > "$dir"/test.txt
This is a test.
සිංහල අක්ෂර මාලාව
ukukrwentshwa isiXhosa
==EOF==
echo test > "$dir"/test2.txt
cp "$EVERYVOICE_REGRESS_ROOT"/run-demo-app-mix.sh "$dir"/run-demo-app.sh
cp "$EVERYVOICE_REGRESS_ROOT"/test-demo-app-mix.py "$dir"/test-demo-app.py
cp "$EVERYVOICE_REGRESS_ROOT"/wait-for-demo-app.py "$dir"/wait-for-demo-app.py
