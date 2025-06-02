import sys
from enum import Enum
from os.path import join

import typer
from pydub import AudioSegment

from everyvoice.utils import generic_dict_loader, read_festival

app = typer.Typer()


class MetadataFileFormat(str, Enum):
    psv = "psv"
    tsv = "tsv"
    csv = "csv"
    festival = "festival"


@app.command(help="Standalone test script for subsampling corpora.")
def subsample(
    metadata_path: str,
    wavs_path: str,
    has_header: bool = typer.Option(
        False,
        "--header",
        help="Whether or not the first line of the metadata document is a header row with column names.",
    ),
    duration: int = typer.Option(
        ...,
        "-d",
        "--duration",
        help="Requested minimum duration of subsample in seconds.",
    ),
    format: MetadataFileFormat = typer.Option(
        ..., "-f", "--format", help="Metadata file format."
    ),
    basename: int = typer.Option(
        0,
        "-b",
        "--basename",
        help="Column number of the .wav file basename. Columns are zero-indexed.",
    ),
    speaker: int = typer.Option(
        -1,
        "-s",
        "--speaker",
        help="Column number of the speaker id. Columns are zero-indexed.",
    ),
    speakerid: str = typer.Option(
        None,
        "-i",
        "--speakerid",
        help="Output only records matching the specified speaker.",
    ),
):
    """
    Outputs new metadata with just enough records to have a duration >= specified duration
    """

    # Perform validation
    if format == MetadataFileFormat.festival and speakerid:
        raise typer.BadParameter("Festival formatted files cannot have a speaker id.")

    # Open metadata file
    f = open(metadata_path)
    file = f.readlines()

    # Read metadata file
    if format == MetadataFileFormat.festival:
        metadata = read_festival(metadata_path)
    else:
        separators = {
            MetadataFileFormat.psv: "|",
            MetadataFileFormat.tsv: "\t",
            MetadataFileFormat.csv: ",",
        }
        # Determine location of basename and speaker
        fieldnames = []
        for i in range(max(basename, speaker) + 1):
            if i == speaker:
                fieldnames.append("speaker")
            elif i == basename:
                fieldnames.append("basename")
            else:
                fieldnames.append(str(i))
        metadata = generic_dict_loader(
            metadata_path,
            delimiter=separators[format],
            fieldnames=fieldnames,
            file_has_header_line=has_header,
        )

    # Print header
    if has_header:
        sys.stdout.write(file[0])

    # Tally duration of each audio file
    current_total_duration: float = 0
    for i, dictionary in enumerate(metadata):
        if (
            speakerid and speakerid != dictionary["speaker"]
        ):  # Filter for correct speaker
            continue
        try:
            filename = dictionary["basename"]
            filename = (
                filename if filename.endswith(".wav") else filename + ".wav"
            )  # Append .wav if needed
            filepath = join(wavs_path, filename)
            audio = AudioSegment.from_file(filepath)
        except FileNotFoundError:
            raise typer.BadParameter(
                "A .wav file could not be found. Check whether you need the --header and that --basename contains the correct index."
            )
        current_total_duration += len(audio) / 1000  # Get duration of audio file
        sys.stdout.write(
            file[i + 1] if has_header else file[i]
        )  # Write the current line
        if current_total_duration >= duration:
            break

    # Close metadata file
    f.close()


# Main
if __name__ == "__main__":
    app()
