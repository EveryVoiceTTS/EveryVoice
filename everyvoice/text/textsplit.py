# Adapted by the NRC for the purposes of EveryVoice, from
# https://github.com/fakerybakery/txtsplit
# which describes itself as
#   Tortoise TTS's text splitter, repackaged. All credit goes to the author of Tortoise TTS.
# Tortoise TTS: https://github.com/neonbjb/tortoise-tts
# License: Apache License 2.0

import re


def chunk_text(text, desired_length=100, max_length=200):  # noqa: C901
    """
    Split text into chunks of approximately `desired_length`,
    trying to preserve sentence boundaries (!?.), with commas as lower-priority fallbacks.
    """
    # Validate arguments
    assert desired_length < max_length

    # Normalize input
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    chunks = []
    current = ""
    pos = 0
    end = len(text)
    in_quote = False
    strong_splits = []
    weak_splits = []

    def peek(offset):
        idx = pos + offset
        if 0 <= idx < end:
            return text[idx]
        return ""

    def add_split(type_):
        if type_ == "strong":
            strong_splits.append(len(current))
        elif type_ == "weak":
            weak_splits.append(len(current))

    def commit():
        nonlocal current, strong_splits, weak_splits
        if current.strip():
            chunks.append(current.strip())
        current = ""
        strong_splits = []
        weak_splits = []

    while pos < end:
        char = text[pos]
        current += char

        # Handle quote toggling
        if char == '"':
            in_quote = not in_quote

        # Record potential split points
        if not in_quote:
            if char in "!?\n":
                add_split("strong")
            elif char == "." and peek(1) in "\n ":
                add_split("strong")
            elif char in ",;:":
                add_split("weak")

        # If chunk is too long, find best place to split
        if len(current) >= max_length:
            split_at = None
            if strong_splits and len(current) > desired_length / 2:
                split_at = strong_splits[-1]
            elif weak_splits and len(current) > desired_length / 2:
                split_at = weak_splits[-1]

            if split_at:
                chunks.append(current[:split_at].strip())
                current = current[split_at:].lstrip()
            else:
                # fallback: force split at max_length
                chunks.append(current.strip())
                current = ""

            strong_splits = []
            weak_splits = []

        # Early commit on strong boundary
        elif not in_quote and (char in "!?\n" or (char == "." and peek(1) in "\n ")):
            if len(current) >= desired_length:
                commit()

        pos += 1

    if current.strip():
        chunks.append(current.strip())

    # Final filtering
    return [s for s in chunks if s and not re.match(r"^[\s\.,;:!?]*$", s)]
