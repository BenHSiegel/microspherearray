#!/usr/bin/env python3
"""
Combine two MKV videos into one using ffmpeg.
Usage: python combine_mkv.py input1.mkv input2.mkv output.mkv
"""

import subprocess
import sys
import os
import tempfile


def combine_mkv(input1: str, input2: str, output: str) -> None:
    # Validate input files
    for f in [input1, input2]:
        if not os.path.exists(f):
            print(f"Error: File '{f}' not found.")
            sys.exit(1)

    print(f"Combining:\n  {input1}\n  {input2}\nInto: {output}\n")

    # Create a temporary concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(f"file '{os.path.abspath(input1)}'\n")
        tmp.write(f"file '{os.path.abspath(input2)}'\n")
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", tmp_path,
            "-c", "copy",       # Stream copy — no re-encoding, very fast
            "-y",               # Overwrite output if it exists
            output,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("ffmpeg error:")
            print(result.stderr)
            sys.exit(1)

        print(f"Done! Combined video saved to: {output}")

    finally:
        os.unlink(tmp_path)


vid1 = r"C:\Users\bensi\Downloads\organize_beginning.mkv"

vid2 = r"C:\Users\bensi\Downloads\organize_ending.mkv"

savename = r"C:\Users\bensi\Downloads\organize_combined.mkv"
combine_mkv(vid1, vid2, savename)