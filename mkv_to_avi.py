'''

'''


import os
import subprocess

def convert_to_avi(mkv_file, output_dir='.'):
    '''
    Uses Ffmpeg to convert mkv format videos to avi for sphere processing
    Inputs:
        mvk_file
    '''
    try:
        subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
    except FileNotFoundError:
        print("Error: ffmpeg is not installed. Please install it from <https://www.ffmpeg.org/download.html> before using this script.")
        return None
    
    filename, file_ext = os.path.splitext(mkv_file)
    if file_ext.lower() != ".mkv":
        print(f"Error: Input file '{mkv_file}' is not a .mkv file.")
        return None
    output_file = os.path.join(output_dir, filename + ".avi")

    command = ["ffmpeg", "-i", mkv_file, "-c:v", "copy", "-c:a", "copy", output_file]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted '{mkv_file}' to '{output_file}'.")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg command failed with exit code {e.returncode}.")
        return None
    
def process_directory(path):
    '''

    '''
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".mkv"):
            avi_file = convert_to_avi(filename)
         
    return

path = r'D:\Lab data\20240909'
os.chdir(path)
process_directory(path)