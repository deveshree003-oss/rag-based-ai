#converts the videos to mp3
# converts the videos to mp3
import os
import subprocess

# Define the absolute path to your ffmpeg executable
FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

# Make sure the audios directory exists so it doesn't throw a different error
if not os.path.exists("audios"):
    os.makedirs("audios")

files = os.listdir("videos")
for file in files:
  if file.endswith(".mp4"): # Only process mp4 files
    print(f"Processing: {file}")
    
    # Extracts the tutorial number (-lec01, -lec02, etc.)
    tutorial_number = file.split("_300k")[0].split("7")[1]
    
    # Clean the file name so it doesn't end up looking like 'file.mp4.mp3'
    clean_name = file.rsplit(".", 1)[0]
    output_file = f"audios/{tutorial_number}_{clean_name}.mp3"
    
    # Run using the absolute path to ffmpeg
    subprocess.run([FFMPEG_PATH, "-i", f"videos/{file}", "-vn", "-c:a", "libmp3lame", "-q:a", "2", output_file])
