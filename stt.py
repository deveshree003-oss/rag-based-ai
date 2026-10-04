import os
import sys

# 1. Direct Python to your local FFmpeg binaries 
ffmpeg_bin_dir = r"C:\Program Files\ffmpeg\bin"
if ffmpeg_bin_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.path.pathsep + ffmpeg_bin_dir

# 2. Tell Python to look at your local "whisper" clone folder first
sys.path.insert(0, os.path.abspath("whisper"))
import whisper

# 3. Load the model cleanly on your CPU
print("Loading local source repository Whisper 'small.en' model...")
model = whisper.load_model("small.en", device="cpu")

AUDIO_DIR = "audios"
OUTPUT_DIR = "transcripts"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 4. Gather the lecture files from your folder
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav'))]

if not audio_files:
    print(f"❌ No files found in the '{AUDIO_DIR}' folder!")
    sys.exit()

print(f"Processing {len(audio_files)} files locally...\n")

for idx, file in enumerate(audio_files, 1):
    audio_path = os.path.join(AUDIO_DIR, file)
    clean_name = file.rsplit(".", 1)[0]
    output_txt_path = os.path.join(OUTPUT_DIR, f"{clean_name}.txt")
    
    print(f"[{idx}/{len(audio_files)}] Transcribing: {file}...")
    
    # Run the transcription safely via FP32 CPU execution
    result = model.transcribe(audio=audio_path, language="en", fp16=False)
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
        
    print(f"✓ Saved transcript to: {output_txt_path}\n")

print("🎉 Local folder clone batch transcription complete!")
