import os
import sys

# 1. Point Python to your FFmpeg path
ffmpeg_bin_dir = r"C:\Program Files\ffmpeg\bin"
if ffmpeg_bin_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.path.pathsep + ffmpeg_bin_dir

from faster_whisper import WhisperModel

# 2. Configure for your 4GB VRAM Graphics Card
# 'int8_float16' compresses the model heavily so it easily fits your GPU memory
MODEL_SIZE = "small.en"
AUDIO_DIR = "audios"
OUTPUT_DIR = "transcripts"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Loading '{MODEL_SIZE}' model onto your NVIDIA RTX GPU...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")

# 3. Gather files
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav'))]

if not audio_files:
    print("❌ No files found in the 'audios' directory!")
    sys.exit()

print(f"🚀 Found {len(audio_files)} files. Starting blazing-fast GPU transcription...\n")

for idx, file in enumerate(audio_files, 1):
    audio_path = os.path.join(AUDIO_DIR, file)
    clean_name = file.rsplit(".", 1)[0]
    output_txt_path = os.path.join(OUTPUT_DIR, f"{clean_name}.txt")
    
    print(f"[{idx}/{len(audio_files)}] Processing: {file}")
    
    # Transcribe with a progress tracker
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    # Open the file and write text segments live as they finish computing
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for segment in segments:
            # Print live text to console so you can watch it working
            print(f"  [{int(segment.start)//60:02d}:{int(segment.start)%60:02d}] {segment.text}")
            f.write(segment.text + " ")
            
    print(f"✓ Saved transcript to: {output_txt_path}\n")

print("🎉 Blazing-fast GPU batch transcription complete!")
