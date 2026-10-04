import os
import sys
import json

# 1. Direct path to your local FFmpeg binaries
ffmpeg_bin_dir = r"C:\Program Files\ffmpeg\bin"
if ffmpeg_bin_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.path.pathsep + ffmpeg_bin_dir

# 2. Point Python to your local whisper folder
sys.path.insert(0, os.path.abspath("whisper"))
import whisper

print("Loading local Whisper 'turbo' model on CPU...")
model = whisper.load_model("turbo", device="cpu")

AUDIO_DIR = "audios"
JSON_OUTPUT_DIR = "jsons"

if not os.path.exists(JSON_OUTPUT_DIR):
    os.makedirs(JSON_OUTPUT_DIR)

# Gather all .mp3 files from the directory
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav'))]

if not audio_files:
    print(f" Error: No audio files found in the '{AUDIO_DIR}' folder!")
    sys.exit()

print(f" Batch processing started for all {len(audio_files)} files...\n")

for idx, filename in enumerate(audio_files, 1):
    audio_path = os.path.join(AUDIO_DIR, filename)
    
    # Generate clean output JSON name path
    output_json_name = f"{filename.rsplit('.', 1)[0]}.json"
    output_json_path = os.path.join(JSON_OUTPUT_DIR, output_json_name)
    
    # Skip processing if this JSON file already exists to save time!
    if os.path.exists(output_json_path):
        print(f"[{idx}/{len(audio_files)}]  Skipping (already exists): {output_json_name}")
        continue

    print(f"[{idx}/{len(audio_files)}] Extracting speech segments for: {filename}...")
    result = model.transcribe(audio=audio_path, language="en", fp16=False)
    
    # --- Advanced Filename Parsing Cleanup ---
    # Target: "-lec01b_ocw-6.189-iap07-lec01b_300k.mp3"
    clean_name = filename.rsplit(".", 1)[0]
    
    try:
        # Extract the lecture code (e.g., '01b')
        if "lec" in clean_name:
            after_lec = clean_name.split("lec")[1]
            tutorial_number = after_lec.split("_")[0]
        else:
            tutorial_number = str(idx)
            
        # Create a beautiful clean title
        tutorial_title = f"MIT 6.189 Lecture {tutorial_number}"
    except Exception:
        tutorial_number = str(idx)
        tutorial_title = clean_name

    structured_chunks_list = []
    
    # 3. Restructure internal matrices into your target JSON schema format
    for segment in result["segments"]:
        chunk_obj = {
            "number": tutorial_number,
            "title": tutorial_title,
            "start": round(segment["start"], 2), # Rounds floating points to 2 clean decimals
            "end": round(segment["end"], 2),
            "text": segment["text"]
        }
        structured_chunks_list.append(chunk_obj)
        
    final_json_payload = {
        "chunks": structured_chunks_list
    }
    
    # 4. Save the indented file logs
    with open(output_json_path, "w", encoding="utf-8") as jf:
        json.dump(final_json_payload, jf, indent=4, ensure_ascii=False)
        
    print(f"✓ Successfully saved: {output_json_path}\n")

print(" Complete! All video lecture JSON chunk logs are ready inside 'jsons/'.")
