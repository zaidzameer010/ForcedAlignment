import argparse
from pathlib import Path

import librosa
import torch
from pydub import AudioSegment
from silero_vad import get_speech_timestamps, load_silero_vad

def split_audio(audio_path: str, segment_minutes: int = 10):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: File {audio_path} not found.")
        return

    # Create output directory
    output_dir = audio_path.parent / audio_path.stem
    output_dir.mkdir(exist_ok=True)
    print(f"Created directory: {output_dir}")

    # Load VAD model
    model = load_silero_vad()
    
    # Read audio for VAD without torchcodec dependency (use librosa + torch)
    sampling_rate = 16000
    y, sr = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
    wav = torch.from_numpy(y)
    
    # Get speech timestamps
    print("Analyzing audio for speech segments...")
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=sampling_rate,
        return_seconds=True
    )

    # Load audio for splitting (pydub)
    print("Loading audio for splitting...")
    audio = AudioSegment.from_file(str(audio_path))
    total_duration_s = len(audio) / 1000.0
    
    target_segment_s = segment_minutes * 60
    split_points = [0.0]
    
    current_target = target_segment_s
    
    # Find split points near target intervals
    while current_target < total_duration_s:
        # Find the best split point near current_target
        # We look for the largest gap (silence) between speech timestamps near the target
        best_split = current_target
        min_diff = float('inf')
        
        # Check gaps between speech segments
        for i in range(len(speech_timestamps) - 1):
            gap_start = speech_timestamps[i]['end']
            gap_end = speech_timestamps[i+1]['start']
            
            # If the target is within or near this gap, it's a good split point
            # We prefer splitting in the middle of a gap
            mid_gap = (gap_start + gap_end) / 2
            diff = abs(mid_gap - current_target)
            
            # Only consider gaps within a reasonable window (e.g., +/- 30 seconds)
            if diff < 30 and diff < min_diff:
                min_diff = diff
                best_split = mid_gap
        
        # If we didn't find a good gap within 30s, just take the current target
        split_points.append(best_split)
        current_target = best_split + target_segment_s

    split_points.append(total_duration_s)
    
    # Perform splitting
    for i in range(len(split_points) - 1):
        start_ms = int(split_points[i] * 1000)
        end_ms = int(split_points[i+1] * 1000)
        
        segment = audio[start_ms:end_ms]
        output_filename = output_dir / f"{i+1}{audio_path.suffix}"
        
        print(f"Exporting segment {i+1}: {start_ms/1000:.2f}s to {end_ms/1000:.2f}s")
        segment.export(str(output_filename), format=audio_path.suffix[1:])

    print(f"Done! Split into {len(split_points)-1} segments in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio into segments using VAD to avoid aggressive cuts.")
    parser.add_argument("audio", type=str, help="Path to the input audio file")
    parser.add_argument("--minutes", type=int, default=10, help="Target segment duration in minutes (default: 10)")
    
    args = parser.parse_args()
    split_audio(args.audio, args.minutes)
