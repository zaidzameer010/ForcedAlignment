# ForcedAlignment.py

A full-featured, GPU-accelerated audio/video transcription and forced alignment pipeline built on top of [stable-ts](https://github.com/jianfch/stable-ts) (Stable Whisper). It is designed to handle extremely long files reliably using Voice Activity Detection (VAD) chunking, and comes out-of-the-box with context-biasing optimized for Islamic/Urdu/Arabic lectures and scholarly content.

---

## Key Features

- **Dual Modes (`--mode`)**:
  - **`align`**: Force-aligns an existing text transcript (`.txt` or `.md`) to an audio/video file.
  - **`transcribe`**: Generates a completely new transcript from audio/video.
- **Robust VAD Segmenting**: Uses `silero-vad` to detect natural speech boundaries and split long audio files into smaller segments (default: 5 minutes) before running Whisper. This prevents GPU memory issues, hallucination loops, and time-drift.
- **Context-Biased Prompting**: Biases the transcription model with specialized religious prompts for Urdu and Arabic vocabulary, ensuring correct spellings of sacred names and scholarly terms (e.g., Allah, Muhammad ﷺ, Quran, Hadith, Imam Abu Hanifa, etc.).
- **Automatic Hardware Acceleration**: Auto-detects and utilizes Apple Silicon MPS (both MLX and PyTorch versions), NVIDIA CUDA, or CPU fallback.
- **Timestamp Refinement**: Optionally runs model-guided refinement (`--refine`) on word boundaries to ensure ultra-precise timing.
- **Advanced Subtitle Regrouping & Splitting**: Fine-tune your output using stable-ts string DSL, character count limits, word count limits, gap duration, or punctuation markers (including Urdu `۔` and Arabic `؟`).
- **Rich Terminal UI**: Displays progressive task progress bars, table-based statistics, and interactive file pickers when command-line arguments are omitted.
- **Multi-Format Export**: Generates SRT, VTT, ASS, TSV, and JSON formats simultaneously with options for word-level highlights and progressive karaoke effects.

---

## Installation

### 1. Install System Dependencies
Make sure you have `ffmpeg` installed on your system.
- **macOS (via Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
  
### 2. Install Python Dependencies
Install the required python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage

You can run `ForcedAlignment.py` interactively or by passing direct CLI arguments.

### 1. Interactive Mode
If you run the script without arguments, it will automatically search your current working directory for audio/video files and transcripts, showing a interactive menu to select files:
```bash
python ForcedAlignment.py
```

### 2. Standard Command-Line Usage

#### Transcription Mode
Generate a transcript from an audio or video file:
```bash
python ForcedAlignment.py path/to/audio.mp3 --mode transcribe --model large-v3-turbo --language ur
```

#### Forced Alignment Mode (Default)
Align an existing text transcript file to an audio or video file:
```bash
python ForcedAlignment.py path/to/audio.mp3 path/to/transcript.txt --mode align
```

---

## Command Line Reference

### Input / Output Options
*   `audio` (Positional): Path to the audio or video file.
*   `transcript` (Positional): Path to the transcript text file (only required/prompted for `align` mode).
*   `--mode`: `align` or `transcribe` (default: `align`).
*   `--output-name <name>`: Custom base name for the output files (defaults to audio file name).
*   `--output-dir <path>`: Directory where output files will be saved (defaults to working directory).
*   `--formats <fmts>`: Comma-separated list of formats to save (options: `srt`, `vtt`, `ass`, `tsv`, `json`; default: `srt`).

### Model Options
*   `--model <model_name>`: Stable-TS Whisper model name or HF hub path (default: `large-v3-turbo`).
*   `--language <lang>`: Language code (e.g., `ur`, `ar`, `en`, `auto`). Default is `ur`.
*   `--initial-prompt <prompt>`: Initial prompt to bias transcription. (Defaults to Urdu/Arabic religious prompt for `ur` or `auto` languages).
*   `--beam-size <size>`: Beam size for transcription (default: `10`).
*   `--temperature <temp>`: Temperature for transcription (default: `0.0`).

### VAD & Segmenting
*   `--segment-length <seconds>`: Length of audio segments for sequential processing (default: `300.0` or 5 minutes).
*   `--vad-threshold <float>`: Silero-VAD threshold for detecting speech (default: `0.35`).

### Timestamp Refinement
*   `--refine`: Enable post-transcription/alignment timestamp refinement using Whisper attention weights.
*   `--refine-precision <seconds>`: Refinement step precision (default: `0.1`).
*   `--refine-prob-threshold <float>`: Probability threshold below which to stop refinement (default: `0.5`).

### Regrouping & Splitting (Subtitle Customization)
*   `--regroup <dsl>`: Regroup segments using stable-ts string DSL (e.g. `da` for default algorithm, or `ms_sg=.5_mg=.15+3`).
*   `--split-gap <seconds>`: Split segments where the gap between consecutive words exceeds this duration.
*   `--split-chars <int>`: Split segments if their character count exceeds this limit.
*   `--split-words <int>`: Split segments if their word count exceeds this limit.
*   `--split-punctuation <list>`: Split segments at these punctuation marks (comma-separated, e.g. `.,?,!,۔,؟`).

### Styling & Highlighting
*   `--word-level`: Enable word-level timestamps/highlights in the generated subtitle files.
*   `--no-segment-level`: Disable segment-level timestamps in the generated subtitle files.
*   `--highlight-tag <tags>`: Wrap the active/spoken word in these tags (default: `<b>,</b>`).
*   `--karaoke`: Use progressive filling highlight (karaoke effect) in ASS subtitle output.

---

## License
This pipeline is open-source. For underlying Whisper model license details, check the [OpenAI Whisper repository](https://github.com/openai/whisper). Refer to [stable-ts](https://github.com/jianfch/stable-ts) for upstream API documentation.
