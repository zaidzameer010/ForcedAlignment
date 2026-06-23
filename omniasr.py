#!/usr/bin/env python3
"""
Urdu/Arabic Specific ASR Transcription Pipeline
Using omnilingual-asr (3B CTC Model) & Silero VAD.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

# Verify rich installation first
try:
    import rich
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
except ImportError:
    print("Error: The 'rich' library is required to run this script. Please install it using: pip install rich")
    sys.exit(1)

console = Console()

# Verify heavy ML libraries and handle PyTorch/TorchAudio mismatch gracefully
try:
    import torch
    import librosa
    import soundfile as sf
    from silero_vad import get_speech_timestamps, load_silero_vad
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
except (ImportError, OSError) as e:
    console.print(Panel.fit(
        f"[bold red]Initialization Error:[/bold red]\n\n"
        f"[yellow]{str(e)}[/yellow]\n\n"
        f"This is typically caused by a version mismatch between [bold]torch[/bold] and [bold]torchaudio[/bold] "
        f"in your environment (e.g. conda environment 'tts').\n\n"
        f"Please run one of the following commands to resolve this issue:\n"
        f"1. [bold cyan]conda install -n tts pytorch torchaudio -c pytorch[/bold cyan]\n"
        f"2. [bold cyan]pip install --force-reinstall torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu[/bold cyan] (CPU only)\n"
        f"3. [bold cyan]pip install --force-reinstall torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121[/bold cyan] (GPU/CUDA)\n",
        title="Environment Setup Check",
        border_style="red"
    ))
    sys.exit(1)


def format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp string HH:MM:SS,mmm"""
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def find_audio_files() -> List[Path]:
    """Find all audio and video files in the current working directory."""
    cwd = Path.cwd()
    extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mkv", ".avi", ".mov"}
    found_files = []
    for p in cwd.iterdir():
        if p.is_file() and p.suffix.lower() in extensions:
            found_files.append(p)
    return sorted(found_files, key=lambda f: f.name.lower())


def chunk_segments(speech_timestamps: List[dict], max_duration: float = 30.0) -> List[dict]:
    """
    Sub-split any speech segments longer than max_duration to prevent
    omnilingual-asr length limit errors (since models were trained on <= 30s segments).
    """
    chunked_segments = []
    for ts in speech_timestamps:
        start = float(ts['start'])
        end = float(ts['end'])
        duration = end - start
        if duration <= max_duration:
            chunked_segments.append({'start': start, 'end': end})
        else:
            curr = start
            while curr < end:
                curr_end = min(curr + max_duration, end)
                # Avoid leaving a tiny fragment at the end
                if end - curr_end < 2.0:
                    curr_end = end
                chunked_segments.append({'start': curr, 'end': curr_end})
                curr = curr_end
    return chunked_segments


def main():
    console.print(Panel(
        "[bold green]Omnilingual ASR Transcription Pipeline[/bold green]\n"
        "Using [bold cyan]omniASR_CTC_3B_v2[/bold cyan] for Urdu & Arabic specific speech transcription.",
        border_style="green"
    ))

    # 1. Discover audio/video files
    files = find_audio_files()
    if not files:
        console.print("[bold red]No supported audio or video files found in the current directory.[/bold red]")
        console.print("Supported formats: .mp3, .wav, .flac, .ogg, .m4a, .aac, .mp4, .mkv, .avi, .mov")
        return

    # 2. Interactive CLI Selection
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", width=4)
    table.add_column("File Name", overflow="fold")
    table.add_column("Size", justify="right", width=12)

    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        table.add_row(str(i), f.name, f"{size_mb:.2f} MB")

    console.print(table)
    choices = [str(i) for i in range(1, len(files) + 1)]
    selected_idx = Prompt.ask("Select an audio/video file to transcribe", choices=choices, default="1")
    selected_file = files[int(selected_idx) - 1]
    console.print(f"Selected file: [bold yellow]{selected_file.name}[/bold yellow]\n")

    # 3. Interactive Device/Parameters Selection
    device_choices = ["auto", "cuda", "mps", "cpu"]
    selected_device = Prompt.ask("Select processing device", choices=device_choices, default="auto")

    # Determine device and dtype
    if selected_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
    else:
        device = selected_device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Model Card selection for quality/accuracy options
    model_choices = ["1", "2"]
    console.print("\n[bold magenta]Select Model Type (CTC for speed, LLM for accuracy):[/bold magenta]")
    console.print(" 1) CTC 3B Model ([cyan]omniASR_CTC_3B_v2[/cyan]) - Fast parallel decoding, no language conditioning")
    console.print(" 2) LLM 3B Model ([cyan]omniASR_LLM_Unlimited_3B_v2[/cyan]) - Autoregressive, supports language conditioning (Recommended for highest accuracy)")
    selected_model_idx = Prompt.ask("Choose model option", choices=model_choices, default="2")
    model_card = "omniASR_CTC_3B_v2" if selected_model_idx == "1" else "omniASR_LLM_Unlimited_3B_v2"

    # Language selection for explicit Urdu/Arabic targeting (always prompted)
    lang_choices = ["1", "2", "3"]
    console.print("\n[bold magenta]Select Target Language (biases script accuracy for LLM model):[/bold magenta]")
    console.print(" 1) Urdu ([cyan]urd_Arab[/cyan]) - For Urdu speech / religious lectures (handles Arabic quotes well)")
    console.print(" 2) Arabic ([cyan]ara_Arab[/cyan]) - For pure Arabic speech / Quranic recitation")
    console.print(" 3) Auto-detect (Let model decide)")
    selected_lang_idx = Prompt.ask("Choose language option", choices=lang_choices, default="1")
    
    lang_code = None
    if selected_lang_idx == "1":
        selected_lang_name = "urd_Arab"
    elif selected_lang_idx == "2":
        selected_lang_name = "ara_Arab"
    else:
        selected_lang_name = None

    if model_card == "omniASR_LLM_Unlimited_3B_v2":
        lang_code = selected_lang_name
    elif selected_lang_name is not None:
        console.print(f"[yellow]Note: CTC model selected. Language conditioning ({selected_lang_name}) is ignored for CTC model.[/yellow]")

    batch_size_str = Prompt.ask("\nEnter batch size (larger values require more memory)", default="8")
    try:
        batch_size = max(1, int(batch_size_str))
    except ValueError:
        batch_size = 8

    console.print(f"\n[bold green]Configuration Summary:[/bold green]")
    console.print(f" - Model: [cyan]{model_card}[/cyan]")
    console.print(f" - Target Device: [cyan]{device}[/cyan]")
    console.print(f" - Target Precision: [cyan]{dtype}[/cyan]")
    console.print(f" - Batch Size: [cyan]{batch_size}[/cyan]")
    if lang_code:
        console.print(f" - Forced Language: [cyan]{lang_code}[/cyan]")
    else:
        console.print(f" - Forced Language: [cyan]Auto-detect / None[/cyan]")
    console.print("")

    # 4. Load VAD Model & Segment Audio
    with console.status("[bold blue]Loading Silero VAD model...[/bold blue]"):
        vad_model = load_silero_vad()

    with console.status(f"[bold blue]Loading and resampling audio to 16kHz...[/bold blue]"):
        try:
            y, sr = librosa.load(str(selected_file), sr=16000, mono=True)
        except Exception as e:
            console.print(f"[bold red]Failed to load audio file:[/bold red] {e}")
            return

    with console.status("[bold blue]Detecting speech segments using VAD...[/bold blue]"):
        wav_tensor = torch.from_numpy(y)
        raw_segments = get_speech_timestamps(
            wav_tensor,
            vad_model,
            sampling_rate=16000,
            return_seconds=True
        )

    if not raw_segments:
        console.print("[bold yellow]Warning: No speech segments detected by Silero VAD.[/bold yellow]")
        console.print("Writing an empty SRT file and exiting...")
        out_srt = selected_file.with_suffix(".srt")
        out_srt.write_text("", encoding="utf-8")
        return

    # Sub-split long segments (>30s) to fit model sequence boundaries
    segments = chunk_segments(raw_segments, max_duration=30.0)
    console.print(f"Detected [bold green]{len(raw_segments)}[/bold green] raw speech segments.")
    console.print(f"Split into [bold green]{len(segments)}[/bold green] chunks (max 30s each) for inference.\n")

    # 5. Initialize omnilingual-asr Pipeline
    with console.status(f"[bold blue]Loading {model_card} model on {device}...[/bold blue]"):
        try:
            pipeline = ASRInferencePipeline(
                model_card=model_card,
                device=device,
                dtype=dtype
            )
        except Exception as e:
            console.print(f"[bold red]Failed to load omnilingual-asr pipeline:[/bold red] {e}")
            if device == "mps":
                console.print("[yellow]Tip: MPS device can sometimes fail with unsupported PyTorch operators. Try running on 'cpu' or 'cuda'.[/yellow]")
            return

    # 6. Run Transcription Loop
    transcriptions = []
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        ) as progress:
            task_id = progress.add_task(f"Transcribing {selected_file.name}", total=len(segments))

            for i in range(0, len(segments), batch_size):
                batch_segs = segments[i:i + batch_size]
                batch_inputs = []

                for seg in batch_segs:
                    start_idx = int(seg['start'] * 16000)
                    end_idx = int(seg['end'] * 16000)
                    waveform = y[start_idx:end_idx]
                    batch_inputs.append({"waveform": waveform, "sample_rate": 16000})

                # Transcribe batch with optional language conditioning
                transcribe_kwargs = {"batch_size": len(batch_inputs)}
                if lang_code:
                    transcribe_kwargs["lang"] = [lang_code] * len(batch_inputs)

                batch_trans = pipeline.transcribe(batch_inputs, **transcribe_kwargs)
                transcriptions.extend(batch_trans)
                progress.update(task_id, advance=len(batch_segs))
    except Exception as e:
        console.print(f"[bold red]Inference error occurred during transcription:[/bold red] {e}")
        return

    # 7. Write SRT Subtitle File
    srt_path = selected_file.with_suffix(".srt")
    try:
        with srt_path.open("w", encoding="utf-8") as f:
            for idx, (seg, text) in enumerate(zip(segments, transcriptions), start=1):
                start_str = format_srt_time(seg['start'])
                end_str = format_srt_time(seg['end'])
                clean_text = (text or "").strip()
                f.write(f"{idx}\n{start_str} --> {end_str}\n{clean_text}\n\n")
    except Exception as e:
        console.print(f"[bold red]Failed to write SRT file:[/bold red] {e}")
        return

    console.print(Panel(
        f"[bold green]Transcription Completed Successfully![/bold green]\n\n"
        f" - [bold]Output SRT File:[/bold] [cyan]{srt_path.name}[/cyan]\n"
        f" - [bold]Total SRT Entries:[/bold] {len(segments)}\n"
        f" - [bold]Audio Duration:[/bold] {len(y)/16000:.2f} seconds\n"
        f" - [bold]Saved in:[/bold] {srt_path.parent}",
        border_style="green",
        title="Success"
    ))


if __name__ == "__main__":
    main()
