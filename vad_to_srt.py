import argparse
from pathlib import Path
from typing import List, Optional

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


def format_srt_time(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vad_to_srt",
        description="Generate an SRT file containing Silero VAD speech segments.",
    )
    parser.add_argument("audio", type=str, help="Path to input audio/video file")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output SRT path (default: alongside input, with .srt extension)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        choices=[8000, 16000],
        help="Audio sampling rate to use for VAD (Silero supports 8000 or 16000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Speech probability threshold (0..1)",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=250,
        help="Discard speech chunks shorter than this duration (ms)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=180,
        help="Minimum silence duration to separate chunks (ms)",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=100,
        help="Padding added to each speech chunk (ms)",
    )
    parser.add_argument(
        "--neg-threshold",
        type=float,
        default=None,
        help="Exit threshold for ending speech (default: threshold - 0.15)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Subtitle text to write for each speech segment",
    )
    parser.add_argument(
        "--number-label",
        action="store_true",
        help="If set, write an incremental number in the subtitle text for each segment (optionally prefixed by --label)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    out_path = Path(args.out) if args.out else audio_path.with_suffix(".srt")

    model = load_silero_vad()
    wav = read_audio(str(audio_path), sampling_rate=int(args.sampling_rate))

    vad_kwargs = {
        "threshold": float(args.threshold),
        "sampling_rate": int(args.sampling_rate),
        "min_speech_duration_ms": int(args.min_speech_ms),
        "min_silence_duration_ms": int(args.min_silence_ms),
        "speech_pad_ms": int(args.speech_pad_ms),
        "return_seconds": True,
    }
    if args.neg_threshold is not None:
        vad_kwargs["neg_threshold"] = float(args.neg_threshold)

    speech_timestamps = get_speech_timestamps(wav, model, **vad_kwargs)

    lines: List[str] = []
    for i, ts in enumerate(speech_timestamps, start=1):
        start_s = float(ts["start"])
        end_s = float(ts["end"])
        if end_s < start_s:
            end_s = start_s
        lines.append(str(i))
        lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
        if args.number_label:
            if args.label:
                lines.append(f"{args.label} {i}")
            else:
                lines.append(str(i))
        else:
            lines.append(args.label)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
