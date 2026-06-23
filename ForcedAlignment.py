from __future__ import annotations

import sys
sys.modules['torchvision'] = None

# Monkeypatch transformers package to bypass deprecated and removed arguments/methods in stable-ts
try:
	from transformers import AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration
	orig_from_pretrained = AutoModelForSpeechSeq2Seq.from_pretrained
	AutoModelForSpeechSeq2Seq.from_pretrained = lambda *a, **k: (k.pop('use_flash_attention_2', None), orig_from_pretrained(*a, **k))[1]
	WhisperForConditionalGeneration.to_bettertransformer = lambda self, *args, **kwargs: self
except Exception:
	pass

import argparse
import textwrap
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

stable_whisper: Any
try:
	import stable_whisper
except Exception as import_error:  # pragma: no cover - surfaced at runtime
	stable_whisper = None
	STABLE_TS_IMPORT_ERROR = import_error
else:
	STABLE_TS_IMPORT_ERROR = None


AUDIO_EXTENSIONS = {
	".aac",
	".flac",
	".m4a",
	".mp3",
	".mp4",
	".ogg",
	".wav",
	".webm",
}

# Default religious/scholarly Urdu/Arabic prompt for biasing model
DEFAULT_RELIGIOUS_PROMPT = (
	"یہ ایک مذہبی بیان ہے جس میں عربی آیاتِ قرآنی، احادیث اور اردو تشریحات شامل ہیں۔ "
	"برائے مہربانی مقدس ناموں اور اصطلاحات کو درست اور مکمل لکھیں جیسے: اللہ، محمد ﷺ، قرآن، حدیث، سبحان اللہ، "
	"الحمد للہ، ان شاء اللہ، سورۃ الفاتحہ، صحیح بخاری، امام ابو حنیفہ، تفسیر ابن کثیر۔"
)


def get_segment_intervals(console: Console, audio_path: Path, segment_length_s: float = 300.0) -> list[tuple[float, float]]:
	try:
		from silero_vad import get_speech_timestamps, load_silero_vad
		import librosa
		import torch
		from pydub import AudioSegment

		console.print("[bold green]Loading VAD model for voice activity detection...[/bold green]")
		sampling_rate = 16000
		y, sr = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
		wav = torch.from_numpy(y)

		vad_model = load_silero_vad()
		speech_timestamps = get_speech_timestamps(
			wav,
			vad_model,
			sampling_rate=sampling_rate,
			return_seconds=True
		)

		audio = AudioSegment.from_file(str(audio_path))
		total_duration_s = len(audio) / 1000.0

		split_points = [0.0]
		current_target = segment_length_s

		while current_target < total_duration_s:
			best_split = current_target
			min_diff = float('inf')

			for i in range(len(speech_timestamps) - 1):
				gap_start = speech_timestamps[i]['end']
				gap_end = speech_timestamps[i+1]['start']

				mid_gap = (gap_start + gap_end) / 2
				diff = abs(mid_gap - current_target)

				if diff < 30.0 and diff < min_diff:
					min_diff = diff
					best_split = mid_gap

			split_points.append(best_split)
			current_target = best_split + segment_length_s

		split_points.append(total_duration_s)

		return [(split_points[i], split_points[i+1]) for i in range(len(split_points) - 1)]

	except Exception as e:
		console.print(f"[bold yellow]VAD segmenting failed or libraries not available ({e}). Falling back to simple uniform chunking.[/bold yellow]")
		try:
			from pydub import AudioSegment
			audio = AudioSegment.from_file(str(audio_path))
			total_duration_s = len(audio) / 1000.0
		except Exception:
			try:
				import soundfile as sf
				info = sf.info(str(audio_path))
				total_duration_s = info.duration
			except Exception:
				total_duration_s = 0.0

		if total_duration_s == 0.0:
			return [(0.0, 0.0)]

		split_points = []
		curr = 0.0
		while curr < total_duration_s:
			next_pt = min(curr + segment_length_s, total_duration_s)
			split_points.append((curr, next_pt))
			curr = next_pt
		return split_points




def list_media_files(directory: Path) -> list[Path]:
	return sorted(
		[path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS],
		key=lambda path: path.name.lower(),
	)


def list_text_files(directory: Path) -> list[Path]:
	return sorted(
		[path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".txt", ".md"}],
		key=lambda path: path.name.lower(),
	)


def select_file(console: Console, files: Sequence[Path], prompt: str) -> Path:
	table = Table(show_header=True, header_style="bold magenta")
	table.add_column("#", justify="right", width=4)
	table.add_column("File", overflow="fold")
	for index, file_path in enumerate(files, start=1):
		table.add_row(str(index), file_path.name)
	console.print(table)
	choices = [str(index) for index in range(1, len(files) + 1)]
	selected = Prompt.ask(prompt, choices=choices, default="1")
	return files[int(selected) - 1]


def choose_runtime_device() -> tuple[str, str]:
	try:
		import torch
	except Exception:
		return "cpu", "CPU"

	if torch.cuda.is_available():
		return "cuda", "NVIDIA CUDA"
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return "mps", "Apple Silicon MPS"
	return "cpu", "CPU"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		prog="ForcedAlignment",
		description="Full-fledged alignment and transcription pipeline using stable-ts.",
	)
	parser.add_argument("audio", nargs="?", help="Path to the audio or video file")
	parser.add_argument("transcript", nargs="?", help="Path to the transcript text file (only required for alignment)")
	
	# Core Pipeline Modes
	parser.add_argument(
		"--mode",
		type=str,
		default="align",
		choices=["align", "transcribe"],
		help="Task mode: 'align' to align transcript, 'transcribe' to generate a new transcript (default: align)",
	)
	
	# Model Loading Configurations
	parser.add_argument(
		"--model",
		type=str,
		default="large-v3-turbo",
		help="Stable-TS Whisper model name or HF hub path",
	)
	parser.add_argument(
		"--language",
		type=str,
		default="ur",
		help="Language code (e.g., ur, ar, en, auto). Default is ur.",
	)
	parser.add_argument(
		"--vad-threshold",
		type=float,
		default=0.35,
		help="VAD threshold for detecting speech. Default is 0.35.",
	)
	parser.add_argument(
		"--segment-length",
		type=float,
		default=300.0,
		help="Length of audio segments in seconds for sequential processing. Default is 300.0 (5 minutes).",
	)
	parser.add_argument(
		"--beam-size",
		type=int,
		default=10,
		help="Beam size for transcription. Default is 5.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Temperature for transcription. Default is 0.0.",
	)
	parser.add_argument(
		"--initial-prompt",
		type=str,
		default=None,
		help="Initial prompt to bias the transcription (defaults to Urdu/Arabic religious prompt for 'ur' language).",
	)
	
	# Refinement Configurations
	parser.add_argument(
		"--refine",
		action="store_true",
		help="Enable post-transcription/alignment timestamp refinement (model.refine).",
	)
	parser.add_argument(
		"--refine-precision",
		type=float,
		default=0.1,
		help="Refinement step precision in seconds. Default is 0.1.",
	)
	parser.add_argument(
		"--refine-prob-threshold",
		type=float,
		default=0.5,
		help="Probability threshold below which to stop refinement. Default is 0.5.",
	)
	
	# Regrouping and Splitting Configurations
	parser.add_argument(
		"--regroup",
		type=str,
		default=None,
		help="Regroup segments using stable-ts string DSL (e.g. 'ms_sg=.5_mg=.15+3' or 'da' for default algorithm).",
	)
	parser.add_argument(
		"--split-gap",
		type=float,
		default=None,
		help="Split segments where the gap between consecutive words exceeds this duration (in seconds).",
	)
	parser.add_argument(
		"--split-chars",
		type=int,
		default=None,
		help="Split segments if their character count exceeds this limit.",
	)
	parser.add_argument(
		"--split-words",
		type=int,
		default=None,
		help="Split segments if their word count exceeds this limit.",
	)
	parser.add_argument(
		"--split-punctuation",
		type=str,
		default=None,
		help="Split segments at these punctuation marks (comma-separated, e.g. '.,?,!,۔,؟').",
	)
	
	# Export and Highlighting Configurations
	parser.add_argument(
		"--formats",
		type=str,
		default="srt",
		help="Comma-separated list of formats to save (srt, vtt, ass, tsv, json). Default is srt.",
	)
	parser.add_argument(
		"--word-level",
		action="store_true",
		help="Enable word-level timestamps/highlights in the generated subtitle files.",
	)
	parser.add_argument(
		"--no-segment-level",
		action="store_true",
		help="Disable segment-level timestamps in the generated subtitle files.",
	)
	parser.add_argument(
		"--highlight-tag",
		type=str,
		default="<b>,</b>",
		help="Highlight tag(s) to wrap the active word. Can be a single tag (e.g., '<u>') or comma-separated pairs (e.g. '<b>,</b>'). Default is '<b>,</b>'.",
	)
	parser.add_argument(
		"--karaoke",
		action="store_true",
		help="Use progressive filling highlight (karaoke effect) in ASS output.",
	)
	parser.add_argument(
		"--output-name",
		type=str,
		default=None,
		help="Base name for the output files (defaults to audio file name).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Directory where output files will be saved (defaults to working directory).",
	)
	
	return parser.parse_args(argv)


def resolve_inputs(console: Console, args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
	cwd = Path.cwd()
	audio_path = Path(args.audio).expanduser() if args.audio else None
	transcript_path = Path(args.transcript).expanduser() if args.transcript else None

	if audio_path is not None and not audio_path.is_absolute():
		audio_path = cwd / audio_path
	if transcript_path is not None and not transcript_path.is_absolute():
		transcript_path = cwd / transcript_path

	console.print(f"[bold blue]Current working directory:[/bold blue] {cwd}")

	if audio_path is None:
		audio_files = list_media_files(cwd)
		if not audio_files:
			raise SystemExit("No supported audio files were found in the current directory.")
		audio_path = select_file(console, audio_files, "Select audio file")

	if args.mode == "align" and transcript_path is None:
		transcript_files = list_text_files(cwd)
		if not transcript_files:
			raise SystemExit("No transcript text files (.txt/.md) were found in the current directory.")
		transcript_path = select_file(console, transcript_files, "Select transcript file")

	if audio_path is None or not audio_path.exists():
		raise SystemExit(f"Audio file not found: {audio_path}")
	if args.mode == "align" and (transcript_path is None or not transcript_path.exists()):
		raise SystemExit(f"Transcript file not found: {transcript_path}")

	return audio_path, transcript_path


def choose_model(console: Console, args: argparse.Namespace):
	device, device_label = choose_runtime_device()
	if STABLE_TS_IMPORT_ERROR is not None:
		raise SystemExit(
			f"stable-ts import failed: {STABLE_TS_IMPORT_ERROR}\n"
			"Install the dependencies listed in requirements.txt."
		)

	standard_models = {
		"tiny.en", "tiny", "base.en", "base", "small.en", "small",
		"medium.en", "medium", "large-v1", "large-v2", "large-v3",
		"large", "large-v3-turbo", "turbo"
	}
	is_hf = args.model not in standard_models

	if is_hf:
		if device == "mps":
			try:
				model = stable_whisper.faster_whisper(args.model, device="mps")
				return model, "mps", "Apple Silicon MPS"
			except Exception as mlx_error:
				console.print(f"[bold yellow]Failed to load HF model using MLX on MPS ({mlx_error}). Trying standard PyTorch MPS...[/bold yellow]")
				try:
					model = stable_whisper.load_hf_whisper(args.model, device="mps")
					return model, "mps", "Apple Silicon MPS (PyTorch)"
				except Exception as mps_error:
					console.print(f"[bold yellow]Failed to load Hugging Face model on MPS, falling back to CPU:[/bold yellow] {mps_error}")
					model = stable_whisper.load_hf_whisper(args.model, device="cpu")
					return model, "cpu", "CPU"
		elif device == "cuda":
			model = stable_whisper.load_hf_whisper(args.model, device="cuda")
			return model, device, device_label
		else:
			model = stable_whisper.load_hf_whisper(args.model, device="cpu")
			return model, device, device_label
	else:
		if device == "mps":
			try:
				model = stable_whisper.load_mlx_whisper(args.model, device="mps")
				return model, "mps", "Apple Silicon MPS"
			except Exception as mlx_error:
				console.print(f"[bold yellow]Failed to load standard model using MLX on MPS ({mlx_error}). Trying standard PyTorch MPS...[/bold yellow]")
				try:
					model = stable_whisper.load_model(args.model, device="mps")
					return model, "mps", "Apple Silicon MPS (PyTorch)"
				except Exception as mps_error:
					console.print(f"[bold yellow]Failed to load standard model on MPS, falling back to CPU:[/bold yellow] {mps_error}")
					model = stable_whisper.load_model(args.model, device="cpu")
					return model, "cpu", "CPU"
		elif device == "cuda":
			model = stable_whisper.load_model(args.model, device="cuda")
			return model, device, device_label
		else:
			model = stable_whisper.load_model(args.model)
			return model, device, device_label


def slice_whisper_result(result, n_words):
	new_segments = []
	word_count = 0
	for segment in result.segments:
		segment_words = []
		for w in segment.words:
			if word_count < n_words:
				segment_words.append(w)
				word_count += 1
			else:
				break
		if segment_words:
			segment.words = segment_words
			segment.start = segment_words[0].start
			segment.end = segment_words[-1].end
			new_segments.append(segment)
		if word_count >= n_words:
			break
	result.segments = new_segments
	return result


def process_audio(
	console: Console,
	model,
	audio_path: Path,
	transcript_path: Optional[Path],
	args: argparse.Namespace,
) -> Any:
	# Build language settings
	language_val = args.language if args.language and args.language != "auto" else None

	# Resolve initial prompt
	init_prompt = args.initial_prompt
	if init_prompt is None and (args.language == "ur" or args.language == "auto"):
		init_prompt = DEFAULT_RELIGIOUS_PROMPT
		console.print("[bold cyan]Using default religious prompt to bias transcription for Urdu/Arabic context.[/bold cyan]")

	# Get segment intervals
	intervals = get_segment_intervals(console, audio_path, segment_length_s=args.segment_length)
	is_mlx = 'mlx' in getattr(model, '__module__', '').lower() or 'mlx' in type(model).__name__.lower()

	# If audio is under the target segment length (1 segment), bypass chunking and process in a single pass
	if len(intervals) <= 1:
		console.print(f"[bold green]Audio duration is under {args.segment_length} seconds. Processing in a single pass.[/bold green]")
		progress = Progress(
			SpinnerColumn(),
			TextColumn("[bold blue]{task.description}"),
			BarColumn(),
			TaskProgressColumn(),
			TimeRemainingColumn(),
			console=console,
			transient=True,
		)
		with progress:
			task_id = progress.add_task(f"Running Whisper {args.mode}", total=None)

			def make_callback(p_obj, t_id):
				def cb(seek, total_duration):
					if total_duration and total_duration > 0:
						p_obj.update(t_id, completed=seek, total=total_duration)
					return cb
				return cb

			callback_func = make_callback(progress, task_id) if not is_mlx else None

			if args.mode == "align":
				if not transcript_path:
					raise SystemExit("Transcript file is required for alignment mode.")
				transcript_text = transcript_path.read_text(encoding="utf-8-sig")
				if not transcript_text.strip():
					raise SystemExit("The transcript file is empty.")

				align_kwargs = {
					"language": language_val,
					"original_split": True,
					"vad_threshold": args.vad_threshold,
				}
				if callback_func:
					align_kwargs["progress_callback"] = callback_func

				result = model.align(
					str(audio_path),
					transcript_text,
					**align_kwargs,
				)
			else:
				transcribe_kwargs = {
					"language": language_val,
					"vad": True,
					"vad_threshold": args.vad_threshold,
					"initial_prompt": init_prompt,
					"temperature": args.temperature,
				}
				if not is_mlx:
					transcribe_kwargs["beam_size"] = args.beam_size
				if callback_func:
					transcribe_kwargs["progress_callback"] = callback_func

				result = model.transcribe(
					str(audio_path),
					**transcribe_kwargs,
				)
	else:
		console.print(f"[bold green]Audio duration exceeds {args.segment_length} seconds. Split into {len(intervals)} segments using VAD. Processing sequentially...[/bold green]")

		from pydub import AudioSegment
		audio = AudioSegment.from_file(str(audio_path))
		all_segments = []

		progress = Progress(
			SpinnerColumn(),
			TextColumn("[bold blue]{task.description}"),
			BarColumn(),
			TaskProgressColumn(),
			TimeRemainingColumn(),
			console=console,
			transient=True,
		)

		if args.mode == "align":
			if not transcript_path:
				raise SystemExit("Transcript file is required for alignment mode.")
			remaining_transcript = transcript_path.read_text(encoding="utf-8-sig").strip()
			if not remaining_transcript:
				raise SystemExit("The transcript file is empty.")

			with progress:
				task_id = progress.add_task("Aligning sequentially", total=len(intervals))
				for idx, (start_s, end_s) in enumerate(intervals, start=1):
					if not remaining_transcript:
						console.print(f"[bold yellow]Segment {idx}/{len(intervals)}: Transcript fully aligned. Skipping remaining segments.[/bold yellow]")
						progress.update(task_id, advance=1)
						continue

					console.print(f"[bold blue]Segment {idx}/{len(intervals)}: Aligning {start_s:.2f}s - {end_s:.2f}s[/bold blue]")

					start_ms = int(start_s * 1000)
					end_ms = int(end_s * 1000)
					chunk = audio[start_ms:end_ms]

					with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
						temp_wav_path = tmp.name
					try:
						chunk.export(temp_wav_path, format="wav")

						align_kwargs = {
							"language": language_val,
							"original_split": True,
							"vad_threshold": args.vad_threshold,
						}

						result = model.align(
							temp_wav_path,
							remaining_transcript,
							**align_kwargs,
						)

						if result and result.segments:
							if not language_val and getattr(result, "language", None):
								language_val = result.language

							words = list(result.all_words())
							epsilon = 0.01
							last_aligned_idx = -1
							chunk_duration = end_s - start_s

							for i, w in enumerate(words):
								is_stuck = (w.start >= chunk_duration - epsilon) and (w.end - w.start < epsilon)
								if not is_stuck:
									last_aligned_idx = i

							if last_aligned_idx == -1:
								console.print(f"[bold red]Warning: Alignment failed to match any words in segment {idx}. Skipping segment to prevent stalling.[/bold red]")
							else:
								sliced_result = slice_whisper_result(result, last_aligned_idx + 1)
								sliced_result.offset_time(start_s)
								all_segments.extend(sliced_result.segments)

								remaining_words = words[last_aligned_idx + 1:]
								if remaining_words:
									first_rem = remaining_words[0]
									last_al = words[last_aligned_idx]
									if first_rem.word.strip() == last_al.word.strip():
										is_rem_stuck = (first_rem.start >= chunk_duration - epsilon) and (first_rem.end - first_rem.start < epsilon)
										if is_rem_stuck:
											remaining_words = remaining_words[1:]

								remaining_transcript = "".join([rw.word for rw in remaining_words]).strip()
						else:
							console.print(f"[bold red]Warning: Alignment returned no results for segment {idx}.[/bold red]")
					finally:
						if os.path.exists(temp_wav_path):
							os.remove(temp_wav_path)

					progress.update(task_id, advance=1)

		else:  # transcribe mode
			current_prompt = init_prompt

			with progress:
				task_id = progress.add_task("Transcribing sequentially", total=len(intervals))
				for idx, (start_s, end_s) in enumerate(intervals, start=1):
					console.print(f"[bold blue]Segment {idx}/{len(intervals)}: Transcribing {start_s:.2f}s - {end_s:.2f}s[/bold blue]")

					start_ms = int(start_s * 1000)
					end_ms = int(end_s * 1000)
					chunk = audio[start_ms:end_ms]

					with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
						temp_wav_path = tmp.name
					try:
						chunk.export(temp_wav_path, format="wav")

						transcribe_kwargs = {
							"language": language_val,
							"vad": True,
							"vad_threshold": args.vad_threshold,
							"initial_prompt": current_prompt,
							"temperature": args.temperature,
						}
						if not is_mlx:
							transcribe_kwargs["beam_size"] = args.beam_size

						result = model.transcribe(
							temp_wav_path,
							**transcribe_kwargs,
						)

						if result and result.segments:
							if not language_val and getattr(result, "language", None):
								language_val = result.language

							result.offset_time(start_s)
							all_segments.extend(result.segments)

							transcribed_text = result.text
							if transcribed_text:
								current_prompt = f"{init_prompt or ''}\n{transcribed_text}"[-1000:]
						else:
							console.print(f"[bold red]Warning: Transcription returned no results for segment {idx}.[/bold red]")
					finally:
						if os.path.exists(temp_wav_path):
							os.remove(temp_wav_path)

					progress.update(task_id, advance=1)

		result = stable_whisper.result.WhisperResult([s.to_dict() for s in all_segments])
		result.path = str(audio_path)
		result.language = language_val

	# Timestamp Refinement
	if args.refine:
		with console.status("[bold yellow]Refining timestamps (model.refine)...[/bold yellow]"):
			model.refine(
				str(audio_path),
				result,
				precision=args.refine_precision,
				prob_threshold=args.refine_prob_threshold,
				word_level=args.word_level,
			)

	# Regrouping & Splitting Post-Processing
	if args.regroup:
		regroup_val = args.regroup
		if regroup_val.lower() == "true":
			result.regroup(True)
		elif regroup_val.lower() == "da":
			result.regroup("da")
		else:
			result.regroup(regroup_val)

	if args.split_gap is not None:
		result.split_by_gap(max_gap=args.split_gap)

	if args.split_chars is not None or args.split_words is not None:
		result.split_by_length(max_chars=args.split_chars, max_words=args.split_words)

	if args.split_punctuation is not None:
		punctuation_list = args.split_punctuation.split(",")
		result.split_by_punctuation(punctuation_list)

	return result



def export_results(
	console: Console,
	result: Any,
	audio_path: Path,
	args: argparse.Namespace,
) -> list[Path]:
	# Setup Output Folder and Name
	output_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)
	
	base_name = args.output_name if args.output_name else audio_path.stem
	formats = [fmt.strip().lower() for fmt in args.formats.split(",")]
	
	# Parse Highlight tags
	tags_raw = args.highlight_tag.split(",")
	if len(tags_raw) == 2:
		highlight_tag = (tags_raw[0].strip(), tags_raw[1].strip())
	elif len(tags_raw) == 1 and tags_raw[0]:
		tag_start = tags_raw[0].strip()
		# Guess matching closing tag
		if tag_start.startswith("<") and not tag_start.endswith("/>"):
			tag_end = tag_start.replace("<", "</")
		else:
			tag_end = tag_start
		highlight_tag = (tag_start, tag_end)
	else:
		highlight_tag = ("<b>", "</b>")

	saved_files = []
	segment_lvl = not args.no_segment_level
	word_lvl = args.word_level

	for fmt in formats:
		out_file = output_dir / f"{base_name}.{fmt}"
		if fmt in ("srt", "vtt"):
			result.to_srt_vtt(
				str(out_file),
				segment_level=segment_lvl,
				word_level=word_lvl,
				tag=highlight_tag,
			)
		elif fmt == "ass":
			result.to_ass(
				str(out_file),
				segment_level=segment_lvl,
				word_level=word_lvl,
				tag=highlight_tag,
				karaoke=args.karaoke,
			)
		elif fmt == "tsv":
			result.to_tsv(
				str(out_file),
				segment_level=segment_lvl,
				word_level=word_lvl,
			)
		elif fmt == "json":
			result.save_as_json(str(out_file))
		else:
			console.print(f"[bold red]Unsupported format skipped:[/bold red] {fmt}")
			continue
		saved_files.append(out_file)

	return saved_files


def print_summary(console: Console, result: Any, saved_files: list[Path]) -> None:
	# Build beautiful stats table
	table = Table(title="Pipeline Run Summary", show_header=True, header_style="bold green")
	table.add_column("Metric", style="cyan")
	table.add_column("Value", style="magenta")

	table.add_row("Detected Language", result.language or "Unknown")
	table.add_row("Total Duration", f"{result.duration:.2f} seconds")
	table.add_row("Total Segments", str(len(result.segments)))
	
	all_words = list(result.all_words())
	if all_words:
		table.add_row("Total Words", str(len(all_words)))
		avg_conf = sum(w.probability for w in all_words) / len(all_words)
		table.add_row("Average Word Confidence", f"{avg_conf * 100:.2f}%")
	
	console.print("\n")
	console.print(table)
	
	console.print("\n[bold green]Saved Outputs:[/bold green]")
	for file_path in saved_files:
		console.print(f"  - [link=file://{file_path}]{file_path}[/link]")


def main(argv: Optional[Sequence[str]] = None) -> None:
	console = Console()
	args = parse_args(argv)

	audio_path, transcript_path = resolve_inputs(console, args)
	model, device, device_label = choose_model(console, args)
	
	console.print(f"[bold cyan]Audio:[/bold cyan] {audio_path}")
	if args.mode == "align":
		console.print(f"[bold cyan]Transcript:[/bold cyan] {transcript_path}")
	console.print(f"[bold cyan]Mode:[/bold cyan] {args.mode}")
	console.print(f"[bold cyan]Model:[/bold cyan] {args.model}")
	console.print(f"[bold cyan]Device:[/bold cyan] {device_label} ({device})")
	
	result = process_audio(
		console=console,
		model=model,
		audio_path=audio_path,
		transcript_path=transcript_path,
		args=args,
	)
	
	saved_files = export_results(console, result, audio_path, args)
	print_summary(console, result, saved_files)


if __name__ == "__main__":
	main(sys.argv[1:])
