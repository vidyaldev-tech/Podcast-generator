"""
Malayalam Podcast Highlight Finder for Hugging Face Spaces (ZeroGPU)

Stages:
  1. CPU: extract audio and create 25-second chunks
  2. GPU: transcribe chunks with IndicTranscribe
  3. CPU: split transcript into sentences with SaT
  4. GPU: classify sentences with YAMA / THX-01
  5. CPU: select and cut highlight clips with FFmpeg

Space secret:
  HF_TOKEN = your Hugging Face access token
"""

# ZeroGPU requirement: import spaces before torch or CUDA-aware packages.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces

import shutil
import subprocess
import sys
import tempfile

import gradio as gr
import librosa
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import login, snapshot_download
from wtpsplit import SaT
import thx01


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ASR_MODEL = "bodhan-ai/indic-transcribe-core"
SAT_MODEL = "sat-3l-sm"
CLASSIFIER_MODEL = "LatentAILabs/yama-indic"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 25
MIN_WORDS = 4

# Each GPU stage requests up to 120 seconds.
# Reduce this if the visitor's ZeroGPU tier/quota does not allow it.
GPU_DURATION_SECONDS = 120

HIGHLIGHT_LABELS = ["question", "agreement", "disagreement"]
HF_TOKEN = os.environ.get("HF_TOKEN")

QUESTIONS = {
    "classification": {
        "type": "choice",
        "instructions": """
Classify the conversational function of this Malayalam podcast
transcript sentence. Classify based on actual conversational meaning.
A sentence mentioning a question is not necessarily a question.
A sentence containing how or what is not always a question.
""",
        "criteria": {
            "question": """
The speaker directly asks another person a question, requests
information, or seeks an answer.
""",
            "agreement": """
The speaker explicitly agrees with, confirms, or supports a
previous speaker's opinion.
""",
            "disagreement": """
The speaker explicitly rejects, challenges, corrects, or opposes
a previous speaker's opinion.
""",
            "other": """
General statements, explanations, descriptions, examples,
suggestions, observations, and narration.
""",
        },
    }
}


# ---------------------------------------------------------------------
# Load models at application startup.
#
# This follows ZeroGPU's documented pattern: models are initialized at
# module scope, not downloaded/initialized inside @spaces.GPU functions.
# `spaces` must be imported before torch.
# ---------------------------------------------------------------------

def load_models():
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)

    model_dir = snapshot_download(
        repo_id=ASR_MODEL,
        token=HF_TOKEN,
    )

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from indic_transcribe import IndicTranscribe

    asr = IndicTranscribe.from_pretrained(model_dir)
    sat = SaT(SAT_MODEL)
    agent = thx01.load(CLASSIFIER_MODEL)

    # Some model wrappers expose .to(); others manage device placement
    # internally. Only call it when the public method exists.
    if hasattr(asr, "to"):
        asr.to("cuda")
    if hasattr(agent, "to"):
        agent.to("cuda")

    return {"asr": asr, "sat": sat, "agent": agent}


try:
    MODELS = load_models()
    MODEL_LOAD_ERROR = None
except Exception as exc:
    MODELS = None
    MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    print("Model loading failed:", MODEL_LOAD_ERROR)


# ---------------------------------------------------------------------
# CPU stage 1: validate video, extract audio, split into chunks
# ---------------------------------------------------------------------

def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise gr.Error(
            "FFmpeg/FFprobe is missing. Add 'ffmpeg' to packages.txt."
        )


def get_video_duration(video_path):
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        audio_path,
    ]
    subprocess.run(command, check=True)


def split_audio_into_chunks(audio_path, work_dir):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_samples = CHUNK_SECONDS * sr

    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    chunk_files = []
    for index, start in enumerate(range(0, len(audio), chunk_samples)):
        end = min(start + chunk_samples, len(audio))
        chunk_path = os.path.join(chunks_dir, f"chunk_{index:04d}.wav")
        sf.write(chunk_path, audio[start:end], sr)

        chunk_files.append({
            "path": chunk_path,
            "start": round(start / sr, 2),
            "end": round(end / sr, 2),
        })

    return chunk_files


def prepare_video(video_path):
    if not video_path:
        raise gr.Error("Please upload a video first.")

    check_ffmpeg()

    work_dir = tempfile.mkdtemp(prefix="podcast_")
    output_dir = tempfile.mkdtemp(prefix="highlights_")

    try:
        duration = get_video_duration(video_path)
        audio_path = os.path.join(work_dir, "audio.wav")

        extract_audio(video_path, audio_path)
        chunk_files = split_audio_into_chunks(audio_path, work_dir)

        if not chunk_files:
            raise gr.Error("No audio could be extracted from this video.")

        state = {
            "video_path": video_path,
            "video_duration": duration,
            "work_dir": work_dir,
            "output_dir": output_dir,
            "chunk_files": chunk_files,
            "transcript_chunks": [],
            "sentences": [],
            "classified_rows": [],
            "highlights": [],
        }

        return state, (
            f"Audio prepared. Video duration: {duration:.1f}s. "
            f"Audio chunks: {len(chunk_files)}."
        )

    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------
# GPU stage 2: transcription
# This function is directly connected to a Gradio event below.
# ---------------------------------------------------------------------

@spaces.GPU(duration=GPU_DURATION_SECONDS)
def transcribe_stage(state, progress=gr.Progress()):
    if not state or not state.get("chunk_files"):
        raise gr.Error("Audio chunks are missing. Please run the app again.")

    if MODELS is None:
        raise gr.Error(
            "Models failed to load when the Space started:\n"
            f"{MODEL_LOAD_ERROR}"
        )

    asr = MODELS["asr"]
    transcript_chunks = []

    for index, chunk in enumerate(state["chunk_files"]):
        text, _language_id = asr.transcribe(
            chunk["path"],
            return_lid=True,
        )

        transcript_chunks.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": text or "",
        })

        progress(
            (index + 1) / max(len(state["chunk_files"]), 1),
            desc=f"Transcribing chunk {index + 1}/{len(state['chunk_files'])}",
        )

    state["transcript_chunks"] = transcript_chunks
    transcript_text = "\n".join(
        f"[{item['start']:.0f}s - {item['end']:.0f}s] {item['text']}"
        for item in transcript_chunks
    )

    return state, transcript_text, "Transcription completed."


# ---------------------------------------------------------------------
# CPU stage 3: sentence segmentation
# ---------------------------------------------------------------------

def clean_segments(sentences, min_words=MIN_WORDS):
    merged = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if buffer:
            sentence = buffer + " " + sentence
            buffer = ""

        if len(sentence.split()) < min_words:
            buffer = sentence
        else:
            merged.append(sentence)

    if buffer:
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    return merged


def split_sentences_stage(state):
    if not state or not state.get("transcript_chunks"):
        raise gr.Error("Transcript is empty. Please check the transcription.")

    sat = MODELS["sat"]
    sentences = []

    for chunk in state["transcript_chunks"]:
        text = chunk["text"].strip()
        if not text:
            continue

        for sentence in clean_segments(sat.split(text)):
            sentences.append({
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"],
                "text": sentence,
            })

    if not sentences:
        raise gr.Error("No speech sentences were found in the transcript.")

    state["sentences"] = sentences
    return state, f"Sentence splitting completed: {len(sentences)} sentences."


# ---------------------------------------------------------------------
# GPU stage 4: YAMA classification
# This function is directly connected to a Gradio event below.
# ---------------------------------------------------------------------

@spaces.GPU(duration=GPU_DURATION_SECONDS)
def classify_stage(state, min_confidence, selected_labels,
                   progress=gr.Progress()):
    if not state or not state.get("sentences"):
        raise gr.Error("No sentences are available for classification.")

    if MODELS is None:
        raise gr.Error(
            "Models failed to load when the Space started:\n"
            f"{MODEL_LOAD_ERROR}"
        )

    agent = MODELS["agent"]
    sentences = state["sentences"]
    classified_rows = []

    for index, item in enumerate(sentences):
        try:
            result = agent.system_one(item["text"], QUESTIONS)
            classification = result["answers"]["classification"]
            probabilities = classification["probabilities"]

            row = {
                **item,
                "prediction": classification["choice"],
                "confidence": round(float(classification["confidence"]), 3),
                "question_prob": round(float(probabilities["question"]), 3),
                "agreement_prob": round(float(probabilities["agreement"]), 3),
                "disagreement_prob": round(float(probabilities["disagreement"]), 3),
                "other_prob": round(float(probabilities["other"]), 3),
            }

        except Exception as exc:
            print("Classification error:", repr(exc))
            row = {
                **item,
                "prediction": "error",
                "confidence": 0.0,
                "question_prob": 0.0,
                "agreement_prob": 0.0,
                "disagreement_prob": 0.0,
                "other_prob": 0.0,
            }

        classified_rows.append(row)

        progress(
            (index + 1) / max(len(sentences), 1),
            desc=f"Classifying sentence {index + 1}/{len(sentences)}",
        )

    dataframe = pd.DataFrame(classified_rows)
    if dataframe.empty:
        counts = pd.DataFrame(columns=["label", "count"])
    else:
        counts = (
            dataframe["prediction"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
        )

    highlights = dataframe[
        dataframe["prediction"].isin(selected_labels)
        & (dataframe["confidence"] >= min_confidence)
    ].to_dict(orient="records")

    state["classified_rows"] = classified_rows
    state["highlights"] = highlights

    return (
        state,
        dataframe,
        counts,
        f"Classification completed: {len(highlights)} highlight sentences found.",
    )


# ---------------------------------------------------------------------
# CPU stage 5: prepare candidates and cut clips
# ---------------------------------------------------------------------

def build_candidates(highlights, clip_len, video_duration):
    candidates = []

    for row in highlights:
        midpoint = (row["chunk_start"] + row["chunk_end"]) / 2
        start = max(0.0, midpoint - clip_len / 2)
        end = min(video_duration, start + clip_len)

        if end - start < clip_len:
            start = max(0.0, end - clip_len)

        candidates.append({
            "start": start,
            "end": end,
            "category": row["prediction"],
            "text": row["text"],
        })

    candidates.sort(key=lambda item: item["start"])

    merged = []
    for clip in candidates:
        if not merged:
            merged.append(clip)
            continue

        previous = merged[-1]
        if clip["start"] < previous["end"]:
            previous["end"] = max(previous["end"], clip["end"])
            previous["text"] += " " + clip["text"]

            categories = set(previous["category"].split(", "))
            if clip["category"] not in categories:
                previous["category"] += ", " + clip["category"]
        else:
            merged.append(clip)

    return merged


def cut_clips(video_path, clips, output_dir, min_duration, max_duration):
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for index, clip in enumerate(clips, start=1):
        start = max(0.0, clip["start"])
        duration = min(clip["end"] - start, max_duration)

        if duration < min_duration:
            continue

        output_path = os.path.join(
            output_dir, f"highlight_{index:03d}.mp4"
        )

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(command, check=True)

        saved.append({
            "path": output_path,
            "label": (
                f"#{index:03d} | {clip['category']} | "
                f"{start:.0f}s-{start + duration:.0f}s"
            ),
            "start": round(start, 1),
            "end": round(start + duration, 1),
            "category": clip["category"],
            "text": clip["text"],
        })

    return saved


def finish_pipeline(state, clip_len, min_duration, max_duration):
    if min_duration > max_duration:
        raise gr.Error(
            "Minimum clip length cannot be greater than maximum clip length."
        )

    if not state:
        raise gr.Error("Pipeline state is missing. Please run the app again.")

    try:
        highlights = state.get("highlights", [])
        clips = []

        if highlights:
            candidates = build_candidates(
                highlights,
                clip_len,
                state["video_duration"],
            )
            clips = cut_clips(
                state["video_path"],
                candidates,
                state["output_dir"],
                min_duration,
                max_duration,
            )

        clips_dataframe = pd.DataFrame(
            [
                {key: value for key, value in clip.items() if key != "path"}
                for clip in clips
            ],
            columns=["label", "start", "end", "category", "text"],
        )

        choices = [(clip["label"], clip["path"]) for clip in clips]
        first_clip = clips[0]["path"] if clips else None

        status = (
            f"Done: {len(state.get('sentences', []))} sentences, "
            f"{len(highlights)} highlight sentences, {len(clips)} clips."
        )

        return (
            clips_dataframe,
            gr.update(choices=choices, value=first_clip),
            first_clip,
            [clip["path"] for clip in clips] or None,
            status,
        )

    except Exception as exc:
        print("Clip extraction error:", repr(exc))
        raise gr.Error(f"Clip extraction failed: {type(exc).__name__}: {exc}")


def show_clip(path):
    return path


# ---------------------------------------------------------------------
# Gradio UI and chained processing stages
# ---------------------------------------------------------------------

with gr.Blocks(title="Malayalam Podcast Highlight Finder") as demo:
    gr.Markdown(
        "# Malayalam Podcast Highlight Finder\n"
        "Upload a Malayalam podcast video. The app transcribes it, "
        "classifies sentences (question / agreement / disagreement / other), "
        "and extracts highlight clips."
    )

    pipeline_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Podcast video",
                sources=["upload"],
                type="filepath",
            )

            labels_input = gr.CheckboxGroup(
                choices=HIGHLIGHT_LABELS,
                value=HIGHLIGHT_LABELS,
                label="Categories to highlight",
            )

            confidence_input = gr.Slider(
                0.0, 1.0,
                value=0.5,
                step=0.05,
                label="Minimum confidence",
            )

            with gr.Accordion("Clip settings", open=False):
                clip_length_input = gr.Slider(
                    15, 90,
                    value=45,
                    step=5,
                    label="Clip window around highlight (seconds)",
                )
                min_duration_input = gr.Slider(
                    5, 60,
                    value=30,
                    step=5,
                    label="Minimum clip length (seconds)",
                )
                max_duration_input = gr.Slider(
                    30, 120,
                    value=60,
                    step=5,
                    label="Maximum clip length (seconds)",
                )

            run_button = gr.Button("Run", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Highlight clips"):
                    clip_select = gr.Dropdown(
                        label="Select clip",
                        choices=[],
                    )
                    clip_player = gr.Video(label="Clip preview")
                    clips_files = gr.Files(label="Download clips")
                    clips_table = gr.Dataframe(
                        label="Clip details",
                        wrap=True,
                    )

                with gr.Tab("Classified sentences"):
                    counts_output = gr.Dataframe(label="Label counts")
                    classified_output = gr.Dataframe(
                        label="All sentences",
                        wrap=True,
                    )

                with gr.Tab("Transcript"):
                    transcript_output = gr.Textbox(
                        label="Transcript",
                        lines=18,
                        interactive=False,
                    )

    # CPU: extract audio and split into chunks.
    prepare_event = run_button.click(
        fn=prepare_video,
        inputs=[video_input],
        outputs=[pipeline_state, status_output],
    )

    # GPU: transcribe. The decorated function is directly registered
    # as a Gradio event handler, as required by ZeroGPU.
    transcription_event = prepare_event.then(
        fn=transcribe_stage,
        inputs=[pipeline_state],
        outputs=[pipeline_state, transcript_output, status_output],
    )

    # CPU: sentence segmentation.
    sentence_event = transcription_event.then(
        fn=split_sentences_stage,
        inputs=[pipeline_state],
        outputs=[pipeline_state, status_output],
    )

    # GPU: classify sentences.
    classification_event = sentence_event.then(
        fn=classify_stage,
        inputs=[pipeline_state, confidence_input, labels_input],
        outputs=[
            pipeline_state,
            classified_output,
            counts_output,
            status_output,
        ],
    )

    # CPU: cut clips and prepare outputs.
    classification_event.then(
        fn=finish_pipeline,
        inputs=[
            pipeline_state,
            clip_length_input,
            min_duration_input,
            max_duration_input,
        ],
        outputs=[
            clips_table,
            clip_select,
            clip_player,
            clips_files,
            status_output,
        ],
    )

    clip_select.change(
        fn=show_clip,
        inputs=clip_select,
        outputs=clip_player,
    )


if __name__ == "__main__":
    demo.queue().launch()
