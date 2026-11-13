"""
Malayalam podcast highlight finder (Gradio app)

Pipeline:
  video -> ffmpeg audio -> 25s chunks -> indic-transcribe-core (ASR)
        -> SaT sentence split -> yama-indic zero-shot classification
        -> highlight clips (question / agreement / disagreement) cut with ffmpeg

Env:
  HF_TOKEN   Hugging Face token (set as a Space secret or env var)

Requirements (requirements.txt):
  gradio
  torch
  transformers
  accelerate
  sentencepiece
  soundfile
  librosa
  wtpsplit
  thx01
  pandas
  huggingface_hub

System packages (packages.txt for HF Spaces):
  ffmpeg
"""
import spaces
import os
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

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ASR_MODEL = "bodhan-ai/indic-transcribe-core"
SAT_MODEL = "sat-3l-sm"
CLASSIFIER_MODEL = "LatentAILabs/yama-indic"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 25
MIN_WORDS = 4
HIGHLIGHT_LABELS = ["question", "agreement", "disagreement"]

HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUESTIONS = {
    "classification": {
        "type": "choice",
        "instructions": """
        Classify the conversational function of this Malayalam
        podcast transcript sentence.

        Classify based on actual conversational meaning.
        A sentence mentioning a question is not necessarily a question.
        A sentence containing how or what is not always a question.
        """,
        "criteria": {
            "question": """
            The speaker directly asks another person a question,
            requests information, or seeks an answer.
            """,
            "agreement": """
            The speaker explicitly agrees with, confirms,
            or supports a previous speaker's opinion.
            """,
            "disagreement": """
            The speaker explicitly rejects, challenges,
            corrects, or opposes a previous speaker's opinion.
            """,
            "other": """
            General statements, explanations, descriptions,
            examples, suggestions, observations, and narration.
            """,
        },
    }
}

# --------------------------------------------------------------------------
# Lazy model loading (loaded once, on first request)
# --------------------------------------------------------------------------
_models = {}


def get_models():
    if _models:
        return _models

    if HF_TOKEN:
        login(token=HF_TOKEN)

    # ASR
    model_dir = snapshot_download(repo_id=ASR_MODEL, token=HF_TOKEN)
    sys.path.insert(0, model_dir)
    from indic_transcribe import IndicTranscribe  # noqa: E402

    _models["asr"] = IndicTranscribe.from_pretrained(model_dir)

    # Sentence segmenter
    from wtpsplit import SaT

    _models["sat"] = SaT(SAT_MODEL)

    # Classifier
    import thx01

    _models["agent"] = thx01.load(CLASSIFIER_MODEL)

    return _models


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        audio_path,
    ]
    subprocess.run(cmd, check=True)


def transcribe_chunks(asr, audio_path, work_dir, progress):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    chunk_samples = CHUNK_SECONDS * sr
    starts = list(range(0, len(audio), chunk_samples))

    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    transcript_chunks = []
    for i, start in enumerate(starts):
        end = min(start + chunk_samples, len(audio))
        chunk_path = os.path.join(chunks_dir, f"chunk_{i}.wav")
        sf.write(chunk_path, audio[start:end], sr)

        text, _lid = asr.transcribe(chunk_path, return_lid=True)

        transcript_chunks.append({
            "start": round(start / sr, 2),
            "end": round(end / sr, 2),
            "text": text,
        })
        progress((i + 1) / len(starts), desc=f"Transcribing chunk {i + 1}/{len(starts)}")

    return transcript_chunks


def clean_segments(sentences, min_words=MIN_WORDS):
    merged, buffer = [], ""
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


def split_sentences(sat, transcript_chunks):
    out = []
    for chunk in transcript_chunks:
        if not chunk["text"] or not chunk["text"].strip():
            continue
        for sentence in clean_segments(sat.split(chunk["text"])):
            out.append({
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"],
                "text": sentence,
            })
    return out


def classify(agent, sentences, progress):
    results = []
    for i, item in enumerate(sentences):
        try:
            result = agent.system_one(item["text"], QUESTIONS)
            c = result["answers"]["classification"]
            probs = c["probabilities"]
            results.append({
                "prediction": c["choice"],
                "confidence": round(float(c["confidence"]), 3),
                "question_prob": round(float(probs["question"]), 3),
                "agreement_prob": round(float(probs["agreement"]), 3),
                "disagreement_prob": round(float(probs["disagreement"]), 3),
                "other_prob": round(float(probs["other"]), 3),
            })
        except Exception as e:  # keep going on a bad sentence
            print("Classification error:", e)
            results.append({
                "prediction": "error", "confidence": 0.0,
                "question_prob": 0.0, "agreement_prob": 0.0,
                "disagreement_prob": 0.0, "other_prob": 0.0,
            })
        progress((i + 1) / max(len(sentences), 1),
                 desc=f"Classifying sentence {i + 1}/{len(sentences)}")
    return results


def build_candidates(highlights_df, clip_len):
    candidates = []
    for _, row in highlights_df.iterrows():
        mid = (row["chunk_start"] + row["chunk_end"]) / 2
        start = max(0.0, mid - clip_len / 2)
        candidates.append({
            "start": start,
            "end": start + clip_len,
            "category": row["prediction"],
            "text": row["text"],
        })

    candidates.sort(key=lambda c: c["start"])

    merged = []
    for clip in candidates:
        if not merged:
            merged.append(clip)
            continue
        prev = merged[-1]
        if clip["start"] < prev["end"]:
            prev["end"] = max(prev["end"], clip["end"])
            prev["text"] += " " + clip["text"]
            cats = set(prev["category"].split(", "))
            if clip["category"] not in cats:
                prev["category"] += ", " + clip["category"]
        else:
            merged.append(clip)
    return merged


def cut_clips(video_path, clips, out_dir, min_dur, max_dur):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, clip in enumerate(clips, start=1):
        start = clip["start"]
        duration = min(clip["end"] - start, max_dur)
        if duration < min_dur:
            continue

        out_path = os.path.join(out_dir, f"highlight_{i:03d}.mp4")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", video_path, "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        saved.append({
            "path": out_path,
            "label": f"#{i:03d} | {clip['category']} | {start:.0f}s-{start + duration:.0f}s",
            "start": round(start, 1),
            "end": round(start + duration, 1),
            "category": clip["category"],
            "text": clip["text"],
        })
    return saved


# --------------------------------------------------------------------------
# Gradio handler
# --------------------------------------------------------------------------
@spaces.GPU
def run_pipeline(video_path, min_conf, labels, clip_len, min_dur, max_dur,
                 progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video first.")
    if not labels:
        raise gr.Error("Select at least one category to highlight.")
    if min_dur > max_dur:
        raise gr.Error("Minimum clip length cannot be greater than the maximum.")

    work_dir = tempfile.mkdtemp(prefix="podcast_")
    try:
        progress(0, desc="Loading models (first run is slow)...")
        models = get_models()

        progress(0, desc="Extracting audio...")
        audio_path = os.path.join(work_dir, "audio.wav")
        extract_audio(video_path, audio_path)

        chunks = transcribe_chunks(models["asr"], audio_path, work_dir, progress)
        transcript_text = "\n".join(
            f"[{c['start']:.0f}s - {c['end']:.0f}s] {c['text']}" for c in chunks
        )

        sentences = split_sentences(models["sat"], chunks)
        if not sentences:
            raise gr.Error("No speech was transcribed from this video.")

        results = classify(models["agent"], sentences, progress)

        df = pd.DataFrame([{**s, **r} for s, r in zip(sentences, results)])
        counts = (
            df["prediction"].value_counts().rename_axis("label")
            .reset_index(name="count")
        )

        highlights = df[
            df["prediction"].isin(labels) & (df["confidence"] >= min_conf)
        ].copy()

        clips = []
        if not highlights.empty:
            progress(0.95, desc="Cutting highlight clips...")
            candidates = build_candidates(highlights, clip_len)
            # keep clips outside the temp work dir lifecycle
            out_dir = tempfile.mkdtemp(prefix="highlights_")
            clips = cut_clips(video_path, candidates, out_dir, min_dur, max_dur)

        clips_df = pd.DataFrame(
            [{k: v for k, v in c.items() if k != "path"} for c in clips],
            columns=["label", "start", "end", "category", "text"],
        )
        choices = [(c["label"], c["path"]) for c in clips]
        first = clips[0]["path"] if clips else None
        status = (
            f"Done: {len(sentences)} sentences, {len(highlights)} highlight sentences, "
            f"{len(clips)} clips."
        )

        return (
            status,
            transcript_text,
            df,
            counts,
            clips_df,
            gr.update(choices=choices, value=first),
            first,
            [c["path"] for c in clips] or None,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def show_clip(path):
    return path


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
with gr.Blocks(title="Malayalam Podcast Highlight Finder") as demo:
    gr.Markdown(
        "# Malayalam Podcast Highlight Finder\n"
        "Upload a podcast video. The app transcribes it, classifies each sentence "
        "(question / agreement / disagreement / other) and cuts highlight clips."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Podcast video", sources=["upload"])
            labels_in = gr.CheckboxGroup(
                choices=HIGHLIGHT_LABELS,
                value=HIGHLIGHT_LABELS,
                label="Categories to highlight",
            )
            conf_in = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                label="Minimum confidence")
            with gr.Accordion("Clip settings", open=False):
                clip_len_in = gr.Slider(15, 90, value=45, step=5,
                                        label="Clip window around highlight (s)")
                min_dur_in = gr.Slider(5, 60, value=30, step=5,
                                       label="Minimum clip length (s)")
                max_dur_in = gr.Slider(30, 120, value=60, step=5,
                                       label="Maximum clip length (s)")
            run_btn = gr.Button("Run", variant="primary")
            status_out = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Highlight clips"):
                    clip_select = gr.Dropdown(label="Select clip", choices=[])
                    clip_player = gr.Video(label="Clip preview")
                    clips_files = gr.Files(label="Download clips")
                    clips_table = gr.Dataframe(label="Clip details", wrap=True)
                with gr.Tab("Classified sentences"):
                    counts_out = gr.Dataframe(label="Label counts")
                    classified_out = gr.Dataframe(label="All sentences", wrap=True)
                with gr.Tab("Transcript"):
                    transcript_out = gr.Textbox(label="Transcript", lines=18,
                                                interactive=False)

    run_btn.click(
        run_pipeline,
        inputs=[video_in, conf_in, labels_in, clip_len_in, min_dur_in, max_dur_in],
        outputs=[status_out, transcript_out, classified_out, counts_out,
                 clips_table, clip_select, clip_player, clips_files],
    )
    clip_select.change(show_clip, inputs=clip_select, outputs=clip_player)


if __name__ == "__main__":
    demo.queue().launch()
