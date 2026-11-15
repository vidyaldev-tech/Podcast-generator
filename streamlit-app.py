
import os
import sys
import shutil
import subprocess
import tempfile
import time

import streamlit as st
import librosa
import soundfile as sf
import pandas as pd
import torch

from huggingface_hub import login, snapshot_download
from wtpsplit import SaT
import thx01


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ASR_MODEL = "bodhan-ai/indic-transcribe-core"
SAT_MODEL = "sat-3l-sm"
CLASSIFIER_MODEL = "LatentAILabs/yama-indic"
# CLASSIFIER_MODEL = "LatentAILabs/laya-indic"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 25
MIN_WORDS = 4

HIGHLIGHT_LABELS = [
    "question",
    "agreement",
    "disagreement"
]

HF_TOKEN = os.environ.get("HF_TOKEN")

QUESTIONS = {
    "classification": {
        "type": "choice",
        "instructions": """
        Classify the conversational function of this
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
            """
        }
    }
}


# --------------------------------------------------
# Load models once and cache them
# --------------------------------------------------

# @st.cache_resource(show_spinner="Loading AI models...")
# def load_models():

#     if HF_TOKEN:
#         login(
#             token=HF_TOKEN,
#             add_to_git_credential=False
#         )

#     model_dir = snapshot_download(
#         repo_id=ASR_MODEL,
#         token=HF_TOKEN
#     )

#     if model_dir not in sys.path:
#         sys.path.insert(0, model_dir)

#     from indic_transcribe import IndicTranscribe

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     asr = IndicTranscribe.from_pretrained(model_dir)

#     sat = SaT(SAT_MODEL)

#     agent = thx01.load(CLASSIFIER_MODEL)

#     # Move models only if their wrappers expose .to()
#     if hasattr(asr, "to"):
#         asr.to(device)

#     if hasattr(agent, "to"):
#         agent.to(device)

#     return {
#         "asr": asr,
#         "sat": sat,
#         "agent": agent,
#         "device": device
#     }

@st.cache_resource(show_spinner=False)
def load_models():

    models = {}

    try:
        st.write("1. Checking Hugging Face authentication...")

        if HF_TOKEN:
            login(
                token=HF_TOKEN,
                add_to_git_credential=False
            )

        st.write("2. Downloading Indic Transcribe model...")

        model_dir = snapshot_download(
            repo_id=ASR_MODEL,
            token=HF_TOKEN
        )

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        st.write("3. Importing Indic Transcribe...")

        from indic_transcribe import IndicTranscribe

        device = "cuda" if torch.cuda.is_available() else "cpu"

        st.write(f"4. Loading ASR model on {device}...")

        models["asr"] = IndicTranscribe.from_pretrained(model_dir)

        st.success("ASR model loaded!")

        st.write("5. Loading sentence segmentation model...")

        models["sat"] = SaT(SAT_MODEL)

        st.success("Sentence segmentation model loaded!")

        st.write("6. Loading classification model...")

        try:
            print("Starting classifier loading...", flush=True)

            models["agent"] = thx01.load(CLASSIFIER_MODEL)

            print("Classifier loaded successfully!", flush=True)
            st.success("Classification model loaded successfully!")

        except Exception as e:
            print(f"Classifier loading failed: {type(e).__name__}: {e}", flush=True)
            st.error(f"Classifier loading failed: {type(e).__name__}: {e}")
            st.exception(e)
            raise

        models["device"] = device

        return models

    except Exception as e:
        st.error(f"Model loading failed: {type(e).__name__}: {e}")
        st.exception(e)
        raise


# --------------------------------------------------
# FFmpeg utilities
# --------------------------------------------------

def check_ffmpeg():

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is missing. Install FFmpeg first."
        )

    if not shutil.which("ffprobe"):
        raise RuntimeError(
            "FFprobe is missing. Install FFmpeg first."
        )


def get_video_duration(video_path):

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


def extract_audio(video_path, audio_path):

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        audio_path
    ]

    subprocess.run(command, check=True)


# --------------------------------------------------
# Audio chunking
# --------------------------------------------------

def split_audio_into_chunks(audio_path, work_dir):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    chunk_samples = CHUNK_SECONDS * sr

    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    chunks = []

    for index, start in enumerate(
        range(0, len(audio), chunk_samples)
    ):

        end = min(
            start + chunk_samples,
            len(audio)
        )

        chunk_path = os.path.join(
            chunks_dir,
            f"chunk_{index:04d}.wav"
        )

        sf.write(
            chunk_path,
            audio[start:end],
            sr
        )

        chunks.append({
            "path": chunk_path,
            "start": round(start / sr, 2),
            "end": round(end / sr, 2)
        })

    return chunks


# --------------------------------------------------
# Sentence cleaning
# --------------------------------------------------

def clean_segments(sentences):

    merged = []
    buffer = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if buffer:
            sentence = buffer + " " + sentence
            buffer = ""

        if len(sentence.split()) < MIN_WORDS:
            buffer = sentence
        else:
            merged.append(sentence)

    if buffer:

        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    return merged


# --------------------------------------------------
# Transcription
# --------------------------------------------------

def transcribe_chunks(models, chunks):

    transcript_chunks = []

    progress = st.progress(0)

    for index, chunk in enumerate(chunks):

        text, language_id = models["asr"].transcribe(
            chunk["path"],
            return_lid=True
        )

        transcript_chunks.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": text or ""
        })

        progress.progress(
            (index + 1) / len(chunks),
            text=f"Transcribing {index + 1}/{len(chunks)}"
        )

    progress.empty()

    return transcript_chunks


# --------------------------------------------------
# Sentence segmentation
# --------------------------------------------------

def split_sentences(models, transcript_chunks):

    sentences = []

    for chunk in transcript_chunks:

        text = chunk["text"].strip()

        if not text:
            continue

        split_result = models["sat"].split(text)

        for sentence in clean_segments(split_result):

            sentences.append({
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"],
                "text": sentence
            })

    return sentences


# --------------------------------------------------
# Classification
# --------------------------------------------------

def classify_sentences(
    models,
    sentences,
    min_confidence,
    selected_labels
):

    rows = []

    progress = st.progress(0)

    for index, item in enumerate(sentences):

        try:

            result = models["agent"].system_one(
                item["text"],
                QUESTIONS
            )

            classification = result[
                "answers"
            ]["classification"]

            probabilities = classification["probabilities"]

            row = {
                **item,
                "prediction": classification["choice"],
                "confidence": round(
                    float(classification["confidence"]), 3
                ),
                "question_prob": round(
                    float(probabilities["question"]), 3
                ),
                "agreement_prob": round(
                    float(probabilities["agreement"]), 3
                ),
                "disagreement_prob": round(
                    float(probabilities["disagreement"]), 3
                ),
                "other_prob": round(
                    float(probabilities["other"]), 3
                )
            }

        except Exception as exc:

            st.warning(
                f"Classification failed for sentence {index + 1}: {exc}"
            )

            row = {
                **item,
                "prediction": "error",
                "confidence": 0.0,
                "question_prob": 0.0,
                "agreement_prob": 0.0,
                "disagreement_prob": 0.0,
                "other_prob": 0.0
            }

        rows.append(row)

        progress.progress(
            (index + 1) / len(sentences),
            text=f"Classifying {index + 1}/{len(sentences)}"
        )

    progress.empty()

    dataframe = pd.DataFrame(rows)

    highlights = dataframe[
        dataframe["prediction"].isin(selected_labels)
        & (dataframe["confidence"] >= min_confidence)
    ].to_dict(orient="records")

    return dataframe, highlights


# --------------------------------------------------
# Highlight selection
# --------------------------------------------------

def build_candidates(
    highlights,
    clip_length,
    video_duration
):

    candidates = []

    for row in highlights:

        midpoint = (
            row["chunk_start"] + row["chunk_end"]
        ) / 2

        start = max(
            0.0,
            midpoint - clip_length / 2
        )

        end = min(
            video_duration,
            start + clip_length
        )

        if end - start < clip_length:
            start = max(0.0, end - clip_length)

        candidates.append({
            "start": start,
            "end": end,
            "category": row["prediction"],
            "text": row["text"]
        })

    candidates.sort(
        key=lambda item: item["start"]
    )

    merged = []

    for clip in candidates:

        if not merged:
            merged.append(clip)
            continue

        previous = merged[-1]

        if clip["start"] < previous["end"]:

            previous["end"] = max(
                previous["end"],
                clip["end"]
            )

            previous["text"] += " " + clip["text"]

            categories = set(
                previous["category"].split(", ")
            )

            if clip["category"] not in categories:
                previous["category"] += ", " + clip["category"]

        else:
            merged.append(clip)

    return merged


# --------------------------------------------------
# Cut video clips
# --------------------------------------------------

def cut_clips(
    video_path,
    candidates,
    output_dir,
    min_duration,
    max_duration
):

    os.makedirs(output_dir, exist_ok=True)

    saved = []

    for index, clip in enumerate(candidates, start=1):

        start = max(0.0, clip["start"])

        duration = min(
            clip["end"] - start,
            max_duration
        )

        if duration < min_duration:
            continue

        output_path = os.path.join(
            output_dir,
            f"highlight_{index:03d}.mp4"
        )

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path
        ]

        subprocess.run(command, check=True)

        saved.append({
            "path": output_path,
            "label": (
                f"#{index:03d} | {clip['category']} | "
                f"{start:.0f}s - {start + duration:.0f}s"
            ),
            "start": round(start, 1),
            "end": round(start + duration, 1),
            "category": clip["category"],
            "text": clip["text"]
        })

    return saved


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title=" Podcast Highlight Finder",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬  Podcast Highlight Finder")

st.write(
    "Upload a  podcast video to transcribe, "
    "classify and extract interesting clips."
)

with st.sidebar:

    st.header("Highlight settings")

    selected_labels = st.multiselect(
        "Categories",
        HIGHLIGHT_LABELS,
        default=HIGHLIGHT_LABELS
    )

    min_confidence = st.slider(
        "Minimum confidence",
        0.0,
        1.0,
        0.5,
        0.05
    )

    clip_length = st.slider(
        "Clip window (seconds)",
        15,
        90,
        45,
        5
    )

    min_duration = st.slider(
        "Minimum clip duration",
        5,
        60,
        30,
        5
    )

    max_duration = st.slider(
        "Maximum clip duration",
        30,
        120,
        60,
        5
    )


uploaded_video = st.file_uploader(
    "Upload podcast video",
    type=["mp4", "mov", "mkv", "avi"]
)

if uploaded_video:

    st.video(uploaded_video)

    if st.button(
        "Find Highlights",
        type="primary"
    ):
        
        if not selected_labels:
            st.error("Select at least one category.")

        elif min_duration > max_duration:
            st.error(
                "Minimum duration cannot exceed maximum duration."
            )

        else:

            work_dir = tempfile.mkdtemp(
                prefix="podcast_"
            )

            output_dir = os.path.join(
                work_dir,
                "highlights"
            )

            try:

                check_ffmpeg()

                video_path = os.path.join(
                    work_dir,
                    uploaded_video.name
                )

                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                # CPU: Prepare video and audio
                with st.spinner("Preparing video and audio..."):

                    video_duration = get_video_duration(
                        video_path
                    )

                    audio_path = os.path.join(
                        work_dir,
                        "audio.wav"
                    )

                    extract_audio(
                        video_path,
                        audio_path
                    )

                    chunks = split_audio_into_chunks(
                        audio_path,
                        work_dir
                    )

                st.success(
                    f"Video prepared. Duration: {video_duration:.1f}s. "
                    f"Audio chunks: {len(chunks)}"
                )

                # Load models
                models = load_models()

                st.info(
                    f"Processing device: {models['device']}"
                )

                # Transcription
                with st.spinner("Transcribing audio..."):

                    transcript_chunks = transcribe_chunks(
                        models,
                        chunks
                    )

                transcript_text = "\n".join(
                    item["text"]
                    for item in transcript_chunks
                )

                # Sentence splitting
                with st.spinner("Splitting transcript into sentences..."):

                    sentences = split_sentences(
                        models,
                        transcript_chunks
                    )

                if not sentences:
                    st.error("No speech sentences found.")
                    st.stop()

                # Classification
                with st.spinner("Classifying sentences..."):

                    classified_df, highlights = classify_sentences(
                        models,
                        sentences,
                        min_confidence,
                        selected_labels
                    )

                # Cut clips
                clips = []

                if highlights:

                    candidates = build_candidates(
                        highlights,
                        clip_length,
                        video_duration
                    )

                    with st.spinner("Extracting highlight clips..."):

                        clips = cut_clips(
                            video_path,
                            candidates,
                            output_dir,
                            min_duration,
                            max_duration
                        )

                # Save results in session state
                st.session_state["transcript"] = transcript_text
                st.session_state["classified"] = classified_df
                st.session_state["clips"] = clips
                st.session_state["output_dir"] = output_dir

                st.success(
                    f"Completed! Found {len(highlights)} "
                    f"highlight sentences and created {len(clips)} clips."
                )

            except Exception as exc:

                st.error(
                    f"Processing failed: {type(exc).__name__}: {exc}"
                )

                st.exception(exc)


# --------------------------------------------------
# Display results
# --------------------------------------------------

if "classified" in st.session_state:

    tab1, tab2, tab3 = st.tabs([
        "🎬 Highlight Clips",
        "📊 Classified Sentences",
        "📝 Transcript"
    ])

    with tab1:

        clips = st.session_state["clips"]

        if clips:

            for clip in clips:

                st.subheader(clip["label"])

                st.video(clip["path"])

                with open(clip["path"], "rb") as f:

                    st.download_button(
                        "Download clip",
                        data=f.read(),
                        file_name=os.path.basename(clip["path"]),
                        mime="video/mp4",
                        key=clip["path"]
                    )

                st.caption(clip["text"])

        else:
            st.info("No highlight clips were found.")

    with tab2:

        dataframe = st.session_state["classified"]

        st.subheader("Classification counts")

        st.dataframe(
            dataframe["prediction"].value_counts().reset_index(),
            use_container_width=True
        )

        st.subheader("All classified sentences")

        st.dataframe(
            dataframe,
            use_container_width=True
        )

        csv = dataframe.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Download classification CSV",
            data=csv,
            file_name="classified_sentences.csv",
            mime="text/csv"
        )

    with tab3:

        st.text_area(
            "Full transcript",
            st.session_state["transcript"],
            height=400
        )
