import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
from huggingface_hub import snapshot_download
from wtpsplit import SaT

# -------------------------------------------------------------------
# Streamlit configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Podcast Highlight Extractor",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Podcast Highlight Extractor")
st.write(
    "Upload a podcast/video to detect **questions, agreement, and disagreement** "
    "in the conversation and generate short highlight clips."
)

CATEGORIES = ["question", "agreement", "disagreement", "other"]

CLASSIFICATION_QUESTIONS = {
    "classification": {
        "type": "choice",
        "instructions": """
        Classify the conversational function of this Malayalam podcast
        transcript sentence.

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


def get_hf_token():
    """Read the Hugging Face token from Streamlit secrets or environment."""
    try:
        token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        token = ""
    return token or os.getenv("HF_TOKEN", "")


@st.cache_resource(show_spinner="Loading speech and classification models...")
def load_models(token):
    """
    Load the Indic transcription model, SaT sentence splitter,
    and Yama Indic conversational classifier once per app session.
    """
    import sys

    if not token:
        raise ValueError(
            "A Hugging Face token is required. Enter it in the sidebar "
            "or configure HF_TOKEN in Streamlit secrets."
        )

    model_name = "bodhan-ai/indic-transcribe-core"
    model_dir = snapshot_download(
        repo_id=model_name,
        token=token,
        local_dir=str(
            Path(__file__).resolve().parent
            / "models"
            / "indic-transcribe-core"
        )
    )

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from indic_transcribe import IndicTranscribe
    import thx01

    asr = IndicTranscribe.from_pretrained(model_dir)
    sentence_splitter = SaT("sat-3l-sm")
    classifier = thx01.load("LatentAILabs/yama-indic")

    return asr, sentence_splitter, classifier


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure it is available "
            "on your system PATH."
        )


def extract_audio(video_path, audio_path):
    check_ffmpeg()
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio_path),
    ]
    subprocess.run(command, check=True)


def transcribe_in_chunks(asr, audio_path, chunk_seconds=25):
    """Transcribe audio in short chunks and preserve each chunk's time range."""
    audio, sample_rate = librosa.load(
        str(audio_path), sr=16000, mono=True
    )
    if len(audio) == 0:
        raise ValueError("No audio could be read from the uploaded file.")

    chunk_samples = chunk_seconds * sample_rate
    transcript_chunks = []

    progress = st.progress(0, text="Transcribing audio...")
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    with tempfile.TemporaryDirectory() as chunk_dir:
        for index, start_sample in enumerate(
            range(0, len(audio), chunk_samples)
        ):
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]
            chunk_path = Path(chunk_dir) / f"chunk_{index:04d}.wav"
            sf.write(str(chunk_path), chunk, sample_rate)

            text, language_info = asr.transcribe(
                str(chunk_path),
                return_lid=True,
            )

            transcript_chunks.append(
                {
                    "start": round(start_sample / sample_rate, 2),
                    "end": round(end_sample / sample_rate, 2),
                    "text": str(text).strip(),
                    "language_info": str(language_info),
                }
            )
            progress.progress(
                (index + 1) / total_chunks,
                text=f"Transcribed chunk {index + 1} of {total_chunks}",
            )

    progress.empty()
    return transcript_chunks, len(audio) / sample_rate


def clean_segments(sentences, min_words=4):
    """Merge very short sentence fragments into the following sentence."""
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


def make_sentence_records(transcript_chunks, sentence_splitter):
    records = []

    for chunk in transcript_chunks:
        sentences = sentence_splitter.split(chunk["text"])
        for sentence in clean_segments(sentences, min_words=4):
            records.append(
                {
                    "chunk_start": chunk["start"],
                    "chunk_end": chunk["end"],
                    "text": sentence,
                }
            )

    return records


def classify_sentences(records, classifier, threshold):
    results = []
    progress = st.progress(0, text="Classifying transcript sentences...")

    for index, record in enumerate(records):
        try:
            result = classifier.system_one(
                record["text"], CLASSIFICATION_QUESTIONS
            )
            classification = result["answers"]["classification"]
            probabilities = classification.get("probabilities", {})

            prediction = classification.get("choice", "other")
            confidence = float(classification.get("confidence", 0.0))

            results.append(
                {
                    "prediction": prediction,
                    "confidence": confidence,
                    "question_prob": float(probabilities.get("question", 0)),
                    "agreement_prob": float(probabilities.get("agreement", 0)),
                    "disagreement_prob": float(probabilities.get("disagreement", 0)),
                    "other_prob": float(probabilities.get("other", 0)),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "prediction": "error",
                    "confidence": 0.0,
                    "question_prob": 0.0,
                    "agreement_prob": 0.0,
                    "disagreement_prob": 0.0,
                    "other_prob": 0.0,
                    "error": str(exc),
                }
            )

        progress.progress(
            (index + 1) / len(records),
            text=f"Classified sentence {index + 1} of {len(records)}",
        )

    progress.empty()

    classified = pd.DataFrame(
        [{**record, **result} for record, result in zip(records, results)]
    )

    if not classified.empty:
        classified["is_highlight"] = (
            classified["prediction"].isin(
                ["question", "agreement", "disagreement"]
            )
            & (classified["confidence"] >= threshold)
        )

    return classified


def build_clip_candidates(highlights, clip_duration, video_duration):
    candidates = []

    for _, row in highlights.iterrows():
        highlight_time = (row["chunk_start"] + row["chunk_end"]) / 2
        start = max(0.0, highlight_time - clip_duration / 2)
        end = min(video_duration, start + clip_duration)
        start = max(0.0, end - clip_duration)

        candidates.append(
            {
                "start": start,
                "end": end,
                "category": row["prediction"],
                "text": row["text"],
            }
        )

    candidates.sort(key=lambda item: item["start"])
    merged = []

    for clip in candidates:
        if not merged or clip["start"] >= merged[-1]["end"]:
            merged.append(clip.copy())
        else:
            previous = merged[-1]
            previous["end"] = max(previous["end"], clip["end"])
            previous["text"] += " " + clip["text"]
            categories = previous["category"].split(", ")
            if clip["category"] not in categories:
                previous["category"] += ", " + clip["category"]

    return merged


def export_clips(video_path, candidates, output_dir, min_duration, max_duration):
    check_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for index, clip in enumerate(candidates, start=1):
        start = clip["start"]
        duration = min(clip["end"] - start, max_duration)

        if duration < min_duration:
            continue

        output_path = output_dir / f"highlight_{len(exported) + 1:03d}.mp4"

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True)

        exported.append(
            {
                **clip,
                "duration": round(duration, 2),
                "file": output_path,
            }
        )

    return exported


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Settings")
hf_token_input = st.sidebar.text_input(
    "Hugging Face token",
    value=get_hf_token(),
    type="password",
    help="Required to download the Indic transcription model. "
         "You can also set HF_TOKEN in Streamlit secrets.",
)
confidence_threshold = st.sidebar.slider(
    "Minimum classification confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
)
clip_duration = st.sidebar.slider(
    "Target clip duration (seconds)",
    min_value=30,
    max_value=60,
    value=45,
    step=5,
)
min_clip_duration = st.sidebar.number_input(
    "Minimum exported clip duration (seconds)",
    min_value=10,
    max_value=60,
    value=30,
    step=5,
)
max_clip_duration = st.sidebar.number_input(
    "Maximum exported clip duration (seconds)",
    min_value=30,
    max_value=120,
    value=60,
    step=5,
)

uploaded_video = st.file_uploader(
    "Upload a podcast or video",
    type=["mp4", "mov", "mkv", "avi", "webm"],
)

if uploaded_video is not None:
    st.video(uploaded_video)

run_button = st.button(
    "Analyze video and create highlights",
    type="primary",
    disabled=uploaded_video is None,
)

if run_button:
    if not hf_token_input:
        st.error("Please provide a Hugging Face token in the sidebar.")
        st.stop()

    if min_clip_duration > max_clip_duration:
        st.error("Minimum clip duration cannot exceed maximum clip duration.")
        st.stop()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            video_path = work_dir / f"input{Path(uploaded_video.name).suffix}"
            audio_path = work_dir / "podcast_audio.wav"
            output_dir = work_dir / "highlight_clips"

            video_path.write_bytes(uploaded_video.getvalue())

            with st.spinner("Loading models. The first run may take some time..."):
                asr, sentence_splitter, classifier = load_models(hf_token_input)

            with st.spinner("Extracting audio from the video..."):
                extract_audio(video_path, audio_path)

            transcript_chunks, video_duration = transcribe_in_chunks(
                asr, audio_path, chunk_seconds=25
            )

            if not any(chunk["text"] for chunk in transcript_chunks):
                st.warning("The transcription model did not return any text.")
                st.stop()

            sentence_records = make_sentence_records(
                transcript_chunks, sentence_splitter
            )

            if not sentence_records:
                st.warning("No transcript sentences were detected.")
                st.stop()

            classified_df = classify_sentences(
                sentence_records, classifier, confidence_threshold
            )

            if classified_df.empty:
                st.warning("No sentences were available for classification.")
                st.stop()

            highlights_df = classified_df[
                classified_df["is_highlight"]
            ].copy()

            st.session_state["classified_df"] = classified_df
            st.session_state["highlights_df"] = highlights_df

            st.success("Transcription and classification completed.")

            st.subheader("Classification summary")
            summary = (
                classified_df["prediction"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Sentences")
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.subheader("Detected highlights")
            if highlights_df.empty:
                st.info(
                    "No question, agreement, or disagreement sentences met "
                    "the selected confidence threshold."
                )
            else:
                display_columns = [
                    "chunk_start", "chunk_end", "text", "prediction",
                    "confidence", "question_prob", "agreement_prob",
                    "disagreement_prob",
                ]
                st.dataframe(
                    highlights_df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("Transcript")
            st.dataframe(
                classified_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = classified_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download full classification CSV",
                data=csv_data,
                file_name="classified_transcript.csv",
                mime="text/csv",
            )

            candidates = build_clip_candidates(
                highlights_df, clip_duration, video_duration
            )

            if not candidates:
                st.info("No highlight clips could be created.")
            else:
                with st.spinner("Rendering highlight clips with FFmpeg..."):
                    exported = export_clips(
                        video_path,
                        candidates,
                        output_dir,
                        int(min_clip_duration),
                        int(max_clip_duration),
                    )

                if not exported:
                    st.info(
                        "No clips met the minimum duration. Try a shorter "
                        "target duration or review the detected highlights."
                    )
                else:
                    st.subheader("Generated highlight clips")
                    for index, clip in enumerate(exported, start=1):
                        st.markdown(
                            f"**Clip {index}** · {clip['category']} · "
                            f"{clip['start']:.1f}s–{clip['end']:.1f}s"
                        )
                        st.caption(clip["text"])
                        st.video(str(clip["file"]))
                        st.download_button(
                            f"Download clip {index}",
                            data=clip["file"].read_bytes(),
                            file_name=clip["file"].name,
                            mime="video/mp4",
                            key=f"download_clip_{index}",
                        )

    except Exception as exc:
        st.exception(exc)

st.divider()
st.caption(
    "Note: the notebook's clip timestamps are based on 25-second transcription "
    "chunk ranges, not word-level timestamps. Clips are therefore approximate "
    "and may start or end between conversational turns."
)
