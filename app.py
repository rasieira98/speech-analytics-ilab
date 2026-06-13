import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


st.set_page_config(
    page_title="Transcriptor de Audio",
    page_icon="🎤",
    layout="wide"
)


@st.cache_resource
def load_model():
    return WhisperModel(
        "base",      # tiny, base, small, medium, large-v3
        device="cpu",  # cuda si tienes GPU NVIDIA
        compute_type="int8"
    )


model = load_model()

st.title("🎤 Transcripción de audio con Faster-Whisper")

uploaded_file = st.file_uploader(
    "Sube un archivo de audio",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)


def transcribe(uploaded_file):

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())
        audio_path = tmp.name

    segments, info = model.transcribe(
        audio_path,
        language="es",
        vad_filter=True,
        beam_size=5
    )

    text = []

    for segment in segments:
        text.append(segment.text)

    return " ".join(text)


if uploaded_file:

    st.audio(uploaded_file)

    if st.button("Transcribir"):

        with st.spinner("Transcribiendo..."):

            transcript = transcribe(uploaded_file)

        st.success("Transcripción completada")

        st.subheader("Resultado")

        st.text_area(
            "Texto transcrito",
            transcript,
            height=400
        )

        st.download_button(
            "Descargar TXT",
            transcript,
            file_name="transcripcion.txt",
            mime="text/plain"
        )
