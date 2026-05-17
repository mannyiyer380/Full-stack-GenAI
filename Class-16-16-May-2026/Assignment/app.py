"""
Multimodal GenAI web app (Assignment - Class 16).

Accepts input in every modality (text / image / audio / video) and generates
output. For EACH modality the user can pick from 2-3 models (all free-tier),
none of which were the primary model used in the live class.

Run:  streamlit run app.py
"""
import os
import tempfile
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

# .env lives at the repo root (two levels up from this Assignment folder)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# label -> (provider, model_id).  provider in {groq, gemini, gtts, opencv}
MODEL_CHOICES = {
    "Text → Text": {
        "Groq · llama-3.3-70b-versatile": ("groq", "llama-3.3-70b-versatile"),
        "Groq · llama-3.1-8b-instant": ("groq", "llama-3.1-8b-instant"),
        "Groq · openai/gpt-oss-20b": ("groq", "openai/gpt-oss-20b"),
    },
    "Image → Text": {
        "Gemini · gemini-2.5-flash-lite": ("gemini", "gemini-2.5-flash-lite"),
        "Gemini · gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
        "Groq · llama-4-scout-17b": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
    },
    "Text → Image": {
        "Gemini · gemini-2.5-flash-image": ("gemini", "gemini-2.5-flash-image"),
        "Gemini · gemini-3.1-flash-image-preview": ("gemini", "gemini-3.1-flash-image-preview"),
        "Gemini · gemini-3-pro-image-preview": ("gemini", "gemini-3-pro-image-preview"),
    },
    "Audio → Text": {
        "Groq · whisper-large-v3-turbo": ("groq", "whisper-large-v3-turbo"),
        "Groq · whisper-large-v3": ("groq", "whisper-large-v3"),
    },
    "Text → Audio": {
        "gTTS · US English (com)": ("gtts", "com"),
        "gTTS · UK English (co.uk)": ("gtts", "co.uk"),
        "gTTS · Australian (com.au)": ("gtts", "com.au"),
    },
    "Video → Text": {
        "Gemini · gemini-2.5-flash-lite": ("gemini", "gemini-2.5-flash-lite"),
        "Gemini · gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
        "Gemini · gemini-2.0-flash": ("gemini", "gemini-2.0-flash"),
    },
    "Text → Video": {
        "Gemini frames · gemini-2.5-flash-image → OpenCV": (
            "gemini", "gemini-2.5-flash-image"),
        "Gemini frames · gemini-3.1-flash-image-preview → OpenCV": (
            "gemini", "gemini-3.1-flash-image-preview"),
    },
}

st.set_page_config(page_title="Multimodal GenAI", page_icon="🧠", layout="centered")
st.title("🧠 Multimodal GenAI")
st.caption("Assignment - pick a task, pick a model, get output. Providers: Groq + Google Gemini (free tier).")

with st.sidebar:
    st.header("Status")
    st.write("Groq key:", "✅" if GROQ_API_KEY else "❌ missing")
    st.write("Google key:", "✅" if GOOGLE_API_KEY else "❌ missing")
    st.info("Gemini free tier is rate-limited. On a 429 error, wait ~30-60s and retry.")

task = st.radio("Choose a task", list(MODEL_CHOICES.keys()))
choice_label = st.selectbox("Choose a model", list(MODEL_CHOICES[task].keys()))
provider, model_id = MODEL_CHOICES[task][choice_label]


# ---------- helpers ----------
def gemini_client():
    """Fresh client per call, with internal retries disabled.

    google-genai retries 429s internally and closes its httpx client between
    attempts, which surfaces as a cryptic
    'RuntimeError: Cannot send a request, as the client has been closed'.
    attempts=1 makes the real error (usually 429 quota) surface cleanly.
    """
    from google import genai
    from google.genai import types
    return genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def show_error(e):
    """Render a human-readable message for common Gemini failures."""
    s = str(e)
    if "RESOURCE_EXHAUSTED" in s or "429" in s:
        st.error("⏳ Gemini free-tier quota hit (HTTP 429). Wait ~30-60s and "
                 "retry, or switch to a Groq model where available.")
    elif "client has been closed" in s:
        st.error("⏳ Gemini closed the connection mid-request (a masked 429 "
                 "quota error). Wait ~30-60s and retry.")
    elif "NOT_FOUND" in s or "404" in s:
        st.error(f"❌ This model isn't available to your API key.\n\n{s[:300]}")
    else:
        st.error(f"❌ {type(e).__name__}: {s[:400]}")


def gemini_generate_image(model, prompt):
    """Return a PIL.Image or None. Handles image-generation models needing response_modalities."""
    from PIL import Image
    from google.genai import types
    kw = {}
    if "image-generation" in model:
        kw["config"] = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    resp = gemini_client().models.generate_content(model=model, contents=prompt, **kw)
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return Image.open(BytesIO(part.inline_data.data))
    return None


# ---------- Text -> Text ----------
if task == "Text → Text":
    st.subheader(f"Text → Text · {model_id}")
    prompt = st.text_area("Your prompt", "Explain multimodal AI in 3 bullet points.")
    if st.button("Generate") and prompt.strip():
        from langchain_groq import ChatGroq
        with st.spinner("Thinking..."):
            llm = ChatGroq(model=model_id, api_key=GROQ_API_KEY, temperature=0.3)
            out = llm.invoke([
                ("system", "You are a concise, helpful assistant."),
                ("human", prompt),
            ]).content
        st.markdown(out)


# ---------- Image -> Text ----------
elif task == "Image → Text":
    st.subheader(f"Image → Text · {model_id}")
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    q = st.text_input("Question about the image", "Describe this image in detail.")
    if up:
        st.image(up, use_container_width=True)
    if st.button("Analyze") and up:
        mime = "image/png" if up.name.lower().endswith("png") else "image/jpeg"
        with st.spinner("Analyzing..."):
            if provider == "gemini":
                from google.genai import types
                try:
                    resp = gemini_client().models.generate_content(
                        model=model_id,
                        contents=[types.Part.from_bytes(data=up.getvalue(), mime_type=mime), q],
                    )
                    st.markdown(resp.text)
                except Exception as e:
                    show_error(e)
            else:  # groq vision
                import base64
                from langchain_groq import ChatGroq
                from langchain_core.messages import HumanMessage
                b64 = base64.b64encode(up.getvalue()).decode()
                msg = HumanMessage(content=[
                    {"type": "text", "text": q},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ])
                st.markdown(ChatGroq(model=model_id, api_key=GROQ_API_KEY).invoke([msg]).content)


# ---------- Text -> Image ----------
elif task == "Text → Image":
    st.subheader(f"Text → Image · {model_id}")
    prompt = st.text_area("Describe the image", "A cute fuzzy cat holding a red umbrella, cartoon style")
    if st.button("Generate") and prompt.strip():
        try:
            with st.spinner("Painting..."):
                img = gemini_generate_image(model_id, prompt)
            if img is not None:
                st.image(img, use_container_width=True)
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button("Download PNG", buf.getvalue(), "generated.png", "image/png")
            else:
                st.warning("Model returned no image (try another prompt or model).")
        except Exception as e:
            show_error(e)


# ---------- Audio -> Text ----------
elif task == "Audio → Text":
    st.subheader(f"Audio → Text · {model_id}")
    up = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a", "ogg", "flac"])
    if up:
        st.audio(up)
    if st.button("Transcribe") and up:
        from groq import Groq
        with st.spinner("Transcribing..."):
            client = Groq(api_key=GROQ_API_KEY)
            tr = client.audio.transcriptions.create(
                file=(up.name, up.getvalue()), model=model_id,
            )
        st.markdown(tr.text)


# ---------- Text -> Audio ----------
elif task == "Text → Audio":
    st.subheader(f"Text → Audio · gTTS ({model_id})")
    text = st.text_area("Text to speak", "Hello students, this speech was generated with gTTS.")
    lang = st.selectbox("Language", ["en", "hi", "fr", "es", "de"], index=0)
    if st.button("Synthesize") and text.strip():
        from gtts import gTTS
        with st.spinner("Synthesizing..."):
            buf = BytesIO()
            gTTS(text=text, lang=lang, tld=model_id).write_to_fp(buf)
            buf.seek(0)
        st.audio(buf, format="audio/mp3")
        st.download_button("Download MP3", buf.getvalue(), "speech.mp3", "audio/mp3")


# ---------- Video -> Text ----------
elif task == "Video → Text":
    st.subheader(f"Video → Text · {model_id}")
    up = st.file_uploader("Upload a short video", type=["mp4", "mov", "webm"])
    q = st.text_input("Question about the video", "Describe what happens in this video.")
    if up:
        st.video(up)
    if st.button("Analyze") and up:
        from google.genai import types
        try:
            with st.spinner("Watching the video..."):
                resp = gemini_client().models.generate_content(
                    model=model_id,
                    contents=[types.Part.from_bytes(data=up.getvalue(), mime_type="video/mp4"), q],
                )
            st.markdown(resp.text)
        except Exception as e:
            show_error(e)


# ---------- Text -> Video ----------
elif task == "Text → Video":
    st.subheader(f"Text → Video · {model_id} → OpenCV MP4")
    st.caption("True text-to-video (Veo/Sora) is paid. Free workaround: generate frames and stitch them.")
    story = st.text_area(
        "One scene per line",
        "A single green seed in dark soil, minimalist\n"
        "A small sprout breaking through the soil, minimalist\n"
        "A young plant with two leaves, minimalist\n"
        "A flowering plant in sunlight, minimalist",
    )
    if st.button("Generate video"):
        import cv2
        import numpy as np

        scenes = [s.strip() for s in story.splitlines() if s.strip()]
        frames = []
        prog = st.progress(0.0)
        try:
            for i, scene in enumerate(scenes):
                img = gemini_generate_image(model_id, scene)
                if img is not None:
                    pil = img.convert("RGB").resize((512, 512))
                    frames.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
                prog.progress((i + 1) / len(scenes))
        except Exception as e:
            show_error(e)

        if frames:
            out_path = os.path.join(tempfile.gettempdir(), "generated_story.mp4")
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (512, 512))
            for fr in frames:
                for _ in range(2):  # hold each frame ~2s
                    vw.write(fr)
            vw.release()
            with open(out_path, "rb") as f:
                data = f.read()
            st.video(data)
            st.download_button("Download MP4", data, "generated_story.mp4", "video/mp4")
        else:
            st.warning("No frames generated (likely Gemini rate limit). Retry shortly.")
