import streamlit as st
import os
import base64
from io import BytesIO

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None

from langchain_core.messages import HumanMessage, SystemMessage

def encode_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def build_multimodal_message(input_modality, prompt_text, uploaded_file):
    if input_modality == "Text":
        return HumanMessage(content=prompt_text)

    # Image
    if input_modality == "Image" and uploaded_file is not None:
        raw = uploaded_file.read()
        mime = uploaded_file.type or "image/png"
        b64 = encode_bytes_to_base64(raw)
        content = [
            {"type": "text", "text": prompt_text or "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
        return HumanMessage(content=content)

    # Audio
    if input_modality == "Audio" and uploaded_file is not None:
        raw = uploaded_file.read()
        mime = uploaded_file.type or "audio/mpeg"
        b64 = encode_bytes_to_base64(raw)
        content = [
            {"type": "text", "text": prompt_text or "Transcribe this audio."},
            {"type": "media", "mime_type": mime, "data": b64},
        ]
        return HumanMessage(content=content)

    return HumanMessage(content=prompt_text or "")

def get_llm(provider, model_name, temperature=0.2):
    if provider == "OpenAI":
        if ChatOpenAI is None:
            raise RuntimeError("langchain_openai.ChatOpenAI not available")
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "Anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain_anthropic.ChatAnthropic not available")
        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError("Unknown provider")

def main():
    st.title("Multimodal LangChain demo — Text / Image / Audio")

    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic"])
    model_name = st.sidebar.text_input("Model", "gpt-4o-mini" if provider == "OpenAI" else "claude-3.5-mini")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

    st.header("Input")
    input_modality = st.radio("Input modality", ["Text", "Image", "Audio"]) 
    prompt_text = None
    uploaded_file = None

    if input_modality == "Text":
        prompt_text = st.text_area("Prompt", value="Write a short summary of LangChain.")
    elif input_modality == "Image":
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        prompt_text = st.text_input("Prompt for image (optional)")
    elif input_modality == "Audio":
        uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])
        prompt_text = st.text_input("Prompt for audio (optional)")

    st.header("Desired output")
    output_modality = st.selectbox("Output", ["Text", "Image", "Audio"])

    if st.button("Generate"):
        try:
            llm = get_llm(provider, model_name, temperature=temperature)
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            return

        message = build_multimodal_message(input_modality, prompt_text, uploaded_file)

        # Image-generation path (uses bind_tools style shown in the notebook)
        if output_modality == "Image":
            try:
                kwargs = {"type": "image_generation", "quality": "low"}
                if hasattr(llm, "bind_tools"):
                    llm_with_tools = llm.bind_tools([kwargs])
                    ai_message = llm_with_tools.invoke(prompt_text or "Generate an image")
                else:
                    ai_message = llm.invoke([message])

                # Try to extract image
                if hasattr(ai_message, "content_blocks"):
                    img = next((b for b in ai_message.content_blocks if b.get("type") == "image"), None)
                    if img and img.get("base64"):
                        st.image(base64.b64decode(img["base64"]))
                        st.success("Image displayed from model response")
                        return
                st.write(ai_message.content if hasattr(ai_message, "content") else ai_message)
            except Exception as e:
                st.error(f"Image generation failed: {e}")
            return

        # Default: invoke model with the prepared message
        try:
            response = llm.invoke([message])
        except Exception as e:
            st.error(f"Model invocation failed: {e}")
            return

        # Show text output
        text_output = None
        if hasattr(response, "content"):
            text_output = response.content
        elif isinstance(response, str):
            text_output = response
        else:
            text_output = str(response)

        if output_modality == "Text":
            st.subheader("Text output")
            st.write(text_output)

        elif output_modality == "Audio":
            # Try TTS fallback using gTTS if installed
            st.subheader("Audio output (TTS)")
            st.write(text_output)
            try:
                from gtts import gTTS
                tts = gTTS(text=text_output, lang="en")
                buf = BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                st.audio(buf.read(), format="audio/mp3")
            except Exception:
                st.info("Install gTTS (`pip install gTTS`) to enable text-to-speech audio playback.")

    st.markdown("---")
    st.caption("Set your API keys in environment variables `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` before running.")

if __name__ == "__main__":
    main()
