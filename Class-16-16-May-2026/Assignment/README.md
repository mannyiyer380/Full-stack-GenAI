# Assignment - Multimodal GenAI (Class 16)

Uses models **different from the live class** (`access-diff-llm-mmllm.ipynb`)
for every modality, plus a Streamlit web app.

## Files
- `multimodal-new-models.ipynb` - one section per modality, runnable end to end.
- `app.py` - Streamlit web app: input in any modality → generated output.
- `requirements.txt` - extra packages on top of the repo's root `requirements.txt`.

## Models (selectable per modality in the app)
The notebook uses the **first** model of each row; the Streamlit app lets the
user pick any of these (all free-tier):

| Modality | Selectable models |
|---|---|
| Text → Text | groq `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-20b` |
| Image → Text | gemini `gemini-2.5-flash-lite`, `gemini-2.5-flash`, groq `llama-4-scout-17b` |
| Text → Image | gemini `gemini-2.5-flash-image`, `gemini-2.0-flash-preview-image-generation` |
| Audio → Text | groq `whisper-large-v3-turbo`, `whisper-large-v3` |
| Text → Audio | gTTS accents: US (`com`), UK (`co.uk`), AU (`com.au`) |
| Video → Text | gemini `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.0-flash` |
| Text → Video | gemini image-model frames (`gemini-2.5-flash-image` / `2.0-flash-preview-image-generation`) → OpenCV MP4 |

True text-to-video (Veo/Sora) is paid, so Text → Video generates image frames
and stitches them into an MP4 with OpenCV.

## Setup
API keys are read from the repo-root `.env` (`GROQ_API_KEY`, `GOOGLE_API_KEY`).

```bash
# from the repo root, using the project venv
./env/bin/python -m pip install -r Class-16-16-May-2026/Assignment/requirements.txt
# (this repo's venv has no pip; packages were installed with:)
#   uv pip install --python ./env/bin/python streamlit gtts opencv-python-headless
```

## Run
```bash
cd Class-16-16-May-2026/Assignment
./run.sh                         # uses the project venv automatically
# or explicitly:
../../env/bin/python -m streamlit run app.py
```

> **Important:** do NOT run a bare `streamlit run app.py`. A global/conda
> `streamlit` (e.g. `~/miniconda3/bin/streamlit`) uses a different Python that
> does not have `langchain_groq` etc., which causes
> `ModuleNotFoundError: No module named 'langchain_groq'`. Always launch with
> the project venv via `./run.sh`.

Open the notebook with the project's Python 3.11 kernel and run cells top to bottom.

## Note on Gemini quota
The Gemini free tier is rate-limited. A `429 RESOURCE_EXHAUSTED` just means
wait ~30-60s and retry. The Groq paths (Text→Text, Audio→Text) are unaffected
and were verified working.
