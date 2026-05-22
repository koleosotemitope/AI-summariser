# AI-summariser

Image-to-text PDF summariser with a Streamlit web UI.

## Run

1. Activate your virtual environment.
2. Install dependencies:

```powershell
python -m pip install transformers torch torchvision accelerate pypdfium2 pillow streamlit
```

3. Start the UI:

```powershell
streamlit run app.py
```

4. In the browser:
- Upload a PDF.
- Choose how many pages to analyze.
- Click Summarize.

## Notes

- First run is slower because model weights are downloaded.
- You can set `HF_TOKEN` to avoid Hugging Face rate-limit warnings.
