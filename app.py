from __future__ import annotations

import tempfile
from pathlib import Path

import pypdfium2 as pdfium
import streamlit as st
from transformers import AutoModelForImageTextToText, MiniCPMV4_6Processor

MODEL_ID = "openbmb/MiniCPM-V-4.6"


@st.cache_resource(show_spinner="Loading model (first run can take a while)...")
def load_model_and_processor(model_id: str):
    processor = MiniCPMV4_6Processor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, processor


def render_pdf_pages(pdf_bytes: bytes, max_pages: int, scale: float):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf_path = Path(temp_dir) / "uploaded.pdf"
        temp_pdf_path.write_bytes(pdf_bytes)

        pdf = pdfium.PdfDocument(str(temp_pdf_path))
        try:
            page_count = len(pdf)
            pages_to_use = min(max_pages, page_count)
            images = [pdf[i].render(scale=scale).to_pil() for i in range(pages_to_use)]
        finally:
            close_method = getattr(pdf, "close", None)
            if callable(close_method):
                close_method()

    return images, page_count, pages_to_use


def summarize_pdf(model, processor, images, user_prompt: str):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"downsample_mode": "16x", "max_slice_nums": 36},
    ).to(model.device)

    generated_ids = model.generate(**inputs, downsample_mode="16x", max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main():
    st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")
    st.title("PDF Summarizer")
    st.write("Upload a PDF and get a summary using MiniCPM-V-4.6.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        max_pages = st.slider("Pages to analyze", min_value=1, max_value=10, value=3)
    with col2:
        render_scale = st.slider("Quality", min_value=1.0, max_value=3.0, value=2.0, step=0.5)

    user_prompt = st.text_area(
        "Instruction",
        value="Summarize this PDF for a non-technical reader in bullet points.",
        height=100,
    )

    summarize_clicked = st.button("Summarize", type="primary", use_container_width=True)

    if not summarize_clicked:
        return

    if uploaded_file is None:
        st.error("Please upload a PDF file first.")
        return

    if not user_prompt.strip():
        st.error("Please provide an instruction.")
        return

    with st.spinner("Reading PDF pages..."):
        images, page_count, pages_used = render_pdf_pages(
            uploaded_file.getvalue(), max_pages=max_pages, scale=render_scale
        )

    st.info(f"Detected {page_count} page(s). Using first {pages_used} page(s).")

    model, processor = load_model_and_processor(MODEL_ID)

    with st.spinner("Generating summary..."):
        summary = summarize_pdf(model, processor, images, user_prompt.strip())

    st.subheader("Summary")
    st.write(summary)


if __name__ == "__main__":
    main()
