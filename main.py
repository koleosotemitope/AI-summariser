from pathlib import Path

import pypdfium2 as pdfium
from transformers import AutoModelForImageTextToText, MiniCPMV4_6Processor

model_id = "openbmb/MiniCPM-V-4.6"

# Change this to whichever PDF you want to analyze from Downloads.
pdf_path = Path.home() / "Downloads" / "OPCS-4.11 Volume 2 Alphabetical Index.pdf"
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

pdf = pdfium.PdfDocument(str(pdf_path))
page_image = pdf[0].render(scale=2.0).to_pil()

processor = MiniCPMV4_6Processor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": page_image},
            {
                "type": "text",
                "text": "Summarize the key information visible on this PDF page.",
            },
        ],
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
print(output_text[0])
