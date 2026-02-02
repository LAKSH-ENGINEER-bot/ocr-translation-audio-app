
import os, cv2, numpy as np, pytesseract, traceback, shutil
from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path
from reportlab.pdfgen import canvas
import gradio as gr
import fitz
from deep_translator import GoogleTranslator
from docx import Document
import easyocr
from gtts import gTTS
from PIL import Image

# ---------------- SETUP ----------------
os.makedirs("output/images", exist_ok=True)
reader = easyocr.Reader(['en'])

VOICE_MAP = {
    "English (US)": ("en", "com"),
    "English (UK)": ("en", "co.uk"),
    "English (India)": ("en", "co.in"),
    "Hindi (India)": ("hi", "co.in"),
    "French": ("fr", "fr"),
    "German": ("de", "de"),
    "Spanish": ("es", "es"),
}

# ---------------- PREPROCESS ----------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

# ---------------- IMAGE EXTRACTION ----------------
def extract_images_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    count = 0
    for i in range(len(doc)):
        page = doc[i]
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            path = f"output/images/page{i+1}_{count}.{base['ext']}"
            with open(path, "wb") as f:
                f.write(base["image"])
            count += 1
    return shutil.make_archive("output/images", "zip", "output/images") if count else None

# ---------------- TRANSLATION ----------------
def translate_text(text, lang):
    translator = GoogleTranslator(source="auto", target=lang)
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    return "".join(translator.translate(c) for c in chunks)

# ---------------- TEXT TO SPEECH (LIVE) ----------------
def text_to_speech(text, voice, speed):
    lang, tld = VOICE_MAP[voice]
    slow = True if speed == "Slow" else False
    audio_path = "output/audio.mp3"
    gTTS(text=text, lang=lang, tld=tld, slow=slow).save(audio_path)
    return audio_path

# ---------------- OCR HANDLER ----------------
def extract_text(file_path):
    ext = file_path.lower().split(".")[-1]

    # PDF (safe page count)
    if ext == "pdf":
        total_pages = pdfinfo_from_path(file_path)["Pages"]
        text = ""
        for page in range(1, total_pages + 1):
            pages = convert_from_path(file_path, dpi=150, first_page=page, last_page=page)
            img = preprocess(np.array(pages[0]))
            text += pytesseract.image_to_string(img)
        return text, extract_images_pdf(file_path)

    # IMAGE (PNG safe + handwritten)
    if ext in ["png","jpg","jpeg","bmp","tiff"]:
        img_pil = Image.open(file_path).convert("RGB")
        img_np = np.array(img_pil)
        printed = pytesseract.image_to_string(preprocess(img_np))
        handwritten = "\n".join(reader.readtext(img_np, detail=0))
        return printed + "\n" + handwritten, None

    # DOCX
    if ext == "docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs), None

    # TXT
    if ext == "txt":
        return open(file_path, "r", encoding="utf-8").read(), None

    raise ValueError("Unsupported file")

# ---------------- MAIN PIPELINE ----------------
def process(file, translate_lang, voice, speed):
    try:
        if file is None:
            return None, None, "", None, None, None, "No file uploaded"

        text, img_zip = extract_text(file)

        translated = translate_text(text, translate_lang)

        # PDFs (optional downloads)
        searchable_pdf = "output/searchable.pdf"
        translated_pdf = "output/translated_searchable.pdf"

        for path, content in [(searchable_pdf, text), (translated_pdf, translated)]:
            c = canvas.Canvas(path)
            t = c.beginText(40,800)
            for line in content.split("\n"):
                t.textLine(line)
                if t.getY() < 40:
                    c.drawText(t); c.showPage(); t = c.beginText(40,800)
            c.drawText(t); c.save()

        audio_path = text_to_speech(translated, voice, speed)

        return (
            searchable_pdf,
            translated_pdf,
            translated,        # LIVE TEXT
            audio_path,        # LIVE AUDIO
            img_zip,
            "Completed successfully"
        )

    except Exception:
        err = traceback.format_exc()
        print(err)
        return None, None, "", None, None, err

# ---------------- GRADIO UI (LIVE) ----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔴 Live OCR + Translation + Audio System")

    file = gr.File(
        label="Upload PDF / Image / DOCX / TXT",
        file_types=[".pdf",".png",".jpg",".jpeg",".bmp",".tiff",".docx",".txt"],
        type="filepath"
    )

    translate_lang = gr.Dropdown(
        ["hi","fr","es","de","ta","en"],
        value="hi",
        label="Translate To"
    )

    voice = gr.Dropdown(
        list(VOICE_MAP.keys()),
        value="English (India)",
        label="Voice / Accent"
    )

    speed = gr.Radio(
        ["Normal", "Slow"],
        value="Normal",
        label="Speech Speed"
    )

    run = gr.Button("Run")

    status = gr.Textbox(label="Status")

    out_pdf = gr.File(label="Searchable PDF (Download)")
    out_tr_pdf = gr.File(label="Translated PDF (Download)")

    live_text = gr.Textbox(
        label="Live Translated Text",
        lines=12
    )

    live_audio = gr.Audio(
        label="Live Audio Playback",
        type="filepath"
    )

    images = gr.File(label="Extracted Images (ZIP)")

    run.click(
        fn=process,
        inputs=[file, translate_lang, voice, speed],
        outputs=[out_pdf, out_tr_pdf, live_text, live_audio, images, status]
    )

demo.launch(inline=True, share=True)
