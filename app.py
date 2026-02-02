import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
import gradio as gr
import traceback

os.makedirs("output", exist_ok=True)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

def ocr_pdf(pdf_file):
    try:
        if pdf_file is None:
            return None, None, "No PDF uploaded"

        # Gradio gives a temp file path (string)
        pdf_path = pdf_file

        out_pdf = "output/searchable.pdf"
        out_txt = "output/extracted.txt"

        c = canvas.Canvas(out_pdf)
        text_obj = c.beginText(40, 800)

        page_num = 1

        with open(out_txt, "w", encoding="utf-8") as txt:
            while True:
                try:
                    pages = convert_from_path(
                        pdf_path,
                        dpi=150,
                        first_page=page_num,
                        last_page=page_num
                    )
                except Exception:
                    break

                img = np.array(pages[0])
                processed = preprocess(img)

                text = pytesseract.image_to_string(
                    processed, config="--oem 3 --psm 6"
                )

                txt.write(f"\n--- Page {page_num} ---\n{text}")

                for line in text.split("\n"):
                    text_obj.textLine(line)
                    if text_obj.getY() < 40:
                        c.drawText(text_obj)
                        c.showPage()
                        text_obj = c.beginText(40, 800)

                del pages, img, processed
                page_num += 1

        c.drawText(text_obj)
        c.save()

        return out_pdf, out_txt, "OCR completed successfully"

    except Exception as e:
        err = traceback.format_exc()
        print(err)              # 👈 this prints REAL error in Colab
        return None, None, err  # 👈 this shows it in UI

with gr.Blocks() as demo:
    gr.Markdown("## PDF Scanner with OCR (Debug Mode)")
    pdf_input = gr.File(type="filepath", label="Upload PDF")
    run_btn = gr.Button("Run OCR")
    status = gr.Textbox(label="Status")
    out_pdf = gr.File(label="Searchable PDF")
    out_txt = gr.File(label="Extracted Text")

    run_btn.click(
        fn=ocr_pdf,
        inputs=pdf_input,
        outputs=[out_pdf, out_txt, status]
    )

demo.launch(share=True)
