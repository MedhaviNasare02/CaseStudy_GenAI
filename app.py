import streamlit as st
import PyPDF2
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fpdf import FPDF

st.title("AI PDF Question Generator")
st.write("Upload a PDF file and generate summary and questions automatically.")

@st.cache_resource
def load_question_model():
    tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    return tokenizer, model

@st.cache_resource
def load_summary_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

q_tokenizer, q_model = load_question_model()
s_tokenizer, s_model = load_summary_model()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    text = text[:1500]

    st.success("PDF uploaded successfully!")

    # ----------------------
    # FEATURE 1 - Summary
    # ----------------------

    if st.button("Generate Summary"):

        st.write("Generating summary...")

        prompt = "summarize: " + text

        input_ids = s_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids

        summary_ids = s_model.generate(
            input_ids,
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

        summary = s_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("PDF Summary")
        st.write(summary)

    # ----------------------
    # FEATURE 2 - Questions
    # ----------------------

    if st.button("Generate Questions"):

        st.write("Generating questions...")

        prompt = "answer: AI context: " + text

        input_ids = q_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids

        outputs = q_model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            num_return_sequences=5,
            early_stopping=True
        )

        questions = []
        st.subheader("Generated Questions")

        for i, output in enumerate(outputs):
            question = q_tokenizer.decode(output, skip_special_tokens=True)
            questions.append(question)
            st.write(f"{i+1}. {question}")

        # ----------------------
        # FEATURE 3 - Download PDF
        # ----------------------

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Generated Questions", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", size=12)

        for i, q in enumerate(questions):
            # FPDF doesn't support unicode so encode safely
            safe_q = q.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, txt=f"{i+1}. {safe_q}")
            pdf.ln(2)

        pdf_file = "questions.pdf"
        pdf.output(pdf_file)

        with open(pdf_file, "rb") as file:
            st.download_button(
                label="Download Questions as PDF",
                data=file,
                file_name="questions.pdf",
                mime="application/pdf"
            )