# AI PDF Summarizer and Question Generator

AI PDF Question Generator is a Generative AI web application that automatically generates questions from uploaded PDF documents.

The system extracts text from a PDF file and uses a pre-trained language model to generate questions based on the document content.

This project demonstrates how Generative AI models can be integrated into simple applications to assist in learning, assessment preparation, and document understanding.

---

## Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- T5 Language Model
- PyPDF2
- FPDF

---

## How the System Works

1. The user uploads a PDF file through the web interface.
2. The application extracts text from the uploaded PDF.
3. The extracted text is processed using a generative AI model.
4. There will be two buttons- generate summary, generate Questions
5. The model Generate a summary of the document if you want summary of document.
6. The model generates questions based on the content of the PDF if you click the generate question button.
7. The generated questions are displayed on the interface.
8. The user can download the generated questions as a PDF.

---

## Installation

Clone the repository:

git clone https://github.com/MedhaviNasare02/CaseStudy_GenAI.git


Navigate to the project folder:

cd AI-PDF-Question-Generator


Install the required libraries:

pip install -r requirements.txt


Run the application:

python -m streamlit run app.py


Open the application in your browser:

http://localhost:8501

---

## Project Structure

CaseStudy_GenAI
│
├── app.py
├── images
├── requirements.txt
├── README.md
└── Code_Explanation.md


