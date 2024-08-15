import os
from werkzeug.utils import secure_filename
import logging
# file text loaders
import docx2txt
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.schema import Document


def get_text_from_doc(doc_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(doc_file.filename))
    try:
        doc_file.save(temp_file_path)
        loader = Docx2txtLoader(temp_file_path)
        data = loader.load()
        return data
    finally:
        os.remove(temp_file_path)

def get_text_from_txt(txt_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(txt_file.filename))
    try:
        txt_file.save(temp_file_path)
        loader = TextLoader(temp_file_path)
        data = loader.load()
        return data
    finally:
        os.remove(temp_file_path)

def get_text_from_pdf(pdf_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
    pdf_file.save(temp_file_path)
    
    data = []
    with pdfplumber.open(temp_file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                document = Document(
                    metadata={'source': temp_file_path, 'page': page_num},
                    page_content=text
                )
                data.append(document)
    
    logging.info(data)
    return data

def get_text_from_files(files):
    text = []  # Initialize an empty list to hold pages from all files
    for file in files:
        logging.info(f'processing file: {file.filename}')
        if file.filename.endswith(".pdf"):
            text.extend(get_text_from_pdf(file))  # Add the pages from this PDF to the list
        elif file.filename.endswith(".doc"):
            text.extend(get_text_from_doc(file))
        elif file.filename.endswith(".docx"):
            text.extend(get_text_from_doc(file))
        elif file.filename.endswith(".txt"):
            text.extend(get_text_from_txt(file))
        else:
            logging.info(f"Unsupported file type: {file.filename}")
    return text