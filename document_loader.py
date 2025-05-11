import os
from typing import List, Dict
from docx import Document
import pypdf


def load_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT: {file_path} - {e}")
        return ""

def load_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX: {file_path} - {e}")
        return ""

def load_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return '\n'.join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF: {file_path} - {e}")
        return ""

def load_documents(folder_path: str) -> List[Dict]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.txt'):
            content = load_txt(file_path)
        elif filename.endswith('.docx'):
            content = load_docx(file_path)
        elif filename.endswith('.pdf'):
            content = load_pdf(file_path)
        else:
            continue  # Unsupported format
        
        documents.append({
            "content": content,
            "metadata": {"filename": filename}
        })
    return documents
