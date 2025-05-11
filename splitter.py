from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_document(content: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(content)
    return chunks
