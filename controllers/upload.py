from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import logging
import os
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore

es_cloud_id="1964ec43953b4c07957e65d979ba0958:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGM1ZTk5M2RjYzIzZTQ5Mzg4NGQzYzY3Zjk5NzczODY5JGY4MjQ4NTI3MDE4YzQ5NmI4MTcwMmEzMGZiM2E3MmUz"
es_api_key="U0VtaTlKQUJFM3llVzNOT1VrZ2M6T1ZteHR4WUZSRWV0RkxyVFJ5TXNFQQ=="
elastic_search_client=Elasticsearch(cloud_id=es_cloud_id, api_key=es_api_key, timeout=300)

def elastic_store(docs, user_session):
    db = ElasticsearchStore.from_documents(
    docs,
    es_cloud_id="1964ec43953b4c07957e65d979ba0958:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGM1ZTk5M2RjYzIzZTQ5Mzg4NGQzYzY3Zjk5NzczODY5JGY4MjQ4NTI3MDE4YzQ5NmI4MTcwMmEzMGZiM2E3MmUz",
        index_name=user_session,
            es_api_key="U0VtaTlKQUJFM3llVzNOT1VrZ2M6T1ZteHR4WUZSRWV0RkxyVFJ5TXNFQQ=="
                )
    db.client.indices.refresh(index=user_session)

def get_text_chunks(pages, user_session):
    # Assuming `pages` is a list of Document objects, each representing a page of the document
    all_chunks = []

    # Iterate over each page and apply hierarchical chunking
    full_text = ""
    for page in pages:
        # Get hierarchical chunks
        hierarchical_chunks = get_hierarchical_chunks([page])
        all_chunks.extend(hierarchical_chunks)
        full_text += page.page_content
    
    if not os.path.exists(user_session):
        os.makedirs(user_session)
    filename = f'{user_session}/content.txt'
    with open(filename, "w") as file:
        file.write(full_text)
    
    return all_chunks


# Function to extract hierarchical chunks
def get_hierarchical_chunks(pages):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    final_chunks = []
    for page in pages:
        md_header_splits = markdown_splitter.split_text(page.page_content)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        for doc in md_header_splits:
            smaller_chunks = text_splitter.split_text(doc.page_content)
            for chunk in smaller_chunks:
                final_chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": page.metadata.get("source", ""),
                        "page": page.metadata.get("page", ""),
                        "header": " > ".join([doc.metadata.get(f"Header {i}", "") for i in range(1, 4) if f"Header {i}" in doc.metadata])
                    }
                ))

    return final_chunks

def get_vector_store(text_chunks, usersession):
    try:
        logging.info('creating vector store')
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        logging.info('embedding model chosen')
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        logging.info(f'faiss vector created')
        vector_store.save_local(usersession)
        logging.info(f'FAISS index stored to local: {usersession}')
    except Exception as e:
        logging.info(e)
        raise

def store_vector(raw_text, user_session):
    text_chunks = get_text_chunks(raw_text, user_session)
    logging.info('text converted to chunks')
    # Store to FAISS index
    get_vector_store(text_chunks, user_session)

    # Store Elastic Search index
    if elastic_search_client.indices.exists(index=user_session):
        elastic_search_client.indices.delete(index=user_session)
        logging.info(f"Index '{user_session}' deleted.")
    elastic_search_client.indices.create(index=user_session)
    logging.info(f"Index '{user_session}' created successfully.")
    elastic_store(text_chunks, user_session)
    logging.info("Chunks stored to Elastic Search")