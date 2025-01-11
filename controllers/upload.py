from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import os
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from elasticsearch import Elasticsearch
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from dotenv import load_dotenv
load_dotenv()
es_cloud_id = os.getenv("ES_CLOUD_ID")
es_api_key = os.getenv("ES_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")
elastic_search_client=Elasticsearch(cloud_id=es_cloud_id, api_key=es_api_key, timeout=300)
def elastic_store(docs, user_session):
    create_index_with_mapping(user_session)
    db = ElasticsearchStore.from_documents(
    docs,
    es_cloud_id=es_cloud_id,
        index_name=user_session,
            es_api_key=es_api_key
                )
    db.client.indices.refresh(index=user_session)
def create_index_with_mapping(index_name):
    mapping = {
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 768 
                },
                "content": {
                    "type": "text"
                },
                "keyword_content": {
                    "type": "keyword"
                }
            }
        }
    }
    # Create the index with the specified mapping
    if elastic_search_client.indices.exists(index=index_name):
        elastic_search_client.indices.delete(index=index_name)
    elastic_search_client.indices.create(index=index_name, body=mapping)
    logging.info(f"Index '{index_name}' with custom mapping created successfully.")
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
            chunk_size=4000,
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
        embeddings =OpenAIEmbeddings(model='text-embedding-3-large', api_key=openai_api_key)
        logging.info('embedding model chosen')
        vector_store = ElasticsearchStore(
    index_name=str(usersession), embedding=embeddings, es_cloud_id=es_cloud_id, es_api_key=es_api_key,vector_query_field="vector"
)
        vector_store.add_documents(text_chunks)
    except Exception as e:
        logging.info(e)
        raise

def get_new_vector_store(text_chunks, user_session):
    try:
        text_chunks=get_text_chunks(text_chunks, user_session)
        logging.info('creating vector store')
        embeddings =OpenAIEmbeddings(model='text-embedding-3-large', api_key=openai_api_key)
        logging.info('embedding model chosen')
        vector_store=ElasticsearchStore(user_session, embedding=embeddings, es_cloud_id=es_cloud_id, es_api_key=es_api_key)
        logging.info("Adding new texts to existing vector index")
        vector_store.add_documents(documents=text_chunks, embedding=embeddings)
        logging.info(f'vector store updated: {user_session}')
    except Exception as e:
        logging.info(f"error in get new: {e}")

def store_vector(raw_text, user_session):
    text_chunks = get_text_chunks(raw_text, user_session)
    logging.info('text converted to chunks')

    # Store Elastic Search index
    if not elastic_search_client.indices.exists(index=user_session):
        elastic_search_client.indices.create(index=user_session)
    logging.info(f"Index '{user_session}' created successfully.")
    # elastic_store(text_chunks, user_session)
    get_vector_store(text_chunks, user_session)
    logging.info("Chunks stored to Elastic Search")