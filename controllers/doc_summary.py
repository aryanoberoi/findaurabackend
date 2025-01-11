from summarizer import Summarizer
from langchain_openai import ChatOpenAI
import logging
import os
from dotenv import load_dotenv
load_dotenv()
es_cloud_id = os.getenv("ES_CLOUD_ID")
es_api_key = os.getenv("ES_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
#
# Extractive summarization using bert summerizer
# Then modify summary as per user query with LLM
# Returns summary text
#
def summarize_document(query, usersession):
    # Read the string from the file
    filename = f'{usersession}/content.txt'
    with open(filename, "r", encoding='utf8') as file:
        full_text = file.read()
    
    model = Summarizer()
    most_important_sents = model(full_text, num_sentences=60) # We specify a number of sentences
    logging.info(f'imp sents: {most_important_sents}')

    # Save extractive summary to create graph
    graphtext_filename = f'{usersession}/graphtext.txt'
    try:
        with open(graphtext_filename, "w") as graphtext_file:
            graphtext_file.write(most_important_sents)
    except IOError as e:
        logging.error(f"Error writing to {graphtext_filename}: {e}")
        return None
    
    prompt = f'''<task>
        <instruction>
        You will be given a series of sentences from a document/paper/article. Your goal is to give a summary of the document or answer specific questions about the document with respect to the query. 
        The query and sentences will be enclosed in triple backticks (```). 
        If the sentences do not provide meaningful information or context for the query, respond with "No relevant information provided."
        </instruction>
        <query>
        ```{query}```
        </query>
        <sentences>
        ```{most_important_sents}```
        </sentences>
        </task>'''

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    summary = llm.invoke(prompt)
    logging.info(f'summary result : {summary}')
    return summary.content