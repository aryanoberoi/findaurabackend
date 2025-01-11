from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import asyncio
from PyPDF2 import PdfReader
from docx import Document
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
#import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama 
from langchain_community.embeddings import OllamaEmbeddings
# from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain_experimental.graph_transformers import LLMGraphTransformer
from graph import graph_from_document
import time
import logging
from pymongo import MongoClient
from langchain_community.embeddings import OpenAIEmbeddings
from datetime import date, timedelta, datetime
from werkzeug.utils import secure_filename
from langchain.chains import RetrievalQAWithSourcesChain
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
mongo_url = "mongodb+srv://user2:user2@cluster0.sfiyids.mongodb.net/"
client = MongoClient(mongo_url)
db = client.test
collection=db.users
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = 'supersecretkey'

# Function to handle asyncio event loop
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
get_or_create_eventloop()

def get_text_from_doc(doc_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(doc_file.filename))
    try:
        doc_file.save(temp_file_path)
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(temp_file_path)
        data = str(loader.load())
        return data
    finally:
        os.remove(temp_file_path)

def get_text_from_docx(docx_file):
    # Save the uploaded file to a temporary location to be processed by docx2txt
    temp_file_path = "temp.docx"
    with open(temp_file_path, "wb") as f:
        f.write(docx_file.read())
    return docx2txt.process(temp_file_path)

def get_text_from_txt(txt_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(txt_file.filename))
    try:
        txt_file.save(temp_file_path)
        from langchain_community.document_loaders.text import TextLoader
        loader = TextLoader(temp_file_path)
        data = str(loader.load())
        return data
    finally:
        os.remove(temp_file_path)

def get_text_from_pdf(pdf_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
    pdf_file.save(temp_file_path)
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    data = str(loader.load())
    return data

def get_text_from_files(files):
    text = ""
    for file in files:
        logging.info(f'processing file: {file.filename}')
        if file.filename.endswith(".pdf"):
            text += get_text_from_pdf(file)
        elif file.filename.endswith(".doc"):
            text += get_text_from_doc(file)
        elif file.filename.endswith(".docx"):
            text += get_text_from_docx(file)
        elif file.filename.endswith(".txt"):
            text += get_text_from_txt(file)
        else:
            return f"Unsupported file type: {file.filename}"
    return text

def get_text_from_urls(urls):
    texts = ""
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            texts += ' '.join(p.get_text() for p in soup.find_all('p'))
        except Exception as e:
            texts += "Error fetching the URL: {e}"
    return texts

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
# take foldername input
def get_vector_store(text_chunks, usersession):
    try:
        logging.info('creating vector store')
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        logging.info('embedding model chosed')
        #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        logging.info(f'storing to local {usersession}')
        vector_store.save_local(usersession)
        logging.info('FAISS index stored to local')
    except Exception as e:
        logging.info(e)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible but only on the provided context.  Review the chat history carefully to provide all necessary details and avoid incorrect information. Treat synonyms or similar words as equivalent within the context. For example, if a question refers to "modules" or "units" instead of "chapters" or "doc" instead of "document" consider them the same. 
    If the question is not related to the provided context, simply respond that the question is out of context and instead provide a summary of the document and example questions that the user can ask.
    Do not make up an answer if the provided question is not within the context. Instead, provide example questions that the user can ask and summary of the document.
    Do not repeat facts in the answer if you have already stated them. 
    If the question is short, like it is asking for the dates, names or requires a very short answer then keep the response short and to the point. 
    If the question asks for a particular keyword that is in context, state information related to that keyword. 
    Example: Question : When was bill gates born?
    Answer: According to the documents you have uploaded, bill gates was born in 1989. 
    However, if the question requires a detailed answer, then consider all possibiliies related to the question and try to answer in Bullet points and clear and concise paragraphs. 
    If asked to summarise the document, try to provide a basic summary of the entire context and cover it in bullet points but keep the answer concise and not too long. 
    If the question mention 'what are the contents of the file' or asks about the uploaded document or anything similar, just cover the entire document as a summary.
    VERY IMPORTANT State the source in the end along with the answer but dont state the source if the question is out of context.
    If the source is like : temp/abc.docx, then just mention the file name like abc.docx.
    In the beginning of the anser, always mention the exact line in quotation marks that is being referred to in the answer and enclose it within **bold** tags.
    Example : "According to the document in the line "The company was started in 1990", to answer your question(state the query in a shorter and concise manner), the company was founded in 1990."
    Highlight Key Points: Enclose each identified key point within `**bold**` tags.
   Highlight Keywords: Enclose each identified keyword within `*italic*` tags.
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    #model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,top_k=20,top_p=0.5)
    #temperature of 0.3 balances between creativity and answering based on context. Since it is less than 0.5, it will stick more to context but also adds a bit of creative freedom.
    #top-p : the model will onsider a broader range of possible next words, balancing relevance with some level of novelty
    #top-k : limits the number of tokens considered to top k most probable ones 
    model = ChatOllama(temperature=0.3, model="llama3", top_p=0.5, top_k=10)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, usersession):
    start_time = time.time()
    # vector-search using locally stored FAISS index
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings=OllamaEmbeddings(model="mxbai-embed-large")
    new_db = FAISS.load_local(usersession, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=7, fetch_k=30)
    logging.info('---- %s seconds to do similarity search ---' % (time.time() - start_time))

    # resposne from LLM with RAG approach
    chain = get_conversational_chain()
    input_data = {
        'input_documents': docs,
        'question': user_question,
    }
    response = chain.invoke(input=input_data)
    logging.info('--- %s seconds to get response from llm ---' % (time.time() - start_time))
    return response

def authenticate(token):
    data = jwt.decode(token, 'secret', algorithms=["HS256"])
    current_user = data['email']
    logging.info(f'user email: {current_user}')
    return current_user

@app.route('/upload', methods=['POST', 'OPTIONS'])
def index():
    start_time = time.time()
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        user_session = authenticate(token)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    logging.info('--- %s seconds to authenticate req ---' % (time.time() - start_time))

    files = request.files.getlist("files")
    logging.info(f'request is \n {request.files}')
    url_input = [] 
    i = 0
    while True:
        temp_url = request.form.get(f'urls[{i}]')
        if temp_url:
            url_input.append(temp_url)
        else:
            break
        i += 1
 
    # Process files
    raw_text = ""
    if files and files[0].filename != '':
        valid_files = all(f.filename.endswith(('.pdf', '.doc', '.docx', '.txt')) for f in files)
        if valid_files:
            raw_text += get_text_from_files(files)
            message = "Files successfully uploaded."
        else:
            message = "Please upload files in PDF, DOC, DOCX, or TXT format."

    # Process URL
    if url_input:
        url_text = get_text_from_urls(url_input)
        raw_text += " " + url_text

    logging.info('--- %s seconds to extract data from file ---' % (time.time() - start_time))
    if raw_text:
        logging.info("here")
        logging.info('converting into chunks')
        text_chunks = get_text_chunks(raw_text)
        logging.info('text converted to chunks')
        get_vector_store(text_chunks, user_session)
    
    logging.info('--- %s seconds to create FAISS index ---' % (time.time() - start_time))
    
    return jsonify({"status": "ok", "message": "request processed successfully!"}), 200

@app.route('/graph', methods=['POST'])
def get_graph():
    start_time = time.time()
    # Authenticate request and extract FAISS directory from token
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        session_name = authenticate(token)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    logging.info(f'Authenticated graph request in {time.time() - start_time} seconds')

    try:
        files = request.files.getlist("files")
        url_input = [] 
        i = 0
        while True:
            temp_url = request.form.get(f'urls[{i}]')
            if temp_url:
                url_input.append(temp_url)
            else:
                break
            i += 1
    except Exception as e:
        logging.info(f'Error while extracting request: {e}')
    
    logging.info(f'Extracted files and urls from request: {len(files)}, {len(url_input)}')

    html_content = graph_from_document(files, url_input)
    logging.info('--- %s seconds to get knowledge graph ---' % (time.time() - start_time))
    
    return Response(html_content, mimetype='text/html')

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return {"message": "app is up and running"}

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()
    data = request.get_json()
    #logging.info(f'ask request: {data}')

    # Authenticate request and extract FAISS directory from token
    try:
        token = data.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        session_name = authenticate(token)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    logging.debug('here, after token')
    #update queries in database in case of a free user 
    condition = {"email": str(session_name)} 
    document = collection.find_one(condition)
    if document:
        paid= int(document.get('paid'))
        email = document.get('email')
        if email:
            print(f"Email: {email}")
        else:
            print("Email field not found in the document.")
    else:
        print("Document not found.")
    queries=int(document.get('queries'))
    if paid==0:
        if queries<10:
            condition = {"email": str(session_name)} 
        
# Define the field to increment and the increment value
            increment_field = "queries"
            increment_value = 1  # Change this to the desired increment value

# Update the document to increment the field
            result = collection.update_one(condition, {"$inc": {increment_field: increment_value}})

            print(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")

        elif queries>=10:
           return jsonify({ "answer": "To ask further questions, please buy a paid subscription to the Qdoc App."})
    # Extract the message from the data
    try:
        user_query = data.get('message')
        input_language = int(data.get('inputLanguage'))
        output_language = int(data.get('outputLanguage'))
    except Exception as e:
        logging.info(e)
        return jsonify({'message': e}), 400

    # translate question to english for LLM processing
    if input_language!=23:
        try:
            payload = {"source_language": input_language, "content": user_query, "target_language": 23}
            user_query = json.loads(requests.post('http://127.0.0.1:8000/scaler/translate', json=payload).content)
            user_query = user_query['translated_content']
            logging.debug(f'translated input: {user_query}')
        except Exception as e:
            return jsonify({'message': 'Error translating data, kindly try with english'}), 500
    
    if user_query and user_query.strip():
        # response from LLM
        res = ''
        try:
            response = user_input(user_query, session_name)
            res = response["output_text"]
        except Exception as e:
            logging.info(f'error in ask: {e}')
            return jsonify({'message': 'Error retreiving response from LLM'}), 500

        # translate to Output Language
        if output_language != 23:
            try:
                payload = {"source_language": input_language, "content": res, "target_language": output_language}
                res = json.loads(requests.post('http://127.0.0.1:8000/scaler/translate', json=payload).content)
                logging.debug(f'total output: {res}')
                res = res['translated_content']
            except Exception as e:
                return jsonify({'message': 'Error translating data, kindly try with english'}), 500

        logging.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
        return jsonify({"answer": res})

    return jsonify({"error": "Error processing request"}), 400

@app.route('/updatepayment', methods=['POST'])
def update_email():
    data = request.get_json()
    email = data.get('email')
    plan= int(data.get('paymentPlan'))
    today = date.today()
    if plan==1:
        future_date = today + timedelta(days=30)
    elif plan==2:
        future_date = today + timedelta(days=90)
    elif plan==3:
        future_date= today + timedelta(days=365)

    # Convert to YYYYMMDD format
    future_date_int = int(future_date.strftime("%Y%m%d"))

    print(f"DDMMYYYY: {future_date_int}")
    if email:
        # Find the document with the given email and update it
        result = collection.update_one(
            {'email': email},
            {'$set': {'paid': plan, 'expiry_date': future_date_int}}
        )

        if result.modified_count > 0:
            return jsonify({'message': 'Email updated successfully'}), 200
        else:
            return jsonify({'message': 'Email not found or already updated'}), 404
    else:
        return jsonify({'message': 'Invalid email'}), 400

@app.route('/check-payment-status', methods=['POST'])
def check_payment_status():
    data = request.get_json()
    email = data.get('email')

    if email:
        user = collection.find_one({'email': email})
        if user:
            payment_status = user.get('paid', 0)
            expiry_date_int = user.get('expiry_date', 0)  # Get expiry_date as int

            try:
                expiry_date_str = str(expiry_date_int)  # Convert int to string
                expiry_date = datetime.strptime(expiry_date_str, "%Y%m%d").date()
                remaining_days = (expiry_date - date.today()).days
                print(remaining_days)
                status = 'paid' if payment_status != 0 else 'not paid'
                if status == 'paid' and remaining_days <= 0:
                    status = 'Expired plan. Please Renew'

                return jsonify({'status': status, 'remaining_days': remaining_days}), 200
            except ValueError:  # Handle invalid date format
                return jsonify({'message': 'Invalid expiry date format'}), 500
        else:
            return jsonify({'message': 'User not found'}), 404
    else:
        return jsonify({'message': 'Invalid email'}), 400
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, ssl_context=('/etc/letsencrypt/live/qdocbackend.carnotresearch.com/fullchain.pem', '/etc/letsencrypt/live/qdocbackend.carnotresearch.com/privkey.pem'))