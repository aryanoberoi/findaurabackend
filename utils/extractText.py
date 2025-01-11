import os
from werkzeug.utils import secure_filename
import logging
# file text loaders
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader
import pdfplumber
from langchain.schema import Document
import base64
import pypdfium2 as pdfium
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to send image to GPT-4 Vision and get the caption
def get_image_caption(base64_image):
    import os
    import google.generativeai as genai

    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    myfile = genai.upload_file(base64_image)
    print(f"{myfile=}")

    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(
    [myfile, "\n\n", "Can you tell me about the text in this photo?"]
)
    return str(result.text)

# 
# Extract text from docx file
# Returns list of Document
#
def get_text_from_doc(doc_file):
    # save file temporarily to load using langchain tool
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

# 
# Extract text from txt file
# Returns list of Document
#
def get_text_from_txt(txt_file):
    # save file temporarily to load using langchain tool
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

# 
# Extract text from pdf file
# Returns list of Document
#
def get_text_from_pdf(pdf_file):
    # save file temporarily to load using langchain tool
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

    logging.info(f'extracted data len: {len(data)}')
    return data

# 
# Extracts and concatenates text from all given files
# Returns list of Document
#
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

def scan_doc(file):
    pdf = pdfium.PdfDocument(file[0])

    # Text file to store captions
    with open("captions.txt", "w", encoding="utf-8") as caption_file:
        # Loop over pages and render each one as an image
        for i in range(len(pdf)):
            page = pdf[i]
            image = page.render(scale=4).to_pil()

            # Save image temporarily
            image_path = f"output_{i:03d}.jpg"
            image.save(image_path)

            # Get caption from GPT-4 Vision, passing the filename directly
            caption = get_image_caption(image_path)

            # Append caption to the text file
            caption_file.write(f"Caption for page {i+1}:\n{caption}\n\n")

            # Optionally remove the image file after processing
            os.remove(image_path)

    print("Captions saved to captions.txt")
