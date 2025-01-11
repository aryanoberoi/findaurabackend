# Text Extraction and Knowledge Graph Generation

This project provides a comprehensive solution for extracting text from various document formats, including PDF, DOC, DOCX, and TXT files, as well as from web pages. It further processes the extracted text to create knowledge graphs, offering a visual representation of the relationships between different entities within the text.

## Features

- **Text Extraction**: Supports multiple file formats and web pages.
- **Knowledge Graph Generation**: Converts extracted text into knowledge graphs.
- **Support for Multiple Document Types**: Handles PDF, DOC, DOCX, and TXT files.
- **Web Page Text Extraction**: Fetches and processes text from given URLs.
- **Temporary File Handling**: Manages temporary files for processing without leaving residuals.
- **Modular Design**: Each functionality is encapsulated in separate functions for easy maintenance and scalability.

## Installation

To set up the project environment, ensure you have Python installed on your system. Then, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
run app.py

Ollama settings :
For Ollama to run behind a proxy, run the following command 
sudo systemctl edit ollama
then run : Ollama serve 