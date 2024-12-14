# Chat with PDFs: A Chainlit-Powered PDF Chatbot

## Overview

This project is a PDF chatbot application powered by **Chainlit**, **LangChain**, and **Chroma**. The app allows users to upload PDF and text documents, process them into chunks, and then query them conversationally. By leveraging the latest in AI-based embeddings and language models, the chatbot provides insightful answers while referencing relevant document sources.

---

## Features

- **File Upload Support**: Users can upload PDF and plain text files to interact with their content.
- **Chunk Processing**: Large documents are split into manageable chunks for efficient search and retrieval.
- **Vector Search with Chroma**: Documents are embedded using **SentenceTransformers** and stored in **Chroma**, a fast and scalable vector store.
- **Conversational QA**: Powered by **ChatGroq**, users can ask questions in natural language and receive detailed responses with cited sources.
- **Real-Time Document Updates**: Uploaded files are dynamically processed and added to the document corpus.
- **Streamed Responses**: Answers are streamed for an engaging user experience.

---

## Technologies Used

### Frameworks and Libraries
- **Chainlit**: For building and deploying the chatbot interface.
- **LangChain**: To manage document loaders, text splitting, and retrieval chains.
- **SentenceTransformers**: For generating embeddings of document text and user queries.
- **Chroma**: A high-performance vector database for storing and searching document embeddings.

### Language Models
- **ChatGroq**: A state-of-the-art LLM for generating responses to user queries.

### Utilities
- **RecursiveCharacterTextSplitter**: For chunking large documents.
- **PyPDFLoader**: For extracting content from PDF files.
- **TextLoader**: For loading plain text files.

---

## Installation

### Prerequisites
1. **Python 3.8+**
2. **pip**: Package manager for Python.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/chat-with-pdfs.git
   cd chat-with-pdfs
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys for **ChatGroq**:
   - Replace the placeholder `qroq_api_key` in the code with your actual API key.

---

## Usage

1. Start the chatbot:
   ```bash
   chainlit run main.py -w
   ```

2. Open the browser interface (usually at `http://localhost:8000`).

3. Upload a document (PDF or text) using the clip icon.

4. Ask questions about your uploaded documents, and get real-time answers with cited sources.

---

## Key Functions

### `process_file(files)`
- Loads files and splits them into chunks for easier embedding and retrieval.

### `get_vec_search(file)`
- Processes the documents and stores them in a Chroma vector store.

### `start()`
- Initializes the chatbot with a welcome message.

### `main(message)`
- Handles user interactions, processes uploaded documents, and answers queries.

---

## Example Workflow

1. **Upload a File**: Drag and drop your PDF or text file into the chatbot interface.
2. **Ask Questions**: Type in queries like:
   - *"What is the main topic of the document?"*
   - *"Provide details from section X of the file."*
3. **Receive Answers**: The bot responds with detailed answers and references to the source document.

---

## Contribution

Feel free to open issues or submit pull requests. Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [LangChain](https://langchain.com/)
- [Chainlit](https://chainlit.io/)
- [SentenceTransformers](https://www.sbert.net/)
- [Chroma](https://www.trychroma.com/) 

Enjoy seamless interaction with your documents! 🚀
