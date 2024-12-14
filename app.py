from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
import chainlit as cl
from langchain_groq import ChatGroq
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List



class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        documents = [str(doc) for doc in documents]
        return self.model.encode(documents).tolist()

    def embed_query(self, query: str) -> List[float]:
        if not isinstance(query, str):
            query = str(query)
        return self.model.encode([query])[0]
    


embedding = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

qroq_api_key = ''
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
llm = ChatGroq(temperature=0, groq_api_key=qroq_api_key, model_name="mixtral-8x7b-32768")

def process_file(files):
    import tempfile
    documents = []
    for file in files:
        # Determine file type and loader
        if hasattr(file, "name") and file.mime == 'text/plain':  
            Loader = TextLoader
        elif hasattr(file, "name") and file.mime == 'application/pdf': 
            Loader = PyPDFLoader
        else:
            raise ValueError(f"Unsupported file type: {file.type}")
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            with open(file.path, 'rb') as f:
                temp_file.write(f.read())
            temp_file.flush()  
            loader = Loader(temp_file.name)
            loaded_docs = loader.load()
            
            # Ensure each document has a source metadata
            for i, doc in enumerate(loaded_docs):
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = f'{file.name}_chunk_{i}'
            
            documents.extend(loaded_docs)

    splits = text_spliter.split_documents(documents)
    
    # Ensure each split document has a source metadata
    for i, split in enumerate(splits):
        if 'source' not in split.metadata:
            split.metadata['source'] = f'split_{i}'

    return splits

def get_vec_search(file):
    doc_splits = process_file(file)
    cl.user_session.set('docs', doc_splits)
    # chroma = Chroma(
    #     collection_name="my_collection",

    # )
    texts = [doc.page_content for doc in doc_splits]
    metadatas = [doc.metadata for doc in doc_splits]
    
    vec_search = Chroma.from_texts(
        texts=texts,
        embedding=embedding, 
        metadatas=metadatas,
        collection_name="dcd_store"
    )

    return vec_search


welcome_message = " Chat with your PDFs and get valuable insights"


@cl.on_chat_start
async def start():
    await cl.Message(content="You can now chat with your pdfs click the clip to upload.").send()

@cl.on_message
async def main(message):
   
    if  len(message.elements) > 0:
        files = [element for element in message.elements if isinstance(element, cl.File)]
        
        splits = process_file(files)
           
        # Update the user session with the new documents
        existing_docs = cl.user_session.get("docs", [])
        updated_docs = existing_docs + splits
        cl.user_session.set("docs", updated_docs)
        vec_search =cl.user_session.get('vec_search')
        if vec_search:
            vec_search.add_documents(splits)
        else:
            vec_search = await cl.make_async(get_vec_search)(files)
            cl.user_session.set('vec_search', vec_search)
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=cl.user_session.get("vec_search").as_retriever(max_tokens_limit=4097)
                )
            cl.user_session.set("chain", chain)

    chain = cl.user_session.get('chain')
    cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True,
    answer_prefix_tokens=["FINAL", "ANSWER"]
)
    cb.answer_reached = True 
    res = await chain.acall(message.content, callbacks=[cb])
    answer =res['answer']
    sources = res['sources'].strip()
    source_elements = []

    docs = cl.user_session.get("docs")
    docs_metadata = [doc.metadata for doc in docs]
    all_sources = [m['source'] for m in docs_metadata]

    if sources:
        found_sources = []
        for source in sources.split(','):
            source_name = source.strip().replace('.','')
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            source_elements.append(cl.Text(content=text, name=source_name))
        if found_sources:
            answer += f'\nSources: {",".join(found_sources)}'
        else:
            answer += '\nNo sources found'
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer,elements=source_elements).send()