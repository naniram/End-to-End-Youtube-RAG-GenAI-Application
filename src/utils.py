import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])    #creates a single string of text
    except Exception as e:
        print(f"An error occured while fetching script: {e}")
        return None
    

def create_vector_db(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

def setup_rag_chain(vector_store):
    llm = GooglePalm(temperature=0.7)

    rag_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vector_store.as_retriever(),
        return_source_documents = True
    )

    return rag_chain
