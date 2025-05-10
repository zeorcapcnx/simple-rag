import os
import json
import time
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Load environment variables from .env file
load_dotenv()

class Document:
    """Represents a document in the knowledge base."""
    
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
        self.embedding = None

class KnowledgeBase:
    """Vector store for documents with semantic search capabilities."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.documents = []
        self.encoder = SentenceTransformer(embedding_model)
        self.index = None
        
    def add_document(self, document: Document):
        """Add a document to the knowledge base."""
        document.embedding = self.encoder.encode(document.text)
        self.documents.append(document)
        self._update_index()
        
    def add_documents(self, documents: List[Document]):
        """Add multiple documents to the knowledge base."""
        for doc in documents:
            doc.embedding = self.encoder.encode(doc.text)
            self.documents.append(doc)
        self._update_index()
        
    def _update_index(self):
        """Update the FAISS index with document embeddings."""
        if not self.documents:
            return
            
        embeddings = np.array([doc.embedding for doc in self.documents]).astype('float32')
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search for the k most similar documents to the query."""
        if not self.index:
            return []
            
        query_vector = self.encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        return [self.documents[idx] for idx in indices[0]]

class DocumentLoader:
    """Utility for loading documents from files."""
    
    @staticmethod
    def load_text_file(file_path: str) -> Document:
        """Load a text file into a Document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        metadata = {
            'source': file_path,
            'type': 'text',
            'created_at': time.ctime(os.path.getctime(file_path))
        }
        
        return Document(text=text, metadata=metadata)
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load a PDF file into multiple Documents (one per page)."""
        documents = []
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if not text.strip():
                    continue
                    
                metadata = {
                    'source': file_path,
                    'type': 'pdf',
                    'page': i + 1,
                    'created_at': time.ctime(os.path.getctime(file_path))
                }
                
                documents.append(Document(text=text, metadata=metadata))
                
        return documents
    
    @staticmethod
    def load_directory(directory_path: str) -> List[Document]:
        """Load all supported files from a directory."""
        documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if filename.endswith('.txt'):
                documents.append(DocumentLoader.load_text_file(file_path))
            elif filename.endswith('.pdf'):
                documents.extend(DocumentLoader.load_pdf(file_path))
                
        return documents

class GeminiService:
    """Interface to the Gemini LLM."""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        # Configure the Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.genai_model = genai.GenerativeModel(model)
        
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a response to the prompt using Gemini."""
        try:
            response = self.genai_model.generate_content(prompt, 
                                                        generation_config=genai.types.GenerationConfig(
                                                            max_output_tokens=max_tokens
                                                        ))
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error generating a response."
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count for Gemini.
        Note: This is a rough approximation as Gemini doesn't expose token counting."""
        # Rough approximation: 4 characters ~ 1 token
        return len(text) // 4
    
class Agent:
    """A simple AI agent that uses RAG to answer questions."""
    
    def __init__(self, knowledge_base: KnowledgeBase, llm_service: GeminiService):
        self.knowledge_base = knowledge_base
        self.llm_service = llm_service
        self.memory = []
        
    def ask(self, question: str) -> str:
        """Process a question and generate a response."""
        # Store question in memory
        self.memory.append({"role": "user", "content": question})
        
        # Retrieve relevant documents
        relevant_docs = self.knowledge_base.search(question)
        
        # Create context from retrieved documents
        context = self._create_context(relevant_docs)
        
        # Generate prompt with context
        prompt = self._create_prompt(question, context)
        
        # Generate response
        response = self.llm_service.generate(prompt)
        
        # Store response in memory
        self.memory.append({"role": "assistant", "content": response})
        
        return response
    
    def _create_context(self, documents: List[Document]) -> str:
        """Create a context string from retrieved documents."""
        if not documents:
            return "No relevant information found."
            
        context_parts = []
        
        for i, doc in enumerate(documents):
            source_info = f"Source: {doc.metadata.get('source', 'Unknown')}"
            if 'page' in doc.metadata:
                source_info += f", Page: {doc.metadata['page']}"
                
            context_parts.append(f"[Document {i+1}]\n{source_info}\n{doc.text}\n")
            
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the LLM with the question and context."""
        return f"""You are a helpful research assistant. Use the provided context to answer the question. 
                 If you don't know the answer based on the context, say so.
                 Context: {context}

                 Question: {question}

                 Answer:"""

def main():
    # Initialize components
    kb = KnowledgeBase()
    llm_service = GeminiService()
    agent = Agent(kb, llm_service)
    
    # Load documents (from a 'docs' directory)
    docs_dir = 'docs'
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        # Create a sample document if none exist
        with open(os.path.join(docs_dir, 'sample.txt'), 'w') as f:
            f.write("This is a sample document about artificial intelligence. AI systems are designed to perform tasks that typically require human intelligence.")
    
    documents = DocumentLoader.load_directory(docs_dir)
    kb.add_documents(documents)
    
    print(f"Loaded {len(documents)} documents into the knowledge base.")
    
    # Simple CLI
    print("RAG Research Assistant with Gemini (type 'exit' to quit)")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break
            
        response = agent.ask(question)
        print("\nAssistant:", response)

if __name__ == "__main__":
    main()