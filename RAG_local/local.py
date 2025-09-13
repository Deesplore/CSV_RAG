import os
import pandas as pd
from dotenv import load_dotenv
# LangChain components for our RAG system
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load your environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not set. Please set it in your environment.")
print("All libraries loaded successfully!")


# Load your CSV file
csv_file_path = r"<Give your csv path here>"  
# Replace with your actual file path
data_frame = pd.read_csv(csv_file_path, nrows=100)
print(f"Loaded {len(data_frame)} rows (limited to 100)")
print(f"Columns available: {list(data_frame.columns)}")
print(f"Data shape: {data_frame.shape}")
# Look at the first few rows to understand our data structure
print("First 5 rows of data:")
print(data_frame.head())   



def create_readable_text_from_row(row):
    description_parts = []
    for column_name, value in row.items():
        if pd.notna(value):  # Only include non-empty values
            description_parts.append(f"{column_name}: {value}")
    # Join everything into one readable sentence
    return ". ".join(description_parts) + "."

text_documents = []
for index, row in data_frame.iterrows():
    # Convert each row to readable text
    readable_description = create_readable_text_from_row(row)
    # Create a Document object (LangChain's format)
    doc = Document(page_content=readable_description)
    text_documents.append(doc)
  
print(f"Created {len(text_documents)} document objects")
# A few examples of what we created
print("\nExamples of converted documents:")
for i in range(min(3, len(text_documents))):
    print(f"Document {i+1}: {text_documents[i].page_content}")


# Set up our embedding system
embedding_model = OpenAIEmbeddings()
print("Embedding system initialized")
print("This will convert our text into numerical vectors that capture meaning")


print("Creating vector store from documents...")
vector_search_store = FAISS.from_documents(text_documents, embedding_model)
print(f"Vector store created with {len(text_documents)} documents")
print("Each document is now represented as a vector for fast similarity search")


# Test our search system
test_query = "Who is at risk for Lymphocytic Choriomeningitis (LCM)? ?"
similar_documents = vector_search_store.similarity_search(test_query, k=3)
print(f"Testing search for: '{test_query}'")
print(f"Found {len(similar_documents)} similar documents:")
for i, doc in enumerate(similar_documents):
    print(f"\nResult {i+1}: {doc.page_content}")



# Initialize our AI language model
ai_assistant = ChatOpenAI(
    temperature=0,  # Low temperature = more focused, consistent answers
    model="gpt-4o-mini"  # Good balance of quality and cost
)
print("AI assistant initialized")
print("Temperature set to 0 for consistent, factual responses")




# Create a retriever from our vector store
document_retriever = vector_search_store.as_retriever(
    search_kwargs={"k": 3}  # Retrieve top 3 most similar documents
)
print("Document retriever created")
print("It will find the 3 most relevant pieces of information for each question")



# Create a prompt template for our AI assistant
answer_prompt = PromptTemplate.from_template("""
You are a helpful data analyst. Use the following information from the CSV data to answer the user's question accurately.
Important instructions:
- Only use information from the provided context
- If you can't find the answer in the context, say "I don't have that information in the data"
- Be specific and include relevant details from the data
- Keep your answer clear and concise
Context from CSV data:
{context}
User question: {question}
Answer:
""")
print("Prompt template created")



# Build the complete RAG chain using LCEL
rag_pipeline = (
    {
        "context": document_retriever,  # Find relevant documents
        "question": RunnablePassthrough()  # Pass the question through
    }
    | answer_prompt  # Format everything into our prompt
    | ai_assistant  # Generate the answer
    | StrOutputParser()  # Clean up the output
)

print("RAG system is ready! Ask questions about your CSV data.")
print("======== YOU CAN NOW ASK THE QUESTIONS PLEASE NOTE ONLY FIRST 100 RECORDS ARE PULLED FROM CSV AND USED AS KNOWLEDGE BASE SO PLEASE ASK QUESTION AMONG THAT 100 QUESTION.=========")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("You: ")
    if user_question.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting Q&A. Goodbye!")
        break
    
    try:
        answer = rag_pipeline.invoke(user_question)
        print(f"AI: {answer}\n")
    except Exception as e:

        print(f"⚠️ Error: {e}\n")
