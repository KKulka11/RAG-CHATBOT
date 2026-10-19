from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
from langchain_core.output_parsers import StrOutputParser

# Loading the keys
load_dotenv()

llm=ChatOpenAI(model="gpt-5-mini")
embedder=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

#Loading the transcripts
loader=DirectoryLoader(path=Path("./data/cleaned"),loader_cls=TextLoader,show_progress=True)

docs=loader.load()

#Vector store:
vs=Chroma(collection_name="rag-demo",embedding_function=embedder,
          persist_directory=Path("./saved_embeddings").as_posix())

#add documents
# vs.add_documents(chunks)

#Create a retriever

retriever=vs.as_retriever(search_kwargs={"k":3})

class RAGSTATE(TypedDict):
    
    query: str
    retrieved_doc: list[Document]
    context: str
    prompt: ChatPromptTemplate
    response: str

def retrieve(state: RAGSTATE)->dict:
    #Fetch your query
    query=state["query"]
    #Create your retriever
    retrieved_doc= retriever.invoke(query) # You will receive your 3 most valid docs
    
    context= "\n\n".join([doc.page_content for doc in retrieved_doc])
    
    return {"retrieved_doc": retrieved_doc,
            
            "context": context}

def augmentation(state: RAGSTATE)-> dict:
    
    prompt=ChatPromptTemplate.from_messages(
        
        [
            
            ("system","""You are a helful assistent. Answer the user's query based on given context only. If you cant find the context, 
             just say I dont know. Do not add any preamble in response"""),
            
            ("human", "context: {context}\n\nquery:{query}")
        ]
        
    )
    
    return {"prompt":prompt}

def generation(state: RAGSTATE)-> dict:
    
    query=state["query"]
    context=state["context"]
    prompt=state["prompt"]
    
    rag_chain= prompt | llm | StrOutputParser()
    
    response=rag_chain.invoke({"context":context,"query":query})
    
    return {"response": response}


#Create graph
graph_build=StateGraph(RAGSTATE)

graph_build.add_node("retrieve",retrieve)
graph_build.add_node("augmentation",augmentation)
graph_build.add_node("generation",generation)

graph_build.add_edge(START,"retrieve")
graph_build.add_edge("retrieve","augmentation")
graph_build.add_edge("augmentation","generation")
graph_build.add_edge("generation",END)

graph=graph_build.compile()

result=graph.invoke({"query":"What is LLMOps?"})
# print(result["response"])
