#!/usr/bin/env python
# coding: utf-8

# In[4]:


from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
#import chainlit as cl
import streamlit as st


DB_FAISS_PATH = 'vectorstore/db_faiss'


custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain


# Loading the model
import torch


def load_llm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = CTransformers(
        model="D:\internships\Rajyug ITSolutions\LLM\model\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
        device=device  # Specify device here
    )
    return llm


# QA Model Function
def qa_bot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}  # Specify device here
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    # db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response['result']  # Extract only the 'result' field from the response


def main():
    st.title("Medical Chatbot")

    query = st.text_input("Enter your medical query:")
    if st.button("Get Answer"):
        if query:
            answer = final_result(query)
            st.write("Bot's Response:")
            st.write(answer)  # Print only the 'result'
        else:
            st.write("Please enter a query.")

    # Call qa_bot and store the returned chain
    qa_chain = qa_bot()

    # Assuming you have a chain named 'my_chain' (commented out)
          # Assuming you have a chain named 'my_chain'

# Old (deprecated):
# result = my_chain()



# Verbosity (if needed)
    from langchain.globals import set_verbose, get_verbose

# Set verbosity to True
    langchain.globals.set_verbose(True)


# Check current verbosity
    langchain.globals.get_verbose(True)
    # ... (code using qa_chain, if applicable)

    # Use the 'invoke' method to execute the chain (fix for deprecation warning)
    # result = qa_chain.invoke()  # Uncomment if you need the result

    # Verbosity section (commented out for clarity)
    # from langchain.globals import set_verbose, get_verbose
    #
    # # Set verbosity to True (optional)
    # # langchain.globals.set_verbose(True)
    #
    # # Check current verbosity
    # # current_verbosity = get_verbose()
    # New (recommended):
    result = qa_chain.invoke(input=query)

if __name__ == "__main__":
    main()
