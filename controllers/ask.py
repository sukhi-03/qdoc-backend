from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import json
import requests
import logging
import time
from controllers.doc_summary import summarize_document
from langchain_elasticsearch.retrievers import ElasticsearchRetriever
from langchain.retrievers import EnsembleRetriever
from elasticsearch import Elasticsearch

client=Elasticsearch(cloud_id="1964ec43953b4c07957e65d979ba0958:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGM1ZTk5M2RjYzIzZTQ5Mzg4NGQzYzY3Zjk5NzczODY5JGY4MjQ4NTI3MDE4YzQ5NmI4MTcwMmEzMGZiM2E3MmUz",api_key="U0VtaTlKQUJFM3llVzNOT1VrZ2M6T1ZteHR4WUZSRWV0RkxyVFJ5TXNFQQ==")

def dynamic_body_func(query):
    return {
        "size": 1,
        "query": {
            "match": {
                "content": query
            }
        },
        "_source": {
            "includes": ["content"]
        }
    }

def is_summary_query(query):
    prompt  =   f"""User is asking questions regarding a document which can be in any format. If the user question is regarding summarizing the document, then respond with a signle integer 1. If the question is not related to summarizing document then respond 0. For general conversation or if any confusion respond with 0. Just give the single integer 0 or 1 as answer.
        Example quesiton: ```Who is bill gates```
        expected response: 0
        Exmaple question: ```Summarize this pdf in 5 sentences```
        expected response: 1
        Exmaple question: ```Summarize this```
        expected response: 1
        Example question: ```hi, how are you```
        expected response: 0

        user question:
        ```{query}```
        """
    
    llm = ChatOllama(model="llama3.1")
    response = llm.invoke(prompt)
    logging.info(f'{response.content} llm response: {response}')
    num = int(response.content)
    logging.info(num)
    if num:
        return True
    return False

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
    model = ChatOllama(temperature=0.1, model="llama3.1")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, usersession):
    start_time = time.time()

    # check if query is regarding summarizing the document
    if is_summary_query(user_question):
        return summarize_document(user_question, usersession)

    # Elastic+FAISS search impl
    es_weight = 0.6
    faiss_weight = 0.4
    weights = [es_weight,faiss_weight]
    esret = ElasticsearchRetriever(es_client=client,body_func=dynamic_body_func,  es_cloud_id="1964ec43953b4c07957e65d979ba0958:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGM1ZTk5M2RjYzIzZTQ5Mzg4NGQzYzY3Zjk5NzczODY5JGY4MjQ4NTI3MDE4YzQ5NmI4MTcwMmEzMGZiM2E3MmUz", index_name=usersession, es_api_key="U0VtaTlKQUJFM3llVzNOT1VrZ2M6T1ZteHR4WUZSRWV0RkxyVFJ5TXNFQQ==", content_field="content")
    embeddings=OllamaEmbeddings(model="mxbai-embed-large")
    new_db = FAISS.load_local(usersession, embeddings, allow_dangerous_deserialization=True).as_retriever()
    ensemble_retriever = EnsembleRetriever(retrievers=[esret, new_db], weights=weights)
    docs = ensemble_retriever.get_relevant_documents(user_question, k=2)  # by default k=4, top documents returned
    logging.info('---- %s seconds to do similarity search ----' % (time.time() - start_time))
    logging.info(f'relevant docs: {docs}')
    
    # resposne from LLM with RAG approach
    chain = get_conversational_chain()
    input_data = {
        'input_documents': docs,
        'question': user_question,
    }
    response = chain.invoke(input=input_data)
    logging.info('--- %s seconds to get response from llm ---' % (time.time() - start_time))
    return response["output_text"]

def translate_input(user_query, input_language):
    payload = {"source_language": input_language, "content": user_query, "target_language": 23}
    user_query = json.loads(requests.post('http://127.0.0.1:8000/scaler/translate', json=payload).content)
    logging.info(f'translated input: {user_query}')
    return user_query['translated_content']

def translate_output(res, output_language):
    payload = {"source_language": 23, "content": res, "target_language": output_language}
    res = json.loads(requests.post('http://127.0.0.1:8000/scaler/translate', json=payload).content)
    logging.info(f'translated output: {res}')
    return res['translated_content']

def get_llm_response(user_query, input_language, output_language, session_name):
    # translate question to english for LLM processing if in other language
    if input_language!=23:
        user_query = translate_input(user_query, input_language)
    
    if user_query and user_query.strip():
        # response from LLM
        res = ''
        try:
            res = user_input(user_query, session_name)
        except Exception as e:
            logging.info(f'error in text generation : {e}')
            raise

        # translate to Output Language
        if output_language != 23:
            res = translate_output(res, output_language)

        return res

def get_general_llm_response(user_query, input_language, output_language, session_name):
    if input_language!=23:
        user_query = translate_input(user_query, input_language)
    
    if user_query and user_query.strip():
        res = ''
        try:# response from LLM
            prompt  =   f"""You are a chat bot called Qdoc created by Carnot Research Pvt Ltd, which answers queries related to a document.
            Right now, user has not provided the document and simply interacting with chat bot.
            So, just answer the below user question in short and ask the user to upload files in the sidebar menu on left.

            Question :
            ```{user_query}```
            """

            llm = ChatOllama(model="llama3.1")
            llm_response = llm.invoke(prompt)
            res= str(llm_response.content)
        except Exception as e:
            logging.info(f'error in text generation : {e}')
            raise

        # translate to Output Language
        if output_language != 23:
            res = translate_output(res, output_language)

        return res