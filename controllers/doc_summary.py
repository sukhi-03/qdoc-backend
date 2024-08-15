from summarizer import Summarizer
from langchain_community.chat_models import ChatOllama 
import logging

def summarize_document(query, usersession):
    # Read the string from the file
    filename = f'{usersession}/content.txt'
    with open(filename, "r") as file:
        full_text = file.read()
    
    model = Summarizer()
    most_important_sents = model(full_text, num_sentences=40) # We specify a number of sentences
    logging.info(f'imp sents: {most_important_sents}')

    graphtext_filename = f'{usersession}/graphtext.txt'
    try:
        with open(graphtext_filename, "w") as graphtext_file:
            graphtext_file.write(most_important_sents)
    except IOError as e:
        logging.error(f"Error writing to {graphtext_filename}: {e}")
        return None
    
    prompt = f'''<task>
        <instruction>
        You will be given a series of sentences from a document/paper/article. Your goal is to give a summary of the document with respect to the query. 
        Do not include any opening statement like "here is the summary," etc. The query and sentences will be enclosed in triple backticks (```). 
        If the sentences do not provide meaningful information or context for the query, respond with "No relevant information provided."
        </instruction>
        <query>
        ```{query}```
        </query>
        <sentences>
        ```{most_important_sents}```
        </sentences>
        </task>'''

    llm = ChatOllama(temp=0, model="llama3.1")
    summary = llm.invoke(prompt)
    logging.info(f'summary result : {summary}')
    return summary.content