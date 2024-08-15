from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import asyncio
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from bs4 import BeautifulSoup
import time
import logging
import os
# import custom functions
from graph import graph_from_document
from utils.extractText import get_text_from_files#, get_text_from_urls
from controllers.upload import store_vector, get_text_chunks
from controllers.database import is_user_limit_over
from controllers.ask import get_llm_response, get_general_llm_response
from controllers.database import upgrade_account, get_account_status

# Logger config
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = 'supersecretkey'

# Function to handle asyncio event loop
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
get_or_create_eventloop()

def authenticate(token):
    data = jwt.decode(token, 'secret', algorithms=["HS256"])
    current_user = data['email']
    logging.info(f'user email: {current_user}')
    return current_user

@app.route('/upload', methods=['POST', 'OPTIONS'])
def index(): 
    start_time = time.time()
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        user_session = authenticate(token)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    logging.info('--- %s seconds to authenticate req ---' % (time.time() - start_time))

    files = request.files.getlist("files")
    logging.info(f'request is \n {request.files}')
 
    # Process files
    raw_text = []
    if files and files[0].filename != '':
        valid_files = all(f.filename.endswith(('.pdf', '.docx', '.txt')) for f in files)
        if valid_files:
            raw_text.extend(get_text_from_files(files))
            message = "Files successfully uploaded."
        else:
            message = "Please upload files in PDF, DOC, DOCX, or TXT format."

    logging.info('--- %s seconds to extract data from file ---' % (time.time() - start_time))

    try:
        if not raw_text:
            logging.info('No data was extracted!')
            return jsonify({'message': 'No data was extracted!'}), 500
        store_vector(raw_text, user_session)
        logging.info('--- %s seconds to create FAISS index ---' % (time.time() - start_time))
        return jsonify({"status": "ok", "message": message}), 200
    except Exception as e:
        logging.info(f'error: {e}')
        return jsonify({'message': 'Error creating vector index'}), 500

@app.route('/add-upload', methods=['POST', 'OPTIONS'])
def newfile():
    start_time = time.time()
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        user_session = authenticate(token)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    logging.info('--- %s seconds to authenticate req ---' % (time.time() - start_time))

    files = request.files.getlist("files")
    logging.info(f'request is \n {request.files}')
 
    # Process files
    raw_text = []
    if files and files[0].filename != '':
        valid_files = all(f.filename.endswith(('.pdf', '.docx', '.txt')) for f in files)
        if valid_files:
            raw_text.extend(get_text_from_files(files))
            message = "Files successfully uploaded."
        else:
            message = "Please upload files in PDF, DOC, DOCX, or TXT format."

    logging.info('--- %s seconds to extract data from file ---' % (time.time() - start_time))

    try:
        if os.path.isdir(user_session):
            logging.info('Directory already exists')
            get_new_vector_store(raw_text, user_session)
        else:
            return jsonify({'message': 'Directory does not exist'}), 500
    except Exception as e:
        logging.info(f'error: {e}')
        return jsonify({'message': 'Error creating vector index'}), 500
    
    logging.info('--- %s seconds to create FAISS index ---' % (time.time() - start_time))
    
    return jsonify({"status": "ok", "message": "request processed successfully!"}), 200

@app.route('/graph', methods=['POST'])
def get_graph():
    start_time = time.time()
    # Authenticate request and extract FAISS directory from token
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        session_name = authenticate(token)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    logging.info(f'Authenticated graph request in {time.time() - start_time} seconds')

    try:
        files = request.files.getlist("files")
        url_input = [] 
        i = 0
        while True:
            temp_url = request.form.get(f'urls[{i}]')
            if temp_url:
                url_input.append(temp_url)
            else:
                break
            i += 1
    except Exception as e:
        logging.info(f'Error while extracting request: {e}')
    
    logging.info(f'Extracted files and urls from request: {len(files)}, {len(url_input)}')

    html_content = graph_from_document(files, url_input)
    logging.info('--- %s seconds to get knowledge graph ---' % (time.time() - start_time))
    
    return Response(html_content, mimetype='text/html')

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return {"message": "app is up and running"}

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()
    data = request.get_json()
    #logging.info(f'ask request: {data}')

    # Authenticate request and extract FAISS directory from token
    session_name = ''
    try:
        token = data.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        session_name = authenticate(token)
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    logging.debug('here, after token')

    #update queries in database in case of a free user 
    if is_user_limit_over(session_name):
        return jsonify({ "answer": "To ask further questions, please upgrade your account."})
    
    # Extract the message from the data
    try:
        user_query = data.get('message')
        input_language = int(data.get('inputLanguage'))
        output_language = int(data.get('outputLanguage'))
        context = data.get('context')
    except Exception as e:
        logging.info(e)
        return jsonify({'message': e}), 400

    # get response from llm
    try:
        if context:
            llm_response = get_llm_response(user_query, input_language, output_language, session_name)
        else:
            llm_response = get_general_llm_response(user_query, input_language, output_language, session_name)
    except Exception as e:
        logging.info(f'Error: {e}')
        return jsonify({'message': 'Error generating response from LLM'}), 500

    logging.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
    return jsonify({"answer": llm_response})

@app.route('/updatepayment', methods=['POST'])
def update_email():
    try:
        data = request.get_json()
        email = data.get('email')
        plan= int(data.get('paymentPlan'))
    except Exception as e:
        logging.info(e)
        return jsonify({'message': 'Error extracting request data'}), 500
    
    if email and plan:
        if plan==1:
            plan_limit_days = 30
        elif plan==2:
            plan_limit_days = 90
        elif plan==3:
            plan_limit_days = 365
        else:
            return jsonify({'message': 'Plan not supported'}), 400

        # update account limits in database
        try:
            update_count = upgrade_account(email, plan_limit_days)
            if update_count:
                return jsonify({'message': 'Account upgraded successfully!'}), 200
            else:
                return jsonify({'message': 'Account with email not found!'}), 400
        except Exception as e:
            logging.info(f'Error upgrading account: {e}')
            return jsonify({'message': 'Error updating database'}), 400
    else:
        return jsonify({'message': 'Either email or plan empty!'}), 500

@app.route('/check-payment-status', methods=['POST'])
def check_payment_status():
    try:
        data = request.get_json()
        email = data.get('email')
    except Exception as e:
        logging.info(e)
        return jsonify({'message': 'Error extracting request data'}), 500

    if email:
        try:
            account_limit = get_account_status(email)
        except Exception as e:
            logging.info(f'Error checking user limits: {e}')
        
        status = 'not paid'
        if account_limit == 0:
            status = 'Expired plan. Please Renew'
        elif account_limit > 0:
            status = 'paid'
        
        return jsonify({'status': status, 'remaining_days': account_limit}), 200
    else:
        return jsonify({'message': 'Invalid email'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)