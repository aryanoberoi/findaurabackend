from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import asyncio
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import time
import logging
import os
from controllers.ask import get_general_llm_response
from controllers.database import upgrade_account, get_account_status
import sys
from dotenv import load_dotenv
mysql_username = os.getenv("MYSQL_USERNAME")
mysql_password = os.getenv("MYSQL_PASSWORD")
sys.stdout.reconfigure(encoding='utf-8')
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
        session_id = request.form.get('sessionId')
        user_session = user_session + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    logging.info('--- %s seconds to authenticate req ---' % (time.time() - start_time))
    ifscan=request.form.get('ifscan')
    print("ifscan value ", ifscan)
    files = request.files.getlist("files")
    logging.info(f'request is \n {request.files}')
 
    # Process files
    valid_files = [f for f in files if f.filename.endswith(('.pdf', '.docx', '.txt'))]
    csv_files = [f for f in files if f.filename.endswith(('.xlsx', '.csv'))]
        # Create a new database for this session
    connection = mysql.connector.connect(
        host="localhost",
        user=str(mysql_username),  # Replace with your MySQL username
        password=str(mysql_password)  # Replace with your MySQL password
    )
    
    cursor = connection.cursor()
    db_name=user_session.replace('@','').replace('.','')
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    
    
    engine = create_engine(f'mysql://{mysql_username}:{mysql_password}@localhost/{db_name}')
    raw_text = []
    if files and files[0].filename != '':
        if csv_files:           
            for file in csv_files:
                # If the file is a CSV
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)  # Read CSV file into a DataFrame
                # If the file is an XLSX
                elif file.filename.endswith('.xlsx'):
                    df = pd.read_excel(file)  # Read XLSX file into a DataFrame
                
                # Use the file name (without extension) as the table name
                table_name = file.filename.rsplit('.', 1)[0]
                
                try:
                    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
                except Exception as e:
                    logging.info(f'error: {e}')
                    return jsonify({'message': 'The given CSV/XLSX has a problem with column names. Please check and reupload.'}), 500
            
            message = "Files successfully uploaded."

        if valid_files:
            raw_text.extend(get_text_from_files(valid_files))  # Process valid files
            message = "Files successfully uploaded."
    else:
        message = "Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."

    logging.info('--- %s seconds to extract data from file ---' % (time.time() - start_time))

    try:
        if not raw_text and not csv_files:
            logging.info('No data was extracted!')
            return jsonify({'message': 'No data was extracted!'}), 500
        if not raw_text and csv_files:
            logging.info('CSV files successfully uploaded!')
            return jsonify({"status": "ok", "message": message}), 200
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
        session_id = request.form.get('sessionId')
        user_session = user_session + str(session_id.lower())
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
        csv_files = [f for f in files if f.filename.endswith(('.xlsx', '.csv'))]

        # Create a new database for this session
        db_name = user_session.replace('@', '').replace('.', '')
        connection = mysql.connector.connect(
        host="localhost",
        user=str(mysql_username),  # Replace with your MySQL username
        password=str(mysql_password)  # Replace with your MySQL password
    )
    
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")

        engine = create_engine(f'mysql://{mysql_username}:{mysql_password}@localhost/{db_name}')

        if valid_files:
            raw_text.extend(get_text_from_files(files))
            message = "Files successfully uploaded."
        else:
            message = "Please upload files in PDF, DOC, DOCX, TXT, XLSX, or CSV format."

        # Process CSV files
        if csv_files:
            for file in csv_files:
                # If the file is a CSV
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)  # Read CSV file into a DataFrame
                # If the file is an XLSX
                elif file.filename.endswith('.xlsx'):
                    df = pd.read_excel(file)  # Read XLSX file into a DataFrame
                
                # Use the file name (without extension) as the table name
                table_name = file.filename.rsplit('.', 1)[0]
                
                # Insert the DataFrame into a new table in the session's database
                df.to_sql(table_name, con=engine, if_exists='replace', index=False)

    logging.info('--- %s seconds to extract data from file ---' % (time.time() - start_time))

    try:
        if not raw_text and not csv_files:
            logging.info('No data was extracted!')
            return jsonify({'message': 'No data was extracted!'}), 500
        if not raw_text and csv_files:
            logging.info('CSV files successfully uploaded!')
            return jsonify({"status": "ok", "message": message}), 200
        
        # Assuming the same function to store vectors
        store_vector(raw_text, user_session)
        logging.info('--- %s seconds to create FAISS index ---' % (time.time() - start_time))
        return jsonify({"status": "ok", "message": message}), 200
    except Exception as e:
        logging.info(f'error: {e}')
        return jsonify({'message': 'Error creating vector index'}), 500


@app.route('/askcisce', methods=['POST'])
def get_cisce():
    start_time = time.time()
    data = request.get_json()
    
    # Extract the message from the data
    session_name = "user20240902t065302"
    try:
        user_query = data.get('message')
        input_language = int(data.get('inputLanguage'))
        output_language = int(data.get('outputLanguage'))
    except Exception as e:
        logging.info(e)
        return jsonify({'message': e}), 400
    
    
    logging.info(f'cisce query: {user_query}')

    # get response from llm
    try:
        llm_response = get_llm_response(user_query, input_language, output_language, session_name)
    except Exception as e:
        logging.info(f'Error: {e}')
        return jsonify({'message': 'Error generating response from LLM'}), 500

    logging.info(f'cisce response: {llm_response}')
    logging.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
    return jsonify({"answer": llm_response})

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return {"message": "app is up and running"}

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()
    data = request.get_json()
    hascsvxl = data.get('hasCsvOrXlsx')
    # Authenticate request and extract vector db directory from token
    session_name = ''
    try:
        token = data.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        session_name = authenticate(token)
        session_id = data.get('sessionId')
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
    temperature=data.get('temperature')
    mode=data.get('mode')
    # get response from llm
    try:
        logging.info(f'user query: {user_query}')
        if context:
            session_name = session_name + str(session_id.lower())
            llm_response = get_llm_response(user_query, input_language, output_language, session_name, hascsvxl=hascsvxl, mode=mode)
        else:
            llm_response = get_general_llm_response(user_query, input_language, output_language, session_name)
    except Exception as e:
        logging.info(f'Error: {e}')
        return jsonify({'message': 'Error generating response from LLM'}), 500

    logging.info(f'llm response: {llm_response}')
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
    

@app.route('/trialAsk', methods=['POST'])
def trial_ask():
        # Extract data from the request
    data = request.json
    message = data.get('message')
    fingerprint = data.get('fingerprint')
    input_language = data.get('inputLanguage')
    output_language = data.get('outputLanguage')
    context = data.get('context')
    mode = data.get('mode')

        # Here you can add your logic to handle the request
        # For example, check if the free trial limit is exhausted
        # and return the appropriate response.

        # Process the message and generate a response
    answer = get_general_llm_response(message, input_language, output_language)

        # Return the response
    return jsonify({"answer": answer}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=443)
    
#, ssl_context=('/etc/letsencrypt/live/qdocbackend.carnotresearch.com/fullchain.pem', '/etc/letsencrypt/live/qdocbackend.carnotresearch.com/privkey.pem')