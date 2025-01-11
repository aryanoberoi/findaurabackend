import logging
from pymongo import MongoClient
from datetime import date, timedelta, datetime

# MongoDB connection
mongo_url = "mongodb+srv://user2:user2@cluster0.sfiyids.mongodb.net/"
client = MongoClient(mongo_url)
db = client.test
collection=db.users

def is_user_limit_over(session_name):
    condition = {"email": str(session_name)} 
    user_details = collection.find_one(condition)
    paid = 0
    if user_details:
        if user_details.get('paid'):
            paid = int(user_details.get('paid'))
    else:
        # user not found (Illegal login)
        return True
    
    if paid == 0:
        queries = int(user_details.get('queries'))
        if queries<10:
            condition = {"email": str(session_name)} 
            # Define the field to increment and the increment value
            increment_field = "queries"
            increment_value = 1  # Change this to the desired increment value

            # Update the document to increment the field
            result = collection.update_one(condition, {"$inc": {increment_field: increment_value}})

            logging.info(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")

        elif queries>=10:
           return True

    return False

def upgrade_account(email, plan_limit_days):
    # calculate expiry date
    expiry_date = date.today() + timedelta(days=plan_limit_days)

    # Convert to YYYYMMDD format
    expiry_date_int = int(expiry_date.strftime("%Y%m%d"))
    logging.info(f"expiry date (DDMMYYYY): {expiry_date_int}")
        
    # Find the document with the given email and update it
    result = collection.update_one(
        {'email': email},
        {'$set': {'paid': plan, 'expiry_date': expiry_date_int}}
    )

    return result.modified_count

def get_account_status(email):
    # get user from database
    user = collection.find_one({'email': email})
    if user:
        payment_status = user.get('paid', 0)
        expiry_date_int = user.get('expiry_date', 0)  # Get expiry_date as int

        expiry_date_str = str(expiry_date_int)  # Convert int to string
        expiry_date = datetime.strptime(expiry_date_str, "%Y%m%d").date()
        remaining_days = (expiry_date - date.today()).days
        logging.info(f'remaining days for user: {remaining_days}')

        status = 'paid' if payment_status != 0 else 'not paid'
        if status == 'paid':
            return remaining_days
    
    return -1