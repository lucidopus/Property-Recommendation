import os
from pymongo import MongoClient

from config import APP_NAME


MONGODB_URI = os.getenv("MONGODB_URI")

db_client = MongoClient(MONGODB_URI)
db = db_client.get_database(APP_NAME)
