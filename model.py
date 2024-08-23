from pymongo import MongoClient
from config import MONGO_URI, MONGO_DBNAME

def get_mongo_client():
  uri = MONGO_URI
  db_name = MONGO_DBNAME
  client = MongoClient(uri)
  db = client[db_name]

  return db

def get_all_contents():
  db = get_mongo_client()
  collection = db['team3']
  documents = collection.find({}, {"_id": 0, "content": 1})

  return [doc['content'] for doc in documents]
