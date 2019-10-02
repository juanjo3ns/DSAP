import pymongo

DB = "test"
COLL = "col1"

def see_db():
    client = pymongo.MongoClient("mongodb://mongo:27017")
    return client.list_database_names()

def see_coll(collection=COLL):
    client = pymongo.MongoClient("mongodb://mongo:27017")
    return client[collection].list_collection_names()
    
def get_all(db=DB, collection=COLL):
    client = pymongo.MongoClient("mongodb://mongo:27017")
    db = client[db]
    col = db[collection]
    return list(col.find({}))

def insert_one(mydict, db=DB, collection=COLL):
    client = pymongo.MongoClient("mongodb://mongo:27017")
    db = client[db]
    col = db[collection]
    return col.insert_one(mydict)



#new_db(mydict = { "name": "John", "address": "Highway 37" })
print(get_all())

