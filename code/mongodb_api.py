import pymongo
import csv
import os
import numpy as np

DB = "DSAP"
COLL = "urban"
client = pymongo.MongoClient("mongodb://mongo:27017")

possible_items ={"city": "Barcelona"}
#------------------------------------------------------ TAGS ------------------------------------------------------
#file_name: "tram-vienna-202-6108-a.wav"
#city: lisbon, lyon, prague, barcelona, helsinki, london, paris, stockholm, vienna
#slot: "202"
#id: "6108"
#tag: airport, bus, shopping_mall, street_pedestrian, street_traffic, metro_station, park, metro, public_square, tram
#split: train, val
#------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------URBAN------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#file_name: "00_00000066.wav"
#split: train, validate
#sensor_id = 1
#annotator_id = 1
#high_labels = [1,1,0,1,0,1,0,1]
#------------------------------------------------------------------------------------------------------------------

def see_db():   #Lists all databases in the Mongod
    return client.list_database_names()

def see_coll(db=DB):   #Lists all collections in the Mongod database selected
    return client[db].list_collection_names()

def get_all(db=DB, collection=COLL): #return a list of all items
    db = client[db]
    col = db[collection]
    return list(col.find({}))

def insert_one(mydict, db=DB, collection=COLL): #insert one intem in the database (be sure has the same structure)
    db = client[db]
    col = db[collection]
    return col.insert_one(mydict)

def clean_db(db=DB, collection=COLL): #remove all information in the database
    db = client[db]
    col = db[collection]
    cursor = list(col.find({}))
    for item in cursor:
        result = col.delete_one({"file_name":item["file_name"]})

def get_from(filter_tag="city", filter_value="barcelona", filt=None, db=DB, collection=COLL): #return a list of all items that are specified in the filter
    # example of query: { "address": "Park Lane 38" }
    db = client[db]
    col = db[collection]

    if filt == None:
        filt = {filter_tag: filter_value}

    return list(col.find(filt))


def read_csv():
    files = [('fold1_evaluate.csv', 'val'), ('fold1_train.csv', 'train'), ('fold1_test.csv', 'test')]
    for f in files:
        #with open(os.path.join('TAU-urban-acoustic-scenes-2019-development/evaluation_setup/', f[0])) as csv_file:
        with open(os.path.join('/home/data/TAU-urban-acoustic-scenes-2019-openset-development/evaluation_setup/', f[0])) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for i, row in enumerate(csv_reader):
                if i:
                    item = {}
                    if row[1] == "unknown":
                        continue

                    item["file_name"] = row[0].split("/")[1]
                    item["city"] = row[0].split("-")[1]
                    item["slot"] = row[0].split("-")[2]
                    item["id"] = row[0].split("-")[3]
                    if len(row) > 1:
                        item["tag"] = row[1]
                    item["split"] = f[1]
                    insert_one(item, db="DSAP", collection="urban")
                    print(item)



def read_csv_task5():
    files = ['annotations-dev.csv']
    for f in files:
        with open(os.path.join('/home/data/task5/', f)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i:
                    item = {}
                    if row[1] == "unknown":
                        continue
                    item["split"] = row[0]
                    item["sensor_id"] = row[1]
                    item["file_name"] = row[2]  #62
                    item["annotator_id"] = row[3]
                    item["high_labels"] = [int(row[62]), int(row[63]), int(float(row[64])), int(row[65]), int(row[66]), int(row[67]), int(row[68]), int(row[69])]
                    insert_one(item, db="DSAP", collection="task5")
                    print(i)


def th(th, x):
    if (np.sum(x)/x.size) > th:
        return 1
    return 0
def repair_task5_collection():
    punt = get_from(filt={"split":"train"}, collection="task5")
    names = []
    dicts = []

    for i, x in enumerate(punt):
        #print("{:.1f}%".format(i/len(punt)*100), end = "\r" )
        sol = []
        name = x["file_name"]
        if name in names:
            continue

        names.append(name)

        a = get_from(filt={"file_name": name, "split":"train"}, collection="task5")
        for n in a:
            sol.append(n["high_labels"])
        sol = np.array(sol).transpose()
        
        mum=0
        for h in range(6,0,-1):
            mum = [th(h/10, x) for x in sol]
            if np.sum(np.array(mum))>0:
                break

        x["high_labels"] = mum

        dicts.append(x)

    clean_db(collection="task5")
    for x in dicts:
        insert_one(x, db="DSAP", collection="task5")
        
    print(len(names))

        
       



