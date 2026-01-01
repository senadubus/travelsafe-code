import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

cred = credentials.Certificate("keykey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

MAX_RECORDS = 20000
data = []

query = (
    db.collection_group("Crime")
      .limit(MAX_RECORDS)
)

docs = query.stream()

for doc in docs:
    d = doc.to_dict()
    data.append({
        "description": d.get("Description"),
        "primary_type": d.get("Primary Type"),
        "fbi_code": d.get("FBI Code"),
        "iucr": d.get("IUCR"),
        "domestic": d.get("Domestic")
    })

df = pd.DataFrame(data)

print("Okunan kayıt:", len(df))
print(df.head())
