import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

engine = create_engine('postgresql://postgres:yeni_sifre@localhost:5432/travelsafe')

query = 'SELECT "Primary Type", "Latitude", "Longitude", "cell_id" FROM crime '
df = pd.read_sql(query, engine)

le = LabelEncoder()
y = le.fit_transform(df['Primary Type'])
X = df[['Latitude', 'Longitude', 'cell_id']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Random Forest Eğitiliyor...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10)
rf.fit(X_train, y_train)

print(f"Random Forest Skoru: {accuracy_score(y_test, rf.predict(X_test))}")