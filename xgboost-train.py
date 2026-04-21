import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Çakışma önleyici ve CPU optimizasyon ayarları
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. DB BAĞLANTISI
# Bilgilerini kendi DB'ne göre güncelle kanka
engine = create_engine('postgresql://postgres:yeni_sifre@localhost:5432/travelsafe')

def fetch_and_process():
    print("Veriler PostgreSQL'den çekiliyor...")
    # Çok eski veriler 2026 tahmini için gürültü yapabilir, 2010 sonrası genelde yeterlidir
    query = """
    SELECT "Date", "crime_group", "Latitude", "Longitude" 
    FROM crime 
    WHERE "Latitude" IS NOT NULL AND "Longitude" IS NOT NULL
    AND EXTRACT(YEAR FROM "Date") >= 2010
    """
    df = pd.read_sql(query, engine)
    
    # --- 2. SPATIAL AGGREGATION (500m Grid) ---
    # 0.0045 derece enlem yaklaşık 500 metreye tekabül eder
    grid_size = 0.0045 
    df['grid_lat'] = (df['Latitude'] / grid_size).astype(int)
    df['grid_lon'] = (df['Longitude'] / grid_size).astype(int)
    # Benzersiz Grid ID oluşturma
    df['grid_id'] = df.groupby(['grid_lat', 'grid_lon']).ngroup()

    # --- 3. TEMPORAL AGGREGATION ---
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Hour'] = df['Date'].dt.hour
    
    # 6 Saatlik Zaman Slotları
    df['time_slot'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], 
                             labels=[0, 1, 2, 3], include_lowest=True).astype(int)
    
    # Mevsim: 0:Kış, 1:Bahar, 2:Yaz, 3:Güz
    df['Season'] = df['Month'].apply(lambda x: 0 if x in [12,1,2] else 1 if x in [3,4,5] else 2 if x in [6,7,8] else 3)
    
    return df

df = fetch_and_process()

# --- SINIF DENGELEME (BALANCING) ---
print("Sınıflar dengeleniyor (Downsampling)...")
# Her gruptan maksimum 100.000 örnek alarak baskın sınıfın (MALA_KARSI) etkisini kırıyoruz
min_sample_size = 100000 
balanced_list = []

for group in df['crime_group'].unique():
    subset = df[df['crime_group'] == group]
    if len(subset) > min_sample_size:
        # Rastgele örnekleme yaparak sayıyı düşür
        subset = subset.sample(n=min_sample_size, random_state=42)
    balanced_list.append(subset)

df = pd.concat(balanced_list)
print(f"Yeni veri seti boyutu: {len(df)} satır.")

# --- 4. HEDEF VE AĞIRLIKLANDIRMA ---
# En popüler 10 suçu tahmin edelim (Multiclass)
top_crimes = df['crime_group'].value_counts().nlargest(10).index
df = df[df['crime_group'].isin(top_crimes)]

le = LabelEncoder()
df['target'] = le.fit_transform(df['crime_group'])

# TIME-DECAY: 2026 yılına olan uzaklığa göre ağırlık veriyoruz
# 2026'ya yaklaştıkça değer 1'e yaklaşır.
df['sample_weight'] = np.exp(-0.08 * (2026 - df['Year']))

# --- 5. EĞİTİM HAZIRLIĞI ---
features = ['grid_id', 'Year', 'Month', 'DayOfWeek', 'time_slot', 'Season', 'Latitude', 'Longitude']
X = df[features]
y = df['target']
weights = df['sample_weight']

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# --- 6. XGBOOST MODELİ ---
print(f"\n500m Grid Modeli eğitiliyor... Toplam Sınıf: {len(le.classes_)}")
model = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    tree_method='hist', # Büyük veri için hızlandırma
    eval_metric='mlogloss'
)

model.fit(X_train, y_train, sample_weight=w_train)

# --- 7. METRİKLER VE GÖRSELLEŞTİRME ---
y_pred = model.predict(X_test)
print(f"\nModel Başarı Skoru: {accuracy_score(y_test, y_pred)}")
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Feature Importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', color='#008080')
plt.title("500m Grid + Time-Decay Model Önem Dereceleri")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()