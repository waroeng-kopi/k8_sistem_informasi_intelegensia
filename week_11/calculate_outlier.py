import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Muat dataset
df = pd.read_csv('Crop_recommendation.csv')

# 1. Pastikan hanya kolom numerik yang diproses
numeric_cols = df.select_dtypes(include=['number']).columns
print("Kolom numerik yang akan diproses:", list(numeric_cols))

# 2. Visualisasi sebelum penanganan outlier (hanya kolom numerik)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title('Distribusi Data Sebelum Penanganan Outlier')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplot_before_outlier.pdf', format='pdf')
plt.close()

# 3. Hitung IQR hanya untuk kolom numerik
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# 4. Filter outlier (pertahankan kolom non-numerik)
mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
df_clean = df[mask]

# 5. Visualisasi setelah penanganan outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[numeric_cols])
plt.title('Distribusi Data Setelah Penanganan Outlier')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplot_after_outlier.pdf', format='pdf')
plt.close()

# 6. Hasil statistik
print(f"\nJumlah sampel sebelum penanganan outlier: {len(df)}")
print(f"Jumlah sampel setelah penanganan outlier: {len(df_clean)}")
print(f"Persentase data yang dihapus: {(len(df)-len(df_clean))/len(df)*100:.2f}%")

print(f"\n\n")
print(f"###################################")
print(f"\n\n")

from sklearn.preprocessing import StandardScaler

# Pisahkan fitur dan label
X = df_clean.drop(columns=['label'])
y = df_clean['label']

# Inisialisasi scaler
scaler = StandardScaler()

# Transformasi fitur
X_scaled = scaler.fit_transform(X)

# Konversi ke DataFrame untuk inspeksi
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled_df.describe().loc[['mean', 'std']])

print(f"\n\n")
print(f"###################################")
print(f"\n\n")

from sklearn.preprocessing import LabelEncoder

# Inisialisasi encoder
le = LabelEncoder()

# Transformasi label
y_encoded = le.fit_transform(y)

# Cek hasil encoding
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Pemetaan label tanaman:")
for crop, code in label_mapping.items():
    print(f"{crop}: {code}")
    
print(f"\n\n")
print(f"###################################")
print(f"\n\n")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Pembagian data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Fungsi evaluasi komprehensif
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Latih model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik dasar
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Hitung ROC-AUC jika memungkinkan
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    # Validasi silang
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')
    metrics['CV Accuracy'] = cv_scores.mean()
    
    return metrics

print(f"\n\n")
print(f"###################################")
print(f"\n\n")

# Import semua library yang diperlukan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset (pastikan path sesuai)
df = pd.read_csv('Crop_recommendation.csv')

# Praproses data (contoh sederhana)
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definisi parameter grid untuk KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Inisialisasi KNN
knn = KNeighborsClassifier()

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,  # 5-fold cross validation
    scoring='accuracy',
    n_jobs=-1  # Gunakan semua core CPU
)

# Jalankan grid search
grid_search.fit(X_train_scaled, y_train)

# Hasil optimasi
print("\n=== Hasil Optimasi KNN ===")
print("Parameter terbaik:", grid_search.best_params_)
print("Akurasi terbaik (CV):", grid_search.best_score_)
print("\nEvaluasi pada test set:")
best_knn = grid_search.best_estimator_
test_accuracy = best_knn.score(X_test_scaled, y_test)
print(f"Akurasi pada data test: {test_accuracy:.4f}")

# Simpan model terbaik jika diperlukan
import joblib
joblib.dump(best_knn, 'best_knn_model.pkl')

print(f"\n\n")
print(f"###################################")
print(f"\n\n")

# Import library yang diperlukan
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load Data dan Model
try:
    # Load dataset
    df = pd.read_csv('Crop_recommendation.csv')
    
    # Pisahkan fitur dan target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Load model yang sudah disimpan
    artifacts = joblib.load('crop_recommender.joblib')
    best_model = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['encoder']
    
except FileNotFoundError as e:
    print(f"Error: {e}. Pastikan file dataset dan model ada di direktori yang sama.")
    exit()

# 2. Preprocessing dan Split Data
X_scaled = scaler.transform(X)
y_encoded = le.transform(y)

# Split data (gunakan random_state untuk reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. Visualisasi Confusion Matrix
plt.figure(figsize=(15, 12), dpi=100)
ConfusionMatrixDisplay.from_estimator(
    best_model, 
    X_test, 
    y_test,
    display_labels=le.classes_,
    xticks_rotation=90,
    cmap='Blues',
    colorbar=False
)
plt.title('Confusion Matrix - Model Terbaik', pad=20, fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Visualisasi Feature Importance (Hanya untuk model yang memiliki feature_importances_)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6), dpi=100)
    importances = best_model.feature_importances_
    feature_names = X.columns
    sorted_idx = importances.argsort()
    
    # Warna berbeda untuk importance tertinggi
    colors = ['lightblue' if x < max(importances) else 'steelblue' for x in importances[sorted_idx]]
    
    plt.barh(
        range(len(sorted_idx)), 
        importances[sorted_idx], 
        align='center',
        color=colors
    )
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance - Model Terbaik', pad=15, fontsize=14)
    
    # Tambah nilai importance di setiap bar
    for i, v in enumerate(importances[sorted_idx]):
        plt.text(v + 0.005, i, f"{v:.3f}", color='black', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.pdf', bbox_inches='tight')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("Model tidak memiliki feature_importances_")

print("Visualisasi berhasil disimpan sebagai:")
print("- confusion_matrix.pdf/png")
print("- feature_importance.pdf/png")

print(f"\n\n")
print(f"###################################")
print(f"\n\n")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 1. Load and preprocess data
df = pd.read_csv('Crop_recommendation.csv')

# Handle outliers
numeric_cols = df.select_dtypes(include=['number']).columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# 2. Prepare features and target
X = df_clean.drop('label', axis=1)
y = df_clean['label']

# 3. Create and save preprocessing objects
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Train and save the model (using RandomForest as example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_encoded)

# 5. Save all artifacts
joblib.dump({
    'model': model,
    'scaler': scaler,
    'encoder': le
}, 'crop_recommender.joblib')

print("Model and preprocessing artifacts saved successfully!")

print(f"\n\n")
print(f"###################################")
print(f"\n\n")

import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# 1. Load Model dan Preprocessing Artifacts
try:
    artifacts = joblib.load('crop_recommender.joblib')
    best_model = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['encoder']
except FileNotFoundError:
    print("Error: File model tidak ditemukan. Pastikan 'crop_recommender.joblib' ada di direktori yang sama.")
    exit()

# 2. Inisialisasi Flask App
app = Flask(__name__)

# 3. Endpoint Prediksi
@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        # Validasi input
        if not request.json:
            return jsonify({"error": "Request harus dalam format JSON"}), 400
        
        # Data input wajib
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(field in request.json for field in required_fields):
            missing = [f for f in required_fields if f not in request.json]
            return jsonify({"error": f"Field yang diperlukan: {missing}"}), 400

        # Konversi input ke numpy array
        input_data = np.array([[
            float(request.json['N']),
            float(request.json['P']),
            float(request.json['K']),
            float(request.json['temperature']),
            float(request.json['humidity']),
            float(request.json['ph']),
            float(request.json['rainfall'])
        ]])

        # Preprocessing
        scaled_input = scaler.transform(input_data)

        # Prediksi
        prediction = best_model.predict(scaled_input)
        crop_name = le.inverse_transform(prediction)[0]

        # Hitung confidence (jika model mendukung predict_proba)
        if hasattr(best_model, "predict_proba"):
            confidence = float(best_model.predict_proba(scaled_input).max())
        else:
            confidence = 1.0  # Default confidence untuk model tanpa probabilitas

        # Response
        return jsonify({
            "recommended_crop": crop_name,
            "confidence": confidence,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

# 4. Health Check Endpoint
@app.route('/')
def health_check():
    return jsonify({
        "status": "ready",
        "message": "Crop Recommendation API is running"
    })

# 5. Jalankan Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)