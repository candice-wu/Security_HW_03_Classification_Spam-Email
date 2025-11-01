import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import time
import json

# ========== 設定檔案路徑 ==========
DATA_PATH = "sms_spam_no_header.csv"
MODELS_DIR = "models"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
VOCAB_PATH = "vocab.json"

# ========== 1️⃣ 載入資料 ==========
print("📂 檢查資料集...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ 找不到 {DATA_PATH}，請確認檔案已放在專案資料夾中！"
    )

print("✅ 找到資料集，開始載入...")
data = pd.read_csv(DATA_PATH, encoding="latin-1")

# 嘗試偵測欄位結構（因資料集可能無標題）
if len(data.columns) >= 2:
    data.columns = ["label", "message"] + list(data.columns[2:])
else:
    raise ValueError("資料格式錯誤，需至少有兩個欄位：label 與 message")

# 僅保留必要欄位
data = data[["label", "message"]]
print(f"📊 資料筆數：{len(data)}")

# ========== 2️⃣ 資料前處理 ==========
print("\n🧹 清理與轉換標籤...")
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# 檢查是否有缺失值
if data.isnull().sum().any():
    print("⚠️ 發現缺失值，將自動移除")
    data = data.dropna()

# ========== 3️⃣ 分割資料集 ==========
print("\n✂️ 分割訓練 / 測試資料...")
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

# ========== 4️⃣ 向量化文字 ==========
print("\n🔤 文字向量化中...")
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"✅ TF-IDF 向量化完成，特徵數量：{X_train_tfidf.shape[1]}")

# ========== 5️⃣ 建立與訓練多個模型 ==========
print("\n🤖 準備訓練多個模型...")

# 定義要訓練的模型
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 建立儲存模型的資料夾
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# 迴圈訓練、評估並儲存每個模型
for name, model in models.items():
    print(f"\n{'='*20}")
    print(f"🏋️‍♀️ 開始訓練：{name}")
    print(f"{'='*20}")
    
    start_time = time.time()
    
    # 訓練模型
    model.fit(X_train_tfidf, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 評估模型
    y_pred = model.predict(X_test_tfidf)
    
    print(f"\n📈 模型評估結果 ({name}):")
    print(f"準確率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"訓練耗時: {training_time:.2f} 秒")
    print("混淆矩陣：")
    print(confusion_matrix(y_test, y_pred))
    print("\n分類報告：")
    print(classification_report(y_test, y_pred))
    
    # 儲存模型
    model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 模型已儲存至: {model_path}")

# ========== 6️⃣ 儲存向量化工具與字彙表 ==========
print(f"\n{'='*20}")
print("🗂️ 儲存共用元件...")
print(f"{'='*20}")

# 儲存向量化工具
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"📁 已儲存向量器至： {VECTORIZER_PATH}")

# 儲存字彙表
vocab = vectorizer.get_feature_names_out()
vocab_dict = {word: idx for idx, word in enumerate(vocab)}
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
print(f"📁 已儲存字彙表至： {VOCAB_PATH}（共 {len(vocab_dict)} 個詞）")


print("\n🎉 所有模型訓練完成！")
print("🎉 可以執行 `streamlit run 5114050013_hw3.py` 開啟更新後的互動式網頁了！")