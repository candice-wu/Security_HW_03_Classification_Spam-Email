import pytest
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import os
import subprocess

# 確保 nltk 資源可用
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ========== 共用設定 ==========
MODEL_PATH = "spam_classifier_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DATA_PATH = "sms_spam_no_header.csv"
VOCAB_PATH = "vocab.json"

# ========== 工具函式 ==========
def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# ========== 測試前置 (fixture) ==========
@pytest.fixture(scope="module")
def model_and_vectorizer():
    # 若模型不存在，自動重新訓練
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        subprocess.run(["python", "5114050013_hw3.py"], check=True)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

@pytest.fixture(scope="module")
def sample_data():
    data = {
        "label": ["ham", "spam", "ham", "spam"],
        "message": [
            "Hey, are you free today to grab lunch?",
            "Congratulations! You've won a free ticket to Bahamas. Text WIN to 12345!",
            "I'll see you at the meeting tomorrow.",
            "Urgent! Your mobile number has won $5000. Call 090123456 now!"
        ],
    }
    df = pd.DataFrame(data)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

# ========== 單元測試 ==========
def test_stem_text():
    text = "running runs runner"
    expected = "run run runner"
    assert stem_text(text) == expected

def test_vectorizer_output(model_and_vectorizer, sample_data):
    _, vectorizer = model_and_vectorizer
    transformed = vectorizer.transform(sample_data["message"].apply(stem_text))
    assert transformed.shape[0] == len(sample_data)
    assert transformed.shape[1] > 0

def test_model_prediction(model_and_vectorizer, sample_data):
    model, vectorizer = model_and_vectorizer
    transformed = vectorizer.transform(sample_data["message"].apply(stem_text))
    preds = model.predict(transformed)
    # 預期 spam/ham 預測應合理
    assert len(preds) == len(sample_data)
    assert set(preds).issubset({0, 1})

def test_model_accuracy(model_and_vectorizer, sample_data):
    model, vectorizer = model_and_vectorizer
    transformed = vectorizer.transform(sample_data["message"].apply(stem_text))
    preds = model.predict(transformed)
    acc = accuracy_score(sample_data["label"], preds)
    assert acc >= 0.75  # 基本合理閾值

def test_vocab_file_exists():
    """檢查 vocab.json 是否存在且可解析"""
    assert os.path.exists(VOCAB_PATH), "❌ vocab.json 檔案不存在"
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, dict)
    assert len(vocab) > 0, "⚠️ vocab.json 檔案為空"
