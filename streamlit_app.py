import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from streamlit_option_menu import option_menu

# ========== 初始化設定 ==========
st.set_page_config(
    page_title="Spam Classifier Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 全域路徑設定 ==========
MODELS_DIR = "models"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DATA_PATH = "sms_spam_no_header.csv"

# ========== 資源載入函式 (快取) ==========
@st.cache_data
def load_data():
    """載入並預處理資料集"""
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ 找不到資料集 {DATA_PATH}！")
        return None
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    if len(df.columns) >= 2:
        df.columns = ["label", "message"] + list(df.columns[2:])
        df = df[["label", "message"]]
        df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    else:
        st.error("⚠️ CSV 結構不符，需包含 label, message 欄位。")
        return None
    return df

@st.cache_resource
def load_vectorizer():
    """載入 TF-IDF 向量化工具"""
    if not os.path.exists(VECTORIZER_PATH):
        return None
    return joblib.load(VECTORIZER_PATH)

@st.cache_resource
def load_models():
    """從 'models' 資料夾載入所有模型"""
    models = {}
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if not model_files:
        return None
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace("_model.pkl", "").replace("_", " ").title()
        models[model_name] = joblib.load(model_path)
    return models

# ========== 側邊選單 ==========
with st.sidebar:
    st.title(" Spam Classifier Pro")
    selected = option_menu(
        menu_title="主選單",
        options=["Live Prediction", "Batch Prediction", "Model Comparison", "Exploratory Data Analysis"],
        icons=["cpu", "upload", "bar-chart-steps", "pie-chart"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.info("""專案已升級！現在您可以：
- 即時測試多種模型
- 比較各模型效能
- 進行深入的資料分析""")

# ========== 載入核心元件 ==========
df = load_data()
vectorizer = load_vectorizer()
models = load_models()

if df is None or vectorizer is None or models is None:
    st.error("❌ 核心元件載入失敗！請先執行 `5114050013_hw3.py` 訓練腳本。")
    st.stop()

# ========== 頁面路由 ==========

# --- 1. 即時預測 (Live Prediction) ---
if selected == "Live Prediction":
    st.title("💬 Live Text Prediction")

    # 處理回饋訊息
    if st.session_state.get('feedback_success', False):
        st.balloons()
        st.success("太棒了！感謝您的正面回饋。")
        st.session_state.feedback_success = False # 顯示一次後重設

    st.markdown("從下拉選單中選擇一個模型，然後輸入您想測試的訊息。")

    # 初始化 session state
    if 'message_input' not in st.session_state:
        st.session_state.message_input = "Congratulations! You've won a $1,000 gift card. Click here to claim."

    def set_example_text(label):
        """Callback function to update the text area's content."""
        sample = df[df['label'] == label].sample(1).iloc[0]
        st.session_state.message_input = sample['message']

    model_name = st.selectbox("**選擇模型**", models.keys())
    
    # 將 text_area 直接綁定到 session_state
    st.text_area("**輸入要分類的訊息：**", height=150, key="message_input")

    col1, col2 = st.columns(2)
    with col1:
        st.button("🎲 載入正常郵件範例 (Ham)", on_click=set_example_text, args=('ham',), use_container_width=True)
    with col2:
        st.button("🎲 載入垃圾郵件範例 (Spam)", on_click=set_example_text, args=('spam',), use_container_width=True)

    if st.button("🔍 分析訊息", use_container_width=True, type="primary"):
        # 從 session_state 讀取要分析的訊息
        message_to_analyze = st.session_state.message_input
        if message_to_analyze:
            model = models[model_name]
            message_tfidf = vectorizer.transform([message_to_analyze])
            
            prediction = model.predict(message_tfidf)[0]
            probability = model.predict_proba(message_tfidf)[0]
            
            # 將結果也存入 session_state 以便在頁面刷新後顯示
            st.session_state.is_spam = (prediction == 1)
            st.session_state.model_name = model_name
            st.session_state.probability = probability[1]
            st.session_state.show_result = True
        else:
            st.warning("請輸入訊息以進行分析。")
            st.session_state.show_result = False

    # 根據 session_state 顯示結果，確保頁面刷新後結果依然存在
    if st.session_state.get('show_result', False):
        if st.session_state.is_spam:
            st.error(f"**預測結果：這是垃圾訊息 (Spam)**", icon="🚨")
        else:
            st.success(f"**預測結果：這不是垃圾訊息 (Ham)**", icon="✅")
        
        st.metric(label=f"模型 `{st.session_state.model_name}` 預測為 Spam 的機率", value=f"{st.session_state.probability:.2%}")

        st.divider()
        st.markdown("#### 這個預測準確嗎？")
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.button("👍 預測正確", use_container_width=True):
                st.session_state.feedback_success = True
                st.rerun() # 使用 rerun 確保氣球特效能正確觸發
        with feedback_col2:
            if st.button("👎 預測錯誤", use_container_width=True):
                st.toast("喔不！感謝您的回饋，我們會參考這項資訊來改進模型。")
                st.info("您的回饋對於提升模型準確度至關重要！")

# --- 2. 批次預測 (Batch Prediction) ---
elif selected == "Batch Prediction":
    st.title("📤 Batch Prediction with CSV")
    st.markdown("上傳包含 `message` 欄位的 CSV 檔案，選擇模型後即可進行整批預測。")

    model_name = st.selectbox("選擇模型", models.keys())
    
    uploaded_file = st.file_uploader("上傳您的 CSV 檔案", type=["csv"])

    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)
        if "message" in test_df.columns:
            with st.spinner("🤖 正在進行批次預測..."):
                model = models[model_name]
                X_test = vectorizer.transform(test_df["message"])
                
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]
                
                result_df = test_df.copy()
                result_df["prediction"] = ["Spam" if p == 1 else "Ham" for p in preds]
                result_df["spam_probability"] = probs
                
                st.success("✅ 預測完成！")
                st.dataframe(result_df)
                
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(result_df)
                st.download_button(
                    label="📥 下載預測結果",
                    data=csv,
                    file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
        else:
            st.error("上傳的 CSV 檔案中必須包含 `message` 欄位。")

# --- 3. 模型比較 (Model Comparison) ---
elif selected == "Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown("調整下方的測試集比例，效能指標和圖表將會自動更新。")

    test_size_ratio = st.slider("**動態調整測試集比例 (Test Set Ratio)**", 0.1, 0.5, 0.2, 0.05)

    @st.cache_data
    def get_performance_metrics(test_size):
        X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label_num"], test_size=test_size, random_state=42)
        # 注意：向量化工具應在訓練集上 fit，然後在測試集上 transform
        # 為簡化此處的動態展示，我們重複使用已訓練好的 vectorizer
        # 在真實專案中，應在每次重切資料時重新 fit vectorizer
        X_test_tfidf = vectorizer.transform(X_test)
        
        performance = []
        for name, model in models.items():
            start_time = time.time()
            y_pred = model.predict(X_test_tfidf)
            end_time = time.time()
            
            performance.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "Prediction Time (s)": end_time - start_time,
            })
        # 為了混淆矩陣，我們需要回傳 y_test 和 X_test_tfidf
        return pd.DataFrame(performance), y_test, X_test_tfidf

    with st.spinner("動態評估中，請稍候..."):
        perf_df, y_test_split, X_test_tfidf_split = get_performance_metrics(test_size_ratio)

    st.subheader(f"基於 {test_size_ratio:.0%} 測試資料的效能指標")
    st.dataframe(perf_df.set_index("Model"))

    st.subheader("F1-Score 比較圖")
    fig, ax = plt.subplots()
    sns.barplot(data=perf_df, x="F1-Score", y="Model", ax=ax, orient='h', palette="viridis")
    ax.set_xlabel("F1-Score")
    ax.set_ylabel("Model")
    ax.set_title("F1-Score by Model")
    st.pyplot(fig)
    
    st.divider()
    st.subheader("混淆矩陣 (Confusion Matrix) 詳細分析")
    cm_model_name = st.selectbox("**選擇一個模型以查看其混淆矩陣**", models.keys(), key="cm_select")
    
    if cm_model_name:
        model = models[cm_model_name]
        y_pred = model.predict(X_test_tfidf_split)
        cm = confusion_matrix(y_test_split, y_pred)
        
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix for {cm_model_name}")
        st.pyplot(fig)

# --- 4. 探索性資料分析 (EDA) ---
elif selected == "Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis (EDA)")

    # 類別分佈
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    df["label"].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    ax.set_ylabel('')
    ax.set_title("Spam vs. Ham")
    st.pyplot(fig)

    # 訊息長度分佈
    st.subheader("Message Length Distribution")
    df['message_len'] = df['message'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='message_len', hue='label', multiple='stack', bins=50, ax=ax)
    ax.set_title("Distribution of Message Length")
    st.pyplot(fig)

    # 文字雲
    st.subheader("Word Clouds")
    max_words = st.slider("**選擇文字雲中的最大詞彙數量**", 50, 500, 200, 50)
    
    @st.cache_data
    def generate_wordcloud(text_series, max_words_wc):
        text = " ".join(msg for msg in text_series)
        wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=max_words_wc).generate(text)
        return wordcloud

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Ham (Not Spam)")
        with st.spinner("正在生成 'Ham' 文字雲..."):
            ham_wc = generate_wordcloud(df[df["label"] == "ham"]["message"], max_words)
            fig, ax = plt.subplots()
            ax.imshow(ham_wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
    with col2:
        st.markdown("#### Spam")
        with st.spinner("正在生成 'Spam' 文字雲..."):
            spam_wc = generate_wordcloud(df[df["label"] == "spam"]["message"], max_words)
            fig, ax = plt.subplots()
            ax.imshow(spam_wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)