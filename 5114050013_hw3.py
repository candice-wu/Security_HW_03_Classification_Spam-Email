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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
from streamlit_option_menu import option_menu

# ========== 初始化設定 ==========
st.set_page_config(
    page_title="垃圾郵件分類器 Pro",
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
    """從 'models' 資料夾載入所有模型 (Robust version)"""
    models = {}
    model_filenames = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "SVM": "svm_model.pkl",
        "Random Forest": "random_forest_model.pkl"
    }

    for display_name, filename in model_filenames.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[display_name] = joblib.load(path)
        else:
            st.error(f"FATAL: Model file not found at {path}. Deployment is incomplete.")
            return None # Return None to trigger the main error message.
            
    if not models: # If the loop finishes but models dict is empty
        return None

    return models

# ========== 側邊選單 ==========
with st.sidebar:
    st.title("🤖 垃圾郵件分類器 Pro")
    selected = option_menu(
        menu_title="主選單",
        options=["即時預測", "批次預測", "模型比較", "探索性資料分析"],
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
    st.error("❌ 核心元件載入失敗！請先執行 `train_model.py` 訓練腳本，並確認模型檔案已上傳至 GitHub。")
    st.stop()

# ========== 頁面路由 ==========

# --- 1. 即時預測 (Live Prediction) ---
if selected == "即時預測":
    st.title("💬 即時文字預測")

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

    st.divider()
    st.subheader("📜 範例訊息")
    filter_option = st.radio("篩選訊息類別", ["All", "Ham", "Spam"])

    if filter_option == "All":
        st.dataframe(df.sample(10))
    else:
        st.dataframe(df[df['label'] == filter_option.lower()].sample(10))

# --- 2. 批次預測 (Batch Prediction) ---
elif selected == "批次預測":
    st.title("📤 使用 CSV 進行批次預測")
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
elif selected == "模型比較":
    st.title("📊 模型效能比較")
    st.markdown("調整下方的測試集比例，效能指標和圖表將會自動更新。")

    test_size_ratio = st.slider("**動態調整測試集比例**", 0.1, 0.5, 0.2, 0.05)

    @st.cache_data
    def get_performance_metrics(test_size):
        X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label_num"], test_size=test_size, random_state=42)
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
        return pd.DataFrame(performance), y_test, X_test_tfidf

    with st.spinner("動態評估中，請稍候..."):
        perf_df, y_test_split, X_test_tfidf_split = get_performance_metrics(test_size_ratio)

    st.subheader(f"📊 基於 {test_size_ratio:.0%} 測試資料的效能指標")
    st.dataframe(perf_df.set_index("Model"))

    st.subheader("🏆 F1-Score 比較圖")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=perf_df, x="F1-Score", y="Model", ax=ax, orient='h', palette="viridis")
    ax.set_xlabel("F1-Score")
    ax.set_ylabel("Model")
    ax.set_title("F1-Score by Model")
    st.pyplot(fig)
    
    st.divider()
    st.subheader("🤔 混淆矩陣 (Confusion Matrix) 詳細分析")
    cm_model_name = st.selectbox("**選擇一個模型以查看其混淆矩陣**", models.keys(), key="cm_select")
    
    if cm_model_name:
        model = models[cm_model_name]
        y_pred = model.predict(X_test_tfidf_split)
        cm = confusion_matrix(y_test_split, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix for {cm_model_name}")
        st.pyplot(fig)

        # ROC & Precision-Recall Curves
        y_probs = model.predict_proba(X_test_tfidf_split)[:, 1]
        
        col1, col2 = st.columns(2)
        with col1:
            fpr, tpr, _ = roc_curve(y_test_split, y_probs)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            st.pyplot(fig)

        with col2:
            precision, recall, _ = precision_recall_curve(y_test_split, y_probs)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc="lower left")
            st.pyplot(fig)

        # Threshold Sweep Table
        thresholds = np.arange(0.1, 1.0, 0.1)
        table_data = []
        for t in thresholds:
            y_pred_thresh = (y_probs >= t).astype(int)
            table_data.append({
                "Threshold": f"{t:.1f}",
                "Precision": precision_score(y_test_split, y_pred_thresh),
                "Recall": recall_score(y_test_split, y_pred_thresh),
                "F1-Score": f1_score(y_test_split, y_pred_thresh)
            })
        st.subheader("🎚️ Threshold Sweep (Precision/Recall/F1)")
        st.table(pd.DataFrame(table_data).set_index("Threshold"))


# --- 4. 探索性資料分析 (EDA) ---
elif selected == "探索性資料分析":
    st.title("📈 探索性資料分析 (EDA)")

    # --- Data Overview ---
    st.subheader("📝 資料總覽")
    df['message_len'] = df['message'].apply(len)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("總訊息數", len(df))
        ham_skew = df[df['label'] == 'ham']['message_len'].skew()
        st.metric("正常郵件長度偏度", f"{ham_skew:.2f}")
    with col2:
        st.metric("垃圾郵件與正常郵件比例", f"{df['label'].value_counts()['spam']} / {df['label'].value_counts()['ham']}")
        spam_skew = df[df['label'] == 'spam']['message_len'].skew()
        st.metric("垃圾郵件長度偏度", f"{spam_skew:.2f}")

    st.subheader("🔢 訊息長度統計分析")
    st.dataframe(df.groupby('label')['message_len'].describe())

    # --- Class Distribution ---
    st.subheader("📊 類別分佈")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(data=df, x='label', ax=ax, palette=['skyblue', 'salmon'])
    ax.set_title("Class Distribution (Bar)")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    df["label"].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    ax.set_ylabel('')
    ax.set_title("Spam vs. Ham")
    st.pyplot(fig)

    # --- Message Length Distribution ---
    st.subheader("📏 訊息長度分佈")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(data=df, x='message_len', hue='label', fill=True, common_norm=False, palette=["skyblue", "salmon"])
    ax.set_title("Density Plot of Message Length by Class")
    st.pyplot(fig)

    # --- Top Tokens by Class ---
    st.subheader("🔥 各類別熱門詞彙")
    top_n = st.slider("選擇 Top-N 詞彙", 5, 25, 10)

    @st.cache_data
    def get_top_tokens(corpus, n=10):
        vec = vectorizer.fit_transform(corpus)
        sum_words = vec.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    ham_corpus = df[df['label'] == 'ham']['message']
    spam_corpus = df[df['label'] == 'spam']['message']

    top_ham_tokens = get_top_tokens(ham_corpus, top_n)
    top_spam_tokens = get_top_tokens(spam_corpus, top_n)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 熱門正常詞彙")
        ham_df = pd.DataFrame(top_ham_tokens, columns=['Token', 'Frequency'])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=ham_df, x='Frequency', y='Token', ax=ax, palette='Greens_r')
        st.pyplot(fig)
    with col2:
        st.markdown("#### 熱門垃圾詞彙")
        spam_df = pd.DataFrame(top_spam_tokens, columns=['Token', 'Frequency'])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=spam_df, x='Frequency', y='Token', ax=ax, palette='Reds_r')
        st.pyplot(fig)

    # --- Word Clouds ---
    st.subheader("☁️ 文字雲")
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