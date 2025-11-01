Streamlit Demo-site：https://hw03-classification-spam-email.streamlit.app

[補充]
1. 與 ChatGPT 對話記錄請參閱「人工智慧與資訊安全_HW03_與 chatGPT 對話記錄.pdf」
2. 與 AI Agent (Gemini CLI and openspec) 對話記錄請參閱 prompt 資料夾裡的 7 份 .txt files.

# 垃圾郵件分類器 Pro

## 專案結構

```
.
├── 5114050013_hw3.py         # 模型訓練腳本
├── AGENTS.md                 # 關於 AI Agent 的說明
├── HW03_吳佩玲_5114050013.pages # 作業文件 (Pages 格式)
├── HW03_吳佩玲_5114050013.pdf  # 作業文件 (PDF 格式)
├── README.md                 # 本檔案
├── requirements.txt          # Python 函式庫依賴列表
├── sms_spam_no_header.csv    # 原始資料集
├── spam_classifier_model.pkl # 訓練好的垃圾郵件分類模型 (單一模型)
├── streamlit_app.py          # Streamlit 應用程式主檔案
├── test_spam_classifier.py   # 測試腳本
├── tfidf_vectorizer.pkl      # TF-IDF 向量化工具
├── vocab.json                # 詞彙表
├── 人工智慧與資訊安全_HW03_與 chatGPT 對話記錄.pdf # 與 ChatGPT 對話記錄
├── models/                   # 儲存多個訓練好的模型
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   └── svm_model.pkl
└── prompt/                   # AI Agent 對話記錄
```

這是一個進階的垃圾郵件分類系統，不僅能分類訊息，還提供一個綜合平台來比較多種機器學習模型。它具有高度互動性的 Streamlit 網頁應用程式，可用於即時預測、批次分析、模型效能評估和探索性資料分析（EDA）。

專案遵循 CRISP-DM（Cross-Industry Standard Process for Data Mining）方法論。

## CRISP-DM 流程說明

### 1. 商業理解 (Business Understanding)

**目標：** 建立一個強大的系統，能夠準確地將訊息分類為「垃圾郵件（spam）」或「非垃圾郵件（ham）」，並評估和比較幾種不同分類演算法的效能，以找出最適合此任務的模型。

**成功標準：**
*   多種模型均有高準確率。
*   一個互動式且使用者友善的網頁介面，用於分析和預測。
*   清晰、並排的視覺化模型效能指標。

### 2. 資料理解 (Data Understanding)

**資料集：** 本專案使用 `sms_spam_no_header.csv` 資料集，這是一個標記為「ham」或「spam」的簡訊集合。

**主要特徵：**
*   `label`: 目標變數 (ham/spam)。
*   `message`: 訊息的原始文字內容。

### 3. 資料準備 (Data Preparation)

原始文字資料經過幾個預處理步驟：
*   **標籤編碼：** 將分類標籤「ham」和「spam」轉換為 0 和 1。
*   **文字向量化：** 使用 TF-IDF（詞頻-逆文件頻率）將訊息轉換為數值特徵向量，以捕捉每個詞的重要性。

### 4. 模型建立 (Modeling)

為了找到此分類問題的最佳方法，我們訓練並評估了四種不同的演算法：

*   **邏輯迴歸 (Logistic Regression):** 一個可靠的線性模型，可作為強大的基準。
*   **樸素貝氏 (Naive Bayes - Multinomial):** 用於文字分類的經典演算法，以其速度和效率著稱。
*   **支持向量機 (Support Vector Machine - SVM):** 一個強大的模型，在像文字資料這樣的高維空間中表現良好。
*   **隨機森林 (Random Forest):** 一種集成方法，結合多個決策樹以提高準確性並控制過擬合。

所有訓練好的模型都會被儲存，並可在 Streamlit 應用程式中選擇進行即時預測。

### 5. 模型評估 (Evaluation)

模型在保留的測試集上進行嚴格評估。Streamlit 應用程式設有專門的「模型比較」頁面，顯示每個模型的主要效能指標，包括：

*   **準確率、精確率、召回率和 F1-Score。**
*   **預測時間。**
*   **F1-Score 的長條圖**，以便快速視覺比較。
*   **ROC 曲線和 Precision-Recall 曲線**
*   **閾值掃描表記錄**

### 6. 部署 (Deployment)

該專案部署為一個互動式的 Streamlit 網頁應用程式，具有以下主要功能：

*   **即時預測：** 選擇任何一個訓練好的模型，並輸入自訂文字，以獲得即時分類和相關的垃圾郵件機率。
*   **批次預測：** 上傳一個包含訊息的 CSV 檔案，以對整個檔案進行預測，然後可以下載結果。
*   **模型比較：** 一個儀表板，用於並排比較所有訓練模型的效能指標。
*   **探索性資料分析 (EDA):** 一個互動頁面，用於探索資料集，包括：
    *   類別分佈的長條圖和圓餅圖。
    *   訊息長度分佈的密度圖。
    *   垃圾郵件和非垃圾郵件的文字雲，以視覺化最頻繁的詞彙。
    *   訊息長度的統計分析表。

## 如何部署到 Streamlit Cloud

1. Push 專案資料夾 to GitHub
2. 至 https://share.streamlit.io ，點擊 “Create app”
3. Repository：下拉選擇 candice-wu/Security_HW_03_Classification_Spam-Email
4. Branch：Main
5. Main file path：5114050013_hw3.py
6. App URL (optional)：預設可以維持，或改掉並另外命名
，如：hw03-classification-spam-email
7. 點擊 “Deploy” 即完成部署

## 如何執行專案

1.  **安裝依賴函式庫**
    ```bash
    pip install -r requirements.txt
    ```

2.  **訓練模型**
    執行訓練腳本。這將訓練所有四個模型，並將它們保存在 `models/` 目錄中。
    ```bash
    python train_model.py
    ```

3.  **啟動 Streamlit 應用程式**
    啟動網頁應用程式。
    ```bash
    streamlit run 5114050013_hw3.py
    ```

## 資料集參考
https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv