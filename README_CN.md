# Spam Classifier Pro (垃圾郵件分類器專業版)

## Streamlit 應用展示
https://hw03-classification-spam-email.streamlit.app

## 專案介紹 (CRISP-DM 方法論)

本專案實作了一個進階的垃圾郵件分類系統，它不僅能分類訊息，更提供一個全面的平台來比較多種機器學習模型。系統的核心是一個高度互動的 Streamlit 網站應用，支援即時預測、批次分析、模型效能評估與探索性資料分析 (EDA)。

本專案遵循跨產業資料探勘標準流程 (CRISP-DM) 方法論。

### 1. 商業理解

**目標：** 建立一個強健的系統，不僅能準確地將訊息分類為「垃圾郵件 (spam)」或「正常郵件 (ham)」，更能評估與比較多種不同分類演算法的效能，以找出最適合此任務的模型。

**成功標準：**
*   多種模型均達到高準確度。
*   提供一個互動且友善的網站介面，以進行分析與預測。
*   清晰、可並列比較的多模型效能指標視覺化。

### 2. 資料理解

**資料集：** 本專案使用 `sms_spam_no_header.csv` 資料集，這是一個包含標記為「正常郵件」或「垃圾郵件」的 SMS 簡訊集合。

**主要特徵：**
*   `label`: 目標變數 (ham/spam)。
*   `message`: 訊息的原始文字內容。

### 3. 資料準備

原始文字資料經過以下幾個預處理步驟：
*   **標籤編碼：** 將分類標籤「ham」和「spam」轉換為數字 0 和 1。
*   **文字向量化：** 使用 TF-IDF (詞頻-逆向文件頻率) 將文字訊息轉換為數值特徵向量，此方法能捕捉每個詞的重要性。

### 4. 模型建構

為了找出解決此分類問題的最佳方法，我們訓練並評估了四種不同的演算法：

*   **邏輯回歸 (Logistic Regression):** 一個可靠的線性模型，可作為強大的效能基準。
*   **樸素貝氏分類器 (Naive Bayes):** 文字分類的經典演算法，以其速度與效率著稱。
*   **支援向量機 (SVM):** 一個強大的模型，在像文字這樣的高維度空間中表現良好。
*   **隨機森林 (Random Forest):** 一種集成學習方法，結合多個決策樹以提升準確度並抑制過擬合。

所有訓練好的模型都會被儲存，並可在 Streamlit 應用中選擇用於即時預測。

### 5. 評估

所有模型都在一個獨立的測試集上進行嚴格評估。Streamlit 應用程式中有一個專門的 **「模型比較 (Model Comparison)」** 頁面，用以展示各模型的關鍵效能指標，包含：

*   **準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall) 與 F1 分數 (F1-Score)。**
*   **預測耗時 (Prediction Time)。**
*   一個長條圖，用於快速視覺化比較各模型的 F1 分數。

### 6. 部署

本專案以一個互動式的 Streamlit 網站應用程式進行部署，具備以下主要功能：

*   **即時預測 (Live Prediction):** 選擇任一已訓練的模型，並輸入自訂文字，即可獲得即時的分類結果與相應的垃圾郵件機率。
*   **批次預測 (Batch Prediction):** 上傳一個包含訊息的 CSV 檔案，即可對整個檔案進行預測，並可下載預測結果。
*   **模型比較 (Model Comparison):** 一個儀表板，可並列比較所有已訓練模型的效能指標。
*   **探索性資料分析 (EDA):** 一個互動頁面，用於探索資料集，包含：
    *   類別分佈的圓餅圖。
    *   訊息長度的分佈直方圖。
    *   正常郵件與垃圾郵件的文字雲，以視覺化最頻繁出現的詞彙。

## 如何執行專案

1.  **設定環境:**
    建議使用虛擬環境。
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # 在 Windows 上，請使用 `venv\Scripts\activate`
    ```

2.  **安裝相依套件:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **訓練模型:**
    執行訓練腳本。這將會訓練所有四個模型，並將它們儲存在 `models/` 目錄中。
    ```bash
    python 5114050013_hw3.py
    ```

4.  **執行 Streamlit 應用:**
    啟動網站應用程式。
    ```bash
    streamlit run streamlit_app.py
    ```

## 資料集參考
https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv