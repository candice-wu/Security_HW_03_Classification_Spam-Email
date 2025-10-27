# Spam Classifier Pro

## Streamlit Demo
[Placeholder for Streamlit Cloud Deployment Link]

## Project Introduction (CRISP-DM Methodology)

This project implements an advanced spam classification system that not only classifies messages but also provides a comprehensive platform for comparing multiple machine learning models. It features a highly interactive Streamlit web application for real-time prediction, batch analysis, model performance evaluation, and exploratory data analysis (EDA).

The project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology.

### 1. Business Understanding

**Objective:** To build a robust system to accurately classify messages as "spam" or "ham" and to evaluate and compare the performance of several different classification algorithms to identify the most effective models for this task.

**Success Criteria:**
*   High accuracy across multiple models.
*   An interactive and user-friendly web interface for analysis and prediction.
*   Clear, side-by-side visualizations of model performance metrics.

### 2. Data Understanding

**Dataset:** The project uses the `sms_spam_no_header.csv` dataset, a collection of SMS messages labeled as "ham" or "spam".

**Key Features:**
*   `label`: The target variable (ham/spam).
*   `message`: The raw text content of the message.

### 3. Data Preparation

The raw text data undergoes several preprocessing steps:
*   **Label Encoding:** The categorical labels "ham" and "spam" are converted to 0 and 1.
*   **Text Vectorization:** Messages are transformed into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency), which captures the importance of each word.

### 4. Modeling

To find the best approach for this classification problem, four different algorithms are trained and evaluated:

*   **Logistic Regression:** A reliable linear model that serves as a strong baseline.
*   **Naive Bayes (Multinomial):** A classic algorithm for text classification, known for its speed and efficiency.
*   **Support Vector Machine (SVM):** A powerful model that works well in high-dimensional spaces like text data.
*   **Random Forest:** An ensemble method that combines multiple decision trees to improve accuracy and control overfitting.

All trained models are saved and can be selected for live prediction in the Streamlit app.

### 5. Evaluation

The models are rigorously evaluated on a held-out test set. The Streamlit application features a dedicated **"Model Comparison"** page that displays key performance metrics for each model, including:

*   **Accuracy, Precision, Recall, and F1-Score.**
*   **Prediction Time.**
*   A bar chart for quick visual comparison of F1-scores.

### 6. Deployment

The project is deployed as an interactive Streamlit web application with the following key features:

*   **Live Prediction:** Select any of the trained models and input custom text to get an instant classification and the associated spam probability.
*   **Batch Prediction:** Upload a CSV file with messages to get predictions for the entire file, which can then be downloaded.
*   **Model Comparison:** A dashboard to compare the performance metrics of all trained models side-by-side.
*   **Exploratory Data Analysis (EDA):** An interactive page to explore the dataset, featuring:
    *   Class distribution pie chart.
    *   Message length distribution histograms.
    *   Word clouds for both spam and ham messages to visualize the most frequent terms.

## How to Run the Project

1.  **Set up Environment:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train Models:**
    Run the training script. This will train all four models and save them in the `models/` directory.
    ```bash
    python 5114050013_hw3.py
    ```

4.  **Run Streamlit App:**
    Launch the web application.
    ```bash
    streamlit run streamlit_app.py
    ```

## Datasets Reference
https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv# Security_HW_03_Classification_Spam-Email
