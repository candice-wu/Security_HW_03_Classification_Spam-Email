## Context
This document outlines the design for a major upgrade to the "Spam Email Classifier" project. The project is evolving from a single-model predictor into a comprehensive, interactive platform for analyzing spam data and comparing the performance of multiple machine learning models. The core of the project is a multi-page Streamlit application.

## Goals / Non-Goals
- **Goals:**
    - Implement and compare four different classification models.
    - Develop a highly interactive, multi-page Streamlit application for analysis and prediction.
    - Provide clear, data-driven visualizations for both exploratory data analysis (EDA) and model performance comparison.
    - Maintain a clean, decoupled architecture between model training and the web application.
- **Non-Goals:**
    - Real-time email inbox integration.
    - Automatic model re-training based on user feedback.

## Decisions

### Architecture Overview
The system is composed of two primary components:
1.  **A Training Script (`5114050013_hw3.py`):** Responsible for all data preprocessing, model training, and serialization.
2.  **A Streamlit Application (`streamlit_app.py`):** A web-based UI for loading the trained models and providing all user-facing features.

---

### 1. Training Script (`5114050013_hw3.py`)

-   **Input:** The `sms_spam_no_header.csv` dataset.
-   **Process:**
    1.  **Load Data:** Reads the CSV into a pandas DataFrame.
    2.  **Preprocess:** Encodes labels and splits the data into training and testing sets.
    3.  **Vectorize:** Uses `TfidfVectorizer` to convert the text messages from the training set into numerical features and saves the fitted vectorizer.
-   **Models:** The script trains and evaluates the following four models:
    -   Logistic Regression
    -   Multinomial Naive Bayes
    -   Support Vector Machine (SVM)
    -   Random Forest
-   **Output (Serialization):**
    -   Each trained model is saved as a separate `.pkl` file in the `models/` directory (e.g., `logistic_regression_model.pkl`).
    -   The fitted `TfidfVectorizer` is saved as `tfidf_vectorizer.pkl`.
    -   The feature vocabulary is saved to `vocab.json`.

---

### 2. Streamlit Application (`streamlit_app.py`)

The application is designed with a multi-page structure to provide a clear and organized user experience.

-   **Page 1: Live Prediction**
    -   **UI:** A dropdown to select a model and a text area for user input.
    -   **Data Flow:**
        1.  User selects a model and enters text.
        2.  The loaded `TfidfVectorizer` transforms the input text.
        3.  The selected model predicts the class (`spam`/`ham`) and the probability.
        4.  The result and probability are displayed to the user.

-   **Page 2: Batch Prediction**
    -   **UI:** A dropdown to select a model and a file uploader for CSV files.
    -   **Data Flow:**
        1.  User uploads a CSV containing a `message` column.
        2.  The `TfidfVectorizer` transforms all messages in the file.
        3.  The selected model predicts the class for each message.
        4.  The results are displayed in a table and made available for download.

-   **Page 3: Model Comparison**
    -   **UI:** A data table and a bar chart.
    -   **Data Flow:**
        1.  The page loads all trained models from the `models/` directory.
        2.  It runs predictions for each model on the same held-out test set.
        3.  It calculates key metrics (Accuracy, Precision, Recall, F1-Score, Prediction Time) for each model.
        4.  The metrics are displayed in a summary table and visualized in a bar chart for easy comparison.

-   **Page 4: Exploratory Data Analysis (EDA)**
    -   **UI:** A series of charts and word clouds.
    -   **Data Flow:**
        1.  The page loads the `sms_spam_no_header.csv` dataset.
        2.  It generates and displays:
            -   A pie chart of the spam vs. ham distribution.
            -   A histogram of message length distribution, separated by class.
            -   Word clouds for the most frequent terms in both spam and ham messages.

## Risks / Trade-offs
- **Increased Complexity:** Managing multiple models and a multi-page app adds complexity. This is mitigated by decoupling the training script from the Streamlit app.
- **Performance:** Word cloud generation and model comparison on the fly can be resource-intensive. This is mitigated by using Streamlit's caching mechanisms (`@st.cache_data`, `@st.cache_resource`) to avoid redundant computations.

## Open Questions
- Further optimization of hyperparameters for each model could be explored.
- The user feedback mechanism is currently simulated; future work could involve storing this feedback to curate a new evaluation dataset.