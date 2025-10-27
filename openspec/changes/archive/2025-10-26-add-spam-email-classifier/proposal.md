## Why
To significantly enhance the "Spam Email Classifier" project by evolving it from a single-model predictor into a comprehensive analysis and comparison platform. The goals are to:
1.  Integrate and evaluate multiple machine learning algorithms to identify the most effective models.
2.  Greatly improve user interaction by providing a dynamic, real-time prediction interface.
3.  Introduce a dedicated section for exploratory data analysis (EDA) to derive deeper insights from the dataset.

## What Changes
- **Multi-Model Training:** The core training script will be updated to train, evaluate, and save four different models: Logistic Regression, Naive Bayes, SVM, and Random Forest.
- **New Streamlit Pages:** The Streamlit application will be completely redesigned with a multi-page structure:
    - **Live Prediction:** An interactive page for real-time classification using a selected model.
    - **Batch Prediction:** A page for uploading a CSV file for bulk classification.
    - **Model Comparison:** A dashboard to visually and numerically compare the performance of all trained models.
    - **Exploratory Data Analysis (EDA):** A new section featuring word clouds and message length distributions.
- **Enhanced Interactivity:** The prediction interface will now show classification probability and include simulated user feedback buttons.

## Impact
- **Affected specs:** The spam classifier spec will be updated to reflect the multi-model architecture and new interactive features.
- **Affected code:**
    - `5114050013_hw3.py`: Will be refactored to handle the training of multiple models.
    - `streamlit_app.py`: Will be completely rewritten to support the new multi-page, interactive dashboard.
    - **New Files:** Model files for each algorithm will be stored in a new `models/` directory.