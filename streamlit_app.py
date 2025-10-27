import streamlit as st
import os

st.title("🔍 Environment Debugger")

st.header("Working Directory")
try:
    cwd = os.getcwd()
    st.write(f"**Current Working Directory:** `{cwd}`")
except Exception as e:
    st.error(f"Could not get current working directory: {e}")

st.header("Contents of Current Directory")
try:
    dir_contents = os.listdir('.')
    st.write(dir_contents)
except Exception as e:
    st.error(f"Could not list contents of current directory: {e}")

st.header("Recursive File Listing")
try:
    all_files = []
    # Limit the walk to avoid excessive output
    for root, dirs, files in os.walk(cwd, topdown=True):
        # Exclude virtual environments and other noise
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.venv']]
        for name in files:
            all_files.append(os.path.join(root, name))
        for name in dirs:
            all_files.append(os.path.join(root, name) + '/')
    
    st.write("Found files and directories (filtered for relevance):")
    # To avoid showing too much, let's filter for what we care about
    relevant_files = [f.replace(cwd, '.', 1) for f in all_files if 'models' in f or '.pkl' in f or '.csv' in f or '.json' in f]
    st.code('\n'.join(sorted(relevant_files)))

except Exception as e:
    st.error(f"Could not perform recursive file listing: {e}")


st.header("Specific Path Checks")
paths_to_check = [
    "models",
    "tfidf_vectorizer.pkl",
    "sms_spam_no_header.csv",
    "models/logistic_regression_model.pkl"
]

for path in paths_to_check:
    exists = os.path.exists(path)
    st.write(f"Does `{path}` exist? **{exists}**")
