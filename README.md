# Smart Phishing Website Detection System

A Streamlit prototype that trains and compares Decision Tree, Random Forest, Logistic Regression, and SVM models on an uploaded labelled CSV dataset.

## Run locally
1. Install Python 3.10+.
2. In a terminal, navigate to this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Deploy free with Streamlit Community Cloud
1. Upload these files into a GitHub repository.
2. Visit https://share.streamlit.io and sign in using GitHub.
3. Click **Create app**, choose the repository, branch, and `app.py`.
4. Deploy.

## Important limitations
- Upload a labelled CSV to train models. Results are computed from that dataset and are not pre-filled or fabricated.
- Choose the correct class/target column when training.
- Predictions require feature values matching the dataset's encoding.
- This prototype classifies supplied feature values; it does not scan live URLs or automatically extract features.
