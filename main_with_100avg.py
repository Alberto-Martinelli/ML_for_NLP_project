import os
import pandas as pd
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import swifter
from rank_bm25 import BM25Okapi
from sklearn.metrics import mean_squared_error

# Initialization
aspects = ["service", "cleanliness", "overall", "value", "location", "sleep_quality", "rooms"]

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def dataframe_preparation(df):
    # --------------------------- Filter data --------------------------------------------
    required_aspects = {"service", "cleanliness", "overall", "value", "location", "sleep_quality", "rooms"}
    filtered_df = df[df['ratings'].apply(lambda x: set(eval(x).keys()) == required_aspects)]
    filtered_df = filtered_df.reset_index(drop=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]    2.1 Data filtered")

    # -------------------------- Take a sample for model testing --------------------------
    sample_df = filtered_df.sample(n=100, random_state=42)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]    2.2 Sample of 100 queries retrieved for model testing")

    # ------------------ Concatenate reviews for the same place ---------------------------
    filtered_df.loc[:, 'ratings'] = filtered_df['ratings'].apply(eval)
    expanded_ratings_df = pd.json_normalize(filtered_df['ratings']).join(filtered_df[['offering_id', 'title', 'text']])

    # Calculate the mean of each rating aspect and concatenate reviews
    final_df = expanded_ratings_df.groupby('offering_id').agg(
        service=('service', 'mean'),  
        cleanliness=('cleanliness', 'mean'),
        overall=('overall', 'mean'),
        value=('value', 'mean'),
        location=('location', 'mean'),
        sleep_quality=('sleep_quality', 'mean'),
        rooms=('rooms', 'mean'),
        text=('text', lambda x: ' '.join(x)), 
    ).reset_index()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]    2.3 Reviews concatenated")
    return sample_df, final_df

# Text pre-processing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Query details extraction 
def extract_query(query_row, aspects, final_df):
    query_id = query_row['offering_id']
    query_text = query_row['text']
    place_ratings = final_df[final_df['offering_id'] == query_id][aspects].iloc[0]
    return query_id, query_text, place_ratings

# BM25 implementation
def apply_bm25(query_id, query_text, place_ratings, final_df, aspects):
    # Exclude the query place from the documents to avoid recommending it
    documents_df = final_df[final_df['offering_id'] != query_id].reset_index(drop=True)

    # Tokenize each document for BM25
    documents_df['text'] = documents_df['text'].astype(str)
    documents = documents_df['text'].apply(lambda x: x.split())
    bm25 = BM25Okapi(documents)
    scores = bm25.get_scores(query_text.split())
    top_match_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0]
    top_match = documents_df.iloc[top_match_index]

    # Calculate the MSE between the ratings of the query place and the recommended place
    recommended_ratings = top_match[aspects]
    mse = mean_squared_error(place_ratings, recommended_ratings)
    return mse


if __name__ == "__main__":
    # -------------------------- Import data --------------------------------------------
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 1. Importing data")
    df = pd.read_csv('reviews.csv')
    
    # --------------------------- Prepare data -------------------------------------------
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 2. Preparing data")
    sample_df, final_df = dataframe_preparation(df)
    
    # -------------------------- Pre-process data ----------------------------------------
    preprocessed_file = "final_df_preprocessed.csv"
    if os.path.exists(preprocessed_file):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 3. Loading preprocessed final_df")
        final_df = pd.read_csv(preprocessed_file)
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 3. Preprocessing final_df")
        final_df['text'] = final_df['text'].swifter.apply(preprocess_text)
        final_df.to_csv(preprocessed_file, index=False)
  
    # ---------------- Calculate MSE for BM25 ------------------------------------------------
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 4. Processing data (BM25)")
    output_file = "results.csv"
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["row_id", "offering_id", "bm25_mse"]).to_csv(output_file, index=False)

    existing_results = pd.read_csv(output_file)
    processed_ids = set(existing_results["row_id"].values)

    for index, row in sample_df.iterrows():
        if index not in processed_ids:
            query_id, query_text, place_ratings = extract_query(row, aspects, final_df)
            mse = apply_bm25(query_id, query_text, place_ratings, final_df, aspects)
            pd.DataFrame([{"row_id": index, "offering_id": query_id, "bm25_mse": mse}]).to_csv(output_file, mode="a", header=False, index=False)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]    -> Row {index}: processed with MSE={mse}")

    # Calculate and print average MSE of the entire file
    all_results = pd.read_csv(output_file)
    if not all_results.empty:
        overall_average_mse = all_results["bm25_mse"].mean()
        res = overall_average_mse
    else:
        res = "No data available in the file"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]    4.1 Calculating average MSE: {res}")
