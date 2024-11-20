import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics import mean_squared_error

def dataframe_preparation(df):
    # Filtering
    required_aspects = {"service", "cleanliness", "overall", "value", "location", "sleep_quality", "rooms"}
    filtered_df = df[df['ratings'].apply(lambda x: set(eval(x).keys()) == required_aspects)]

    # Convert ratings field to dictionary and expand it to columns
    filtered_df = filtered_df.copy()  # Ensure it's a separate DataFrame, not a view
    filtered_df['ratings'] = filtered_df['ratings'].apply(eval) #Modify filtered_df to convert ratings field from string to dictionary (json format)
    filtered_df = filtered_df.reset_index(drop=True) #Reset index of filtered_df to avoid misalignment issues
    ratings_expanded_df = pd.json_normalize(filtered_df['ratings']) #expand dictionary to columns
    selected_columns_df = filtered_df[['text', 'offering_id']] 
    combined_df = pd.concat([ratings_expanded_df, selected_columns_df], axis=1) # join with offering_id, title and text
    df = combined_df

    # Calculate the mean of each rating aspect and concatenate texts of reviews
    df = df.groupby('offering_id').agg(
        service=('service', 'mean'),  # Average rating for 'service' aspect (you can add others as needed)
        cleanliness=('cleanliness', 'mean'),
        overall=('overall', 'mean'),
        value=('value', 'mean'),
        location=('location', 'mean'),
        sleep_quality=('sleep_quality', 'mean'),
        rooms=('rooms', 'mean'),
        text=('text', lambda x: ' '.join(x)), # Concatenate all text entries
    ).reset_index()
    return df

def extract_query(df):
    # Select a random query place from the dataset
    query_place = df.sample(1).iloc[0]
    # And print its details
    query_id = query_place['offering_id']
    query_ratings = query_place[['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']]
    query_text = query_place['text']
    # print("Query Place Details:")
    # print(f"Offering ID: {query_id}")
    # print("Ratings:")
    # for aspect, rating in query_ratings.items():
    #     print(f"  {aspect}: {rating}")
    # print("\nReviews texts concatenated:")
    # print(query_text)

    # Exclude the query place from the documents to avoid recommending it
    documents_df = df[df['offering_id'] != query_id].reset_index(drop=True)
    return query_place, documents_df

def apply_bm25(documents_df, query_text, query_ratings):
    # Load and preprocess the reviews (assuming df contains concatenated reviews by 'offering_id')
    documents = documents_df['text'].apply(lambda x: x.split())  # Tokenize each document by splitting words
    bm25 = BM25Okapi(documents)

    # Define a query (a review or set of reviews from a specific place)
    tokenized_query = query_text.split()  # Tokenized query

    # Get BM25 scores for the query across all documents
    scores = bm25.get_scores(tokenized_query)

    # Step 5: Retrieve the top matching place based on BM25 score
    top_n = 1  # Number of top matches to retrieve
    top_match_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n][0] # Sorts the scores in descending order and retrieves the highest score
    top_match = documents_df.iloc[top_match_index]

    # Step 6: Calculate the MSE between the ratings of the query place and the recommended place
    recommended_ratings = top_match[['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']]
    mse = mean_squared_error(query_ratings, recommended_ratings)
    return top_match, mse


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 300)
    df = pd.read_csv('reviews.csv')
    df = dataframe_preparation(df)
    query_place, df = extract_query(df)
    query_ratings = query_place[['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']]
    top_match, mse = apply_bm25(df, query_place['text'], query_ratings)
    print("Mean square error: ", mse)





