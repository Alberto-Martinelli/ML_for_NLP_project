 query_place[['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']]
    top_match, mse = apply_bm25(df, query_place['text'], query_ratings)
    print("Mean sq