## **Project One: TripAdvisor Recommendation Challenge - Beating BM25**
_**Authors:** Alberto MARTINELLI, Alessia SARRITZU_

The goal of the project is to develop a recommendation system that relies solely on user reviews to suggest similar places based on given queries. The system will propose the most relevant location based on the text of the reviews.

1. **Data pre-processing**:
    - Utilise only the reviews where ratings are composed strictly with this aspects on reviews with ratings for the following aspects: **service**, **cleanliness**, **overall**, **value**, **location**, **sleep quality**, and **rooms**.
    - Concatenate reviews by `offering_id` to compute average ratings for evaluation.

2. **BM25 Implementation**:
    - Implement a BM25 baseline using the **Rank-BM25** library.
    - Measure the performance of BM25 through Mean Square Error (MSE) between the ratings of the query place and the recommended place.

3. **Enhanced Unsupervised Model**:
    - Create a new unsupervised model to outperform BM25, potentially integrating it with other methods, while ensuring the model does not directly utilize ratings in its learning process.
    - Measure performance of the **Enhanced Model** through Mean Square Error (MSE) between the ratings of the query place and the recommended place, with the aim of achieving a lower MSE than the BM25 baseline.

4. **To Deliver**:
    - A **report** detailing the techniques and approaches used, along with results and analysis.
    - A shared Colab notebook that contains the implementation of the project.
