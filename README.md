## **Project One: TripAdvisor Recommendation Challenge - Beating BM25**
_**Authors:** Alberto MARTINELLI, Alessia SARRITZU_

### **Introduction**
The goal of this project is to develop an unsupervised recommendation system that uses user reviews to suggest similar locations, outperforming the BM25 baseline. The system is evaluated using **Mean Squared Error (MSE)** between query and recommended location ratings, focusing exclusively on review text.

### **Development Phases**
1. **Data and Libraries import**
   - Import the dataset.
   - Import the necessary libraries.
   - Initialize necessary variables and utilities.

2. **Data Preparation:**
   - Filter reviews to include only those with ratings strictly covering seven aspects: **service**, **cleanliness**, **overall**, **value**, **location**, **sleep quality**, and **rooms**.
   - Concatenate reviews by `offering_id` and compute average ratings for each aspect to represent each location.
   - Take a random sample of 100 queries from the dataset for consistent evaluation of model performance.

3. **Data Pre-Processing:**
   - Apply text preprocessing to standardize review text:
     - Tokenization: Split text into words.
     - Stop word removal: Exclude common, irrelevant words.
     - Lemmatization: Reduce words to their base forms.

4. **BM25 Implementation:**
   - Use the **Rank-BM25** library to recommend locations based on textual similarity.
   - Evaluate performance by calculating MSE between query and recommended location ratings.

5. **Enhanced Model Implementation:**
   - Create a more advanced unsupervised model to outperform BM25.
   - Use **TF-IDF vectorization** and **cosine similarity** to capture semantic relationships between reviews.

6. **Evaluation and Comparison:**
   - Compare MSE results to determine the improved performance of the enhanced model.
