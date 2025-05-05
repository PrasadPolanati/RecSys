# 🎬 Two-Tower Deep Neural Network on MovieLens Dataset

This repository implements a **Two-Tower Deep Neural Network (DNN)** architecture for movie recommendation using the [MovieLens dataset](https://grouplens.org/datasets/movielens/). The two-tower model separately encodes users and movies into embeddings and learns their interactions to predict user preferences efficiently. This approach is commonly used in large-scale retrieval systems such as YouTube and Amazon.

## 📌 Features

- Two-tower architecture for learning user and item embeddings.
- Efficient retrieval using dot product similarity.
- Evaluation using recall@k metrics.
- Built with TensorFlow/Keras.
- Supports training, validation, and inference.



The Two-Tower Deep Neural Network (DNN) architecture is designed for scalable recommendation tasks, especially in scenarios where millions of users and items interact. It's particularly effective for retrieval-based recommendation systems, where the goal is to find the most relevant items from a large corpus for a given user.

🔧 Architecture Details
The model consists of two separate "towers":

🧍 User Tower
Inputs: User-specific features such as:

user_id

Demographic information (e.g., gender, age, occupation) (optional)

Historical interactions (optional enhancement)

Embedding Layer: Transforms categorical inputs into dense vectors.

Dense Layers: A stack of fully connected layers to learn user representation.

Output: A fixed-size embedding vector representing the user.

🎥 Item (Movie) Tower
Inputs: Movie-specific features such as:

movie_id

genres

movie_title (optional: can be encoded using word embeddings or BERT)

Embedding Layer: Converts categorical and text features into dense embeddings.

Dense Layers: Learns a meaningful representation of each movie.

Output: A fixed-size embedding vector representing the movie.

🔗 Interaction Layer
During training, the model computes the dot product (or cosine similarity) between user and item embeddings to measure relevance.

This similarity is passed through a loss function, typically sampled softmax or contrastive loss, to distinguish positive interactions from negative samples.

🧮 Training Objective
The model is trained using a pairwise ranking loss, where:

Positive pairs are known user-item interactions (e.g., movies a user has rated or watched).

Negative pairs are randomly sampled non-interactions.

The goal is to maximize the similarity for positive pairs while minimizing it for negative pairs.

📦 Why Two Towers?
Scalability: Each tower can be precomputed independently and stored, allowing efficient ANN (Approximate Nearest Neighbor) search during inference.

Flexibility: Easy to add features to only one side (e.g., enrich movie tower with NLP or vision-based embeddings).

Deployment Ready: You can use tools like FAISS, ScaNN, or Annoy to scale retrieval across millions of candidates.

