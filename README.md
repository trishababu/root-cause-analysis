# root-cause-analysis
in this code, i have given the root cause analysis which is based on the common themes



GMM Ticket Clustering with Text Summarization
This repository contains code that uses Natural Language Processing (NLP) techniques, Gaussian Mixture Models (GMM), and clustering to analyze and group similar ticket/issue summaries. The goal is to help identify recurring issues and themes in ticket data, enabling efficient ticket management.

Features
Text Preprocessing: Tokenization, lowercasing, lemmatization, and removal of stopwords and special characters to clean the input data.
TF-IDF Vectorization: Converts text data into numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF), which reflects the importance of words in the dataset.
Clustering with Gaussian Mixture Models (GMM): Clusters the ticket summaries based on their semantic similarity, grouping them into predefined clusters.
PCA Visualization: Provides a 2D scatter plot visualization of the clustered data using Principal Component Analysis (PCA).
Cluster Summarization: Generates a summary of the top keywords for each cluster, giving insights into the themes of the grouped tickets.
Cluster Prediction: Predicts the cluster for a new ticket/issue based on its summary.
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/repo-name.git
Navigate to the project directory:
bash
Copy code
cd repo-name
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Requirements
Python 3.x
pandas
numpy
nltk
scikit-learn
matplotlib
You can install the necessary Python packages using the command:

bash
Copy code
pip install pandas numpy nltk scikit-learn matplotlib
Usage
Preprocess and Cluster Tickets:
The script preprocesses the ticket summaries by removing unwanted characters and lemmatizing the text. Then, it uses the TF-IDF vectorizer to convert the text into numerical data, which is used by the GMM to cluster the tickets.

Generate Clustering Visualization:
After clustering, the tickets are visualized using PCA to reduce the feature space to two dimensions. This allows the visualization of clusters on a 2D scatter plot.

Summarize Clusters:
The script provides summaries for each cluster, showing the top keywords that describe the cluster’s theme.

Predict Cluster for New Tickets:
The script allows for predicting the cluster of a new ticket based on its summary, enabling classification of future tickets.

Example
To run the code, execute the following:

python
Copy code
# Running the clustering and visualization process
python yourscript.py
Example of predicting a new ticket's cluster:

python
Copy code
# Example: Predicting cluster for a new ticket
new_ticket_summary = "site notification"
predicted_cluster = predict_cluster(new_ticket_summary)
print(f'The new ticket belongs to cluster: {predicted_cluster}')
Input Data Format
The input data file (dataset.csv) should have a single column Issue_id_Summary, which contains the issue ID followed by the ticket summary, separated by a tab character (\t).

Example:

vbnet
Copy code
Issue_id_Summary
12345\tSystem error occurred in application
23456\tUser unable to login to the portal
Output
Visualization: The script will generate and save a PNG file (gmm_clustering_visualization.png) that shows the clusters in 2D space.
Clustered Data: The clustered tickets are saved to a CSV file (grouped_tickets_gmm.csv), which includes the assigned cluster for each ticket.
Cluster Summaries: Top keywords for each cluster are printed to the console.
Contributing
Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.




there is the exact code:

import pandas as pd  # Import pandas for data manipulation
import re  # Import regex for text preprocessing
import nltk  # Import NLTK for natural language processing
from nltk.corpus import stopwords  # Import stopwords to filter common words
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer for word lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF Vectorizer for text feature extraction
from sklearn.mixture import GaussianMixture  # Import GMM for clustering
from sklearn.decomposition import PCA  # Import PCA for data visualization
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import numpy as np  # Import numpy for array manipulation

# Download necessary NLTK resources
nltk.download('stopwords')  # Download the stopwords for filtering
nltk.download('wordnet')  # Download the WordNet data for lemmatization

# Preprocessing function to clean and prepare text data
def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove digits from the text
    text = re.sub(r'\d+', '', text)
    # Replace non-word characters (like punctuation) with space
    text = re.sub(r'\W+', ' ', text)
    # Split text into tokens (words)
    tokens = text.split()
    # Initialize the WordNetLemmatizer for word lemmatization
    lemmatizer = WordNetLemmatizer()
    # Lemmatize tokens and remove stopwords (common words like 'the', 'is')
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Load dataset containing issue summaries
data = pd.read_csv('dataset.csv', encoding='Windows-1252', error_bad_lines=False)

# Rename the single column as 'Issue_id_Summary'
data.columns = ['Issue_id_Summary']

# Split the 'Issue_id_Summary' column into 'Issue id' and 'Summary' based on the tab ('\t') separator
data[['Issue id', 'Summary']] = data['Issue_id_Summary'].str.split('\t', expand=True)

# Apply the preprocessing function to the 'Summary' column and create a new column 'clean_summary'
data['clean_summary'] = data['Summary'].apply(preprocess)

# TF-IDF Vectorization
# Initialize the TF-IDF vectorizer, limiting the vocabulary to the top 1000 most important words
vectorizer = TfidfVectorizer(max_features=1000)
# Apply the vectorizer to the 'clean_summary' column, converting the text data to numerical data (TF-IDF matrix)
X = vectorizer.fit_transform(data['clean_summary']).toarray()

# Gaussian Mixture Models (GMM) for clustering
# Define the number of clusters to identify using GMM (21 clusters in this case)
n_components = 21  # Adjust the number of components as needed
# Initialize the GMM model with the number of components and a random seed for reproducibility
gmm = GaussianMixture(n_components=n_components, random_state=42)
# Fit the GMM model on the TF-IDF data and assign each issue to a cluster
data['cluster'] = gmm.fit_predict(X)

# Visualization using PCA (Principal Component Analysis)
# Initialize PCA to reduce the TF-IDF data to 2 dimensions for visualization
pca = PCA(n_components=2)
# Fit and transform the TF-IDF data into 2-dimensional PCA components
X_pca = pca.fit_transform(X)
# Set the size of the plot
plt.figure(figsize=(10, 6)) 
# Create a scatter plot of the 2D PCA components, coloring each point by its assigned cluster
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster'], cmap='viridis', s=50)
# Set the title and labels for the plot
plt.title('GMM Clustering Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
# Add a color bar to indicate which color corresponds to which cluster
plt.colorbar(label='Cluster')

# Save the visualization to a PNG file instead of displaying it interactively
plt.savefig('gmm_clustering_visualization.png')

# Save the clustered data into a CSV file for future use
data.to_csv('grouped_tickets_gmm.csv', index=False)

# Function to predict the cluster for a new ticket summary
def predict_cluster(new_summary):
    # Preprocess the new summary using the same preprocessing function
    processed_summary = preprocess(new_summary)
    # Convert the preprocessed summary into a TF-IDF vector
    summary_vector = vectorizer.transform([processed_summary]).toarray()
    # Predict which cluster the summary belongs to using the trained GMM model
    cluster = gmm.predict(summary_vector)
    # Return the predicted cluster index
    return cluster[0]

# Function to generate a summary for each cluster
def summarize_clusters(X, data, vectorizer, n_top_words=10):
    # Get the feature names (words) used in the TF-IDF matrix
    feature_names = vectorizer.get_feature_names_out()
    # Initialize a list to store the cluster summaries
    summaries = []
    # Loop through each cluster to generate a summary
    for cluster_num in range(n_components):
        # Get the indices of the data points that belong to the current cluster
        cluster_indices = data[data['cluster'] == cluster_num].index
        # Check if the cluster is empty (i.e., no data points assigned to it)
        if len(cluster_indices) == 0:
            # If empty, append a message indicating no data points
            summaries.append(f'Cluster {cluster_num}: No data points')
            continue
        
        # Compute the average TF-IDF scores for the words in this cluster
        cluster_tfidf = X[cluster_indices].mean(axis=0)
        # Find the indices of the top words with the highest average TF-IDF scores in the cluster
        top_terms_indices = np.argsort(cluster_tfidf)[-n_top_words:]
        # Get the actual words corresponding to these indices
        top_terms = [feature_names[i] for i in top_terms_indices]
        # Create a summary for the cluster using the top terms (combining the two highest and two lowest scoring words)
        summary = ', '.join(top_terms[-2:]) + ', ' + ', '.join(top_terms[:2])
        # Append the summary to the list
        summaries.append(f'Cluster {cluster_num}: ' + summary)
    return summaries

# Generate and print summaries for each cluster
cluster_summaries = summarize_clusters(X, data, vectorizer)
for summary in cluster_summaries:
    print(summary)

# Example of predicting a cluster for a new ticket
new_ticket_summary = "site notification"  # Example input for a new ticket
# Predict the cluster for the new ticket summary
predicted_cluster = predict_cluster(new_ticket_summary)
# Print the predicted cluster index for the new ticket
print(f'The new ticket belongs to cluster: {predicted_cluster}')

