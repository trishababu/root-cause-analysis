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
