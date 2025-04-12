from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def vectorise_text(corpus, max_features=1000, stop_words='english'):
    """Convert a list of documents into TF-IDF features."""
    vectoriser = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X = vectoriser.fit_transform(corpus)
    return X, vectoriser

def run_kmeans(X, num_clusters=5, random_state=42):
    """Cluster the data using KMeans."""
    model = KMeans(n_clusters=num_clusters, random_state=random_state)
    model.fit(X)
    return model

def reduce_dimensions(X, n_components=2):
    """Reduce high-dimensional TF-IDF vectors to 2D for visualisation."""
    reducer = PCA(n_components=n_components)
    X_reduced = reducer.fit_transform(X.toarray())
    return X_reduced, reducer

def summarise_clusters(dataframe, labels_col='cluster', text_col='title'):
    """Print the number of items per cluster and sample titles."""
    summary = dataframe.groupby(labels_col)[text_col].apply(lambda x: x.sample(min(len(x), 5))).reset_index()
    return summary
