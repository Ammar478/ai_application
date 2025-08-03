from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

class SimpleRetriever:
    def __init__(self, documents):
        """
        documents: list of dicts with 'source' and 'content'
        """
        self.documents = documents
        self.contents = [d.get("content", "") for d in self.documents]
        self.sources = [d.get("source", "") for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.contents)

    def retrieve(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        similarities = linear_kernel(q_vec, self.doc_matrix).flatten()
        if not np.any(similarities > 0):
            return []
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] <= 0:
                continue
            results.append({
                "source": self.sources[idx],
                "content": self.contents[idx],
                "score": float(similarities[idx])
            })
        return results