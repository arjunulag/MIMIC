"""
GloVe Word Vector Loader and Similarity Finder

This script loads GloVe word vectors and provides utilities to find
similar words at specified distances from a target word.

Download GloVe vectors from: https://nlp.stanford.edu/projects/glove/
Recommended: glove.6B.zip (822MB) - contains 50d, 100d, 200d, 300d vectors
"""

import numpy as np
from pathlib import Path
import pickle


class GloVeLoader:
    def __init__(self, glove_path: str = None):
        """
        Initialize the GloVe loader.
        
        Args:
            glove_path: Path to GloVe text file (e.g., 'glove.6B.100d.txt')
        """
        self.word_to_vec = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vectors = None
        self.dimension = None
        
        if glove_path:
            self.load_glove(glove_path)
    
    def load_glove(self, glove_path: str):
        """
        Load GloVe vectors from a text file.
        
        Args:
            glove_path: Path to the GloVe .txt file
        """
        print(f"Loading GloVe vectors from {glove_path}...")
        
        words = []
        vectors = []
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                if self.dimension is None:
                    self.dimension = len(vector)
                
                words.append(word)
                vectors.append(vector)
                self.word_to_vec[word] = vector
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                
                if (idx + 1) % 100000 == 0:
                    print(f"  Loaded {idx + 1} words...")
        
        self.vectors = np.array(vectors, dtype=np.float32)
        # Normalize vectors for cosine similarity
        self.normalized_vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        
        print(f"Loaded {len(words)} words with {self.dimension}-dimensional vectors.")
    
    def save_cache(self, cache_path: str):
        """Save loaded vectors to a pickle cache for faster loading."""
        data = {
            'word_to_vec': self.word_to_vec,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vectors': self.vectors,
            'normalized_vectors': self.normalized_vectors,
            'dimension': self.dimension
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved cache to {cache_path}")
    
    def load_cache(self, cache_path: str):
        """Load vectors from a pickle cache."""
        print(f"Loading from cache: {cache_path}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        self.word_to_vec = data['word_to_vec']
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']
        self.vectors = data['vectors']
        self.normalized_vectors = data['normalized_vectors']
        self.dimension = data['dimension']
        print(f"Loaded {len(self.word_to_vec)} words from cache.")
    
    def get_vector(self, word: str) -> np.ndarray:
        """Get the vector for a word."""
        word = word.lower()
        if word not in self.word_to_vec:
            raise KeyError(f"Word '{word}' not found in vocabulary")
        return self.word_to_vec[word]
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_similar_words(self, word: str, n: int = 10, 
                          min_similarity: float = None,
                          max_similarity: float = None) -> list:
        """
        Find the n most similar words to the given word.
        
        Args:
            word: Target word
            n: Number of similar words to return
            min_similarity: Minimum cosine similarity threshold (optional)
            max_similarity: Maximum cosine similarity threshold (optional)
            
        Returns:
            List of tuples (word, similarity_score)
        """
        word = word.lower()
        if word not in self.word_to_vec:
            raise KeyError(f"Word '{word}' not found in vocabulary")
        
        word_vec = self.word_to_vec[word]
        word_vec_norm = word_vec / np.linalg.norm(word_vec)
        
        # Calculate all similarities at once
        similarities = np.dot(self.normalized_vectors, word_vec_norm)
        
        # Apply thresholds if specified
        mask = np.ones(len(similarities), dtype=bool)
        if min_similarity is not None:
            mask &= (similarities >= min_similarity)
        if max_similarity is not None:
            mask &= (similarities <= max_similarity)
        
        # Get valid indices
        valid_indices = np.where(mask)[0]
        valid_similarities = similarities[valid_indices]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(valid_similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            actual_idx = valid_indices[idx]
            found_word = self.idx_to_word[actual_idx]
            if found_word != word:  # Exclude the word itself
                results.append((found_word, valid_similarities[idx]))
                if len(results) >= n:
                    break
        
        return results
    
    def find_words_at_distance(self, word: str, n: int = 10,
                               target_similarity: float = 0.5,
                               tolerance: float = 0.1) -> list:
        """
        Find words at approximately a certain similarity distance from the target.
        
        Args:
            word: Target word
            n: Number of words to return
            target_similarity: Target cosine similarity (0-1 range)
            tolerance: How much deviation from target_similarity is allowed
            
        Returns:
            List of tuples (word, similarity_score)
        """
        min_sim = target_similarity - tolerance
        max_sim = target_similarity + tolerance
        
        return self.find_similar_words(
            word, 
            n=n,
            min_similarity=min_sim,
            max_similarity=max_sim
        )
    
    def word_in_vocab(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        return word.lower() in self.word_to_vec
    
    def get_vocabulary(self) -> list:
        """Get list of all words in vocabulary."""
        return list(self.word_to_vec.keys())


def demo():
    """
    Demonstration of the GloVe loader functionality.
    """
    import os
    
    # Look for GloVe file
    glove_files = [
        'dolma_300_2024_1.2M.100_combined.txt',
        'glove.6B.100d.txt',
        'glove.6B.50d.txt', 
        'glove.6B.200d.txt',
        'glove.6B.300d.txt',
    ]
    
    glove_path = None
    cache_path = 'glove_cache.pkl'
    
    loader = GloVeLoader()
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        loader.load_cache(cache_path)
    else:
        # Find available GloVe file
        for gf in glove_files:
            if os.path.exists(gf):
                glove_path = gf
                break
        
        if glove_path is None:
            print("=" * 60)
            print("GloVe vectors not found!")
            print("Please download from: https://nlp.stanford.edu/projects/glove/")
            print("Recommended: glove.6B.zip")
            print("Extract and place the .txt file in this directory.")
            print("=" * 60)
            return None
        
        loader.load_glove(glove_path)
        loader.save_cache(cache_path)
    
    # Demo: Find similar words
    test_word = "king"
    print(f"\n{'=' * 60}")
    print(f"Finding 10 most similar words to '{test_word}':")
    print("=" * 60)
    
    similar = loader.find_similar_words(test_word, n=10)
    for word, score in similar:
        print(f"  {word:20s} (similarity: {score:.4f})")
    
    # Demo: Find words at a certain distance
    print(f"\n{'=' * 60}")
    print(f"Finding words at ~0.5 similarity distance from '{test_word}':")
    print("=" * 60)
    
    distant = loader.find_words_at_distance(test_word, n=10, 
                                            target_similarity=0.5, 
                                            tolerance=0.05)
    for word, score in distant:
        print(f"  {word:20s} (similarity: {score:.4f})")
    
    return loader


if __name__ == "__main__":
    loader = demo()
