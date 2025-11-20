"""
Biological NLP Module
---------------------
Implements language processing inspired by biological neural networks,
avoiding heavy Transformers/LLMs in favor of Sparse Distributed Representations (SDRs)
and Associative Memory.

Key Concepts:
1. SDR (Sparse Distributed Representation):
   - Words are mapped to large, sparse binary vectors (e.g., 2048 dimensions).
   - Semantic meaning is encoded in the *overlap* of active bits.
   - Mimics the "Grandmother Cell" or sparse coding in the neocortex.

2. Associative Memory:
   - Responses are not "generated" token-by-token.
   - They are "retrieved" whole from a memory bank based on semantic similarity.
   - This ensures grammatical correctness without a language model.
"""

import mlx.core as mx
from typing import Tuple

class SDREncoder:
    """
    Encodes text into Sparse Distributed Representations (SDRs).
    Uses a semantic hashing approach where similar words activate overlapping bits.
    """
    def __init__(self, vocab_size: int = 2048, sparsity: float = 0.05):
        self.vocab_size = vocab_size
        self.sparsity = sparsity
        self.num_active = int(vocab_size * sparsity)
        
        # A fixed random projection matrix to convert characters/subwords to SDR space
        # This acts like a fixed "embedding layer" that doesn't need training
        # but preserves some structural similarity.
        self.projection = mx.random.normal((256, vocab_size))

    def encode(self, text: str) -> mx.array:
        """
        Convert text to a sparse binary vector.
        """
        # 1. Simple character-level encoding (summed)
        # In a real system, we'd use n-grams or BPE, but this is raw Python/MLX.
        char_vecs = []
        for char in text.lower():
            # Create a one-hot-like vector for the character (hashed)
            seed = ord(char)
            key = mx.random.key(seed)
            v = mx.random.normal((self.vocab_size,), key=key)
            char_vecs.append(v)
            
        if not char_vecs:
            return mx.zeros((self.vocab_size,))
            
        # Sum all character vectors to get a "bag of characters" representation
        # This is a primitive form of semantic encoding
        dense_vec = mx.sum(mx.stack(char_vecs), axis=0)
        
        # 2. Apply Sparsity (k-Winner-Take-All)
        # Only the top k neurons fire
        # This creates the SDR
        top_k_indices = mx.argpartition(dense_vec, -self.num_active)[-self.num_active:]
        
        sdr = mx.zeros((self.vocab_size,))
        sdr[top_k_indices] = 1.0
        
        return sdr

class AssociativeMemory:
    """
    A content-addressable memory bank for conversation.
    Stores (ConceptVector -> ResponseText) pairs.
    """
    def __init__(self, input_dim: int = 128):
        self.input_dim = input_dim
        self.keys = []   # The "Thought Vectors" (Concept)
        self.values = [] # The "Response Texts" (Language)
        
    def add(self, concept_vector: mx.array, response: str):
        """Store a response linked to a specific concept/thought."""
        # Ensure vector is normalized for cosine similarity
        norm = mx.linalg.norm(concept_vector)
        if norm > 0:
            concept_vector = concept_vector / norm
            
        self.keys.append(concept_vector)
        self.values.append(response)
        
    def retrieve(self, query_vector: mx.array, temperature: float = 0.1) -> Tuple[str, float]:
        """
        Find the best matching response for the current thought.
        """
        if not self.keys:
            return "...", 0.0
            
        # Stack keys into a matrix
        key_matrix = mx.stack(self.keys) # [N, dim]
        
        # Normalize query
        query_norm = mx.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
            
        # Cosine similarity: query @ keys.T
        scores = query_vector @ key_matrix.T # [N]
        
        # Softmax with temperature to pick a response
        probs = mx.softmax(scores / temperature)
        
        # Greedy selection (argmax) for now, or sample
        best_idx = mx.argmax(probs).item()
        confidence = scores[best_idx].item()
        
        return self.values[int(best_idx)], confidence

    def load_defaults(self):
        """Populate with some basic conversational pairs."""
        # We need a way to generate concept vectors for these keys.
        # For now, we'll assume the user adds them with the encoder.
