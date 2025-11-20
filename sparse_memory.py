import mlx.core as mx
import numpy as np
import time

class SparseMemory:
    """
    Implements Option C: Sparse Memory with AMX Instructions.
    
    Leverages the fact that high-dimensional memory representations are naturally sparse.
    Instead of storing dense vectors (128 floats), we store only the top-k activations.
    
    This mimics biological sparsity (neurons fire sparsely) and allows for massive capacity scaling.
    """
    
    def __init__(self, capacity=1_000_000, dim=128, sparsity=0.95):
        self.capacity = capacity
        self.dim = dim
        self.sparsity = sparsity
        self.active_elements = int(dim * (1 - sparsity))
        
        print("🧠 Initializing Sparse Memory System")
        print(f"   Capacity: {capacity:,} memories")
        print(f"   Dimension: {dim}")
        print(f"   Sparsity: {sparsity*100:.1f}% ({self.active_elements} active neurons)")
        
        # Sparse storage: Indices and Values
        # We store flattened arrays for efficiency
        self.indices = np.zeros((capacity, self.active_elements), dtype=np.int16)
        self.values = np.zeros((capacity, self.active_elements), dtype=np.float16)
        self.count = 0
        
    def add(self, vector: mx.array):
        """Add a dense vector to sparse memory."""
        if self.count >= self.capacity:
            # Simple ring buffer for now
            idx = self.count % self.capacity
        else:
            idx = self.count
            
        # Convert to numpy for sparse encoding (simulating custom kernel)
        v = np.array(vector)
        
        # Find top-k elements (Sparsification)
        # In a real implementation, this would be an AMX-optimized kernel
        top_k_indices = np.argpartition(np.abs(v), -self.active_elements)[-self.active_elements:]
        top_k_values = v[top_k_indices]
        
        self.indices[idx] = top_k_indices
        self.values[idx] = top_k_values.astype(np.float16)
        self.count += 1
        
    def retrieve(self, query: mx.array, k=5):
        """
        Retrieve nearest neighbors using sparse dot product.
        
        This is where the speedup happens: we only multiply non-zero elements.
        """
        # In a real implementation, this would be a custom Metal kernel
        # Here we simulate the sparse operation logic
        
        q = np.array(query)
        
        # Sparse Dot Product: sum(q[idx] * val) for each memory
        # We can vectorize this in numpy for the simulation
        
        # Gather query values at the stored indices
        # shape: (count, active_elements)
        q_values = q[self.indices[:self.count]]
        
        # Dot product = sum(q_val * stored_val)
        scores = np.sum(q_values * self.values[:self.count], axis=1)
        
        # Find top-k matches
        best_indices = np.argpartition(scores, -k)[-k:]
        
        results = []
        for idx in best_indices:
            # Reconstruct dense vector (for return)
            dense = np.zeros(self.dim)
            dense[self.indices[idx]] = self.values[idx]
            results.append((dense, scores[idx]))
            
        return results

def benchmark_sparse_vs_dense():
    """Compare memory usage and theoretical capacity."""
    print("\n" + "="*70)
    print("📉 SPARSE MEMORY BENCHMARK")
    print("="*70)
    
    dim = 1024  # High dimensional vector
    count = 100_000
    sparsity = 0.95
    
    # Dense Memory Cost
    dense_bytes = count * dim * 4  # FP32
    dense_mb = dense_bytes / (1024**2)
    
    # Sparse Memory Cost
    # Store: index (int16) + value (float16) per active element
    active = int(dim * (1 - sparsity))
    sparse_bytes = count * active * (2 + 2)
    sparse_mb = sparse_bytes / (1024**2)
    
    print(f"Scenario: {count:,} memories, {dim} dimensions")
    print(f"Dense Storage (FP32):   {dense_mb:.2f} MB")
    print(f"Sparse Storage (95%):   {sparse_mb:.2f} MB")
    print(f"Compression Ratio:      {dense_mb/sparse_mb:.1f}x")
    
    # Capacity Projection
    memory_budget_gb = 32
    dense_capacity = (memory_budget_gb * 1024**3) / (dim * 4)
    sparse_capacity = (memory_budget_gb * 1024**3) / (active * 4)
    
    print("\n🚀 CAPACITY PROJECTION (32GB Budget):")
    print(f"   Dense Capacity:  {int(dense_capacity):,} memories")
    print(f"   Sparse Capacity: {int(sparse_capacity):,} memories")
    print(f"   Gain:            +{int(sparse_capacity - dense_capacity):,} memories")

    # Retrieval Simulation
    print("\n⚡ RETRIEVAL SIMULATION")
    mem = SparseMemory(capacity=10000, dim=128, sparsity=0.90)
    
    # Fill memory
    print("   Populating memory...")
    for _ in range(1000):
        mem.add(mx.random.normal((128,)))
        
    # Query
    print("   Querying...")
    start = time.time()
    q = mx.random.normal((128,))
    results = mem.retrieve(q)
    elapsed = time.time() - start
    
    print(f"   Retrieval time (1k items): {elapsed*1000:.3f}ms")
    print(f"   Top match score: {results[0][1]:.3f}")

if __name__ == "__main__":
    benchmark_sparse_vs_dense()
