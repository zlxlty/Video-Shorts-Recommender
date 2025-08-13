"""
FAISS-based candidate generation for video recommendations.
Implements multiple retrieval strategies including vector similarity search.
"""

import numpy as np
import pandas as pd
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import yaml
from sklearn.preprocessing import normalize
from tqdm import tqdm


class FAISSRetriever:
    """FAISS-based video retrieval for candidate generation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize FAISS retriever."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.features_dir = Path(self.config['paths']['data']['features'])
        self.n_candidates = self.config['candidate_generation']['n_candidates']
        
        # FAISS index parameters
        self.index_type = self.config['candidate_generation']['faiss'].get('index_type', 'Flat')
        self.nprobe = self.config['candidate_generation']['faiss'].get('nprobe', 10)
        
        # Strategy weights
        self.strategy_weights = self.config['candidate_generation']['strategy_weights']
        
        # Initialize indices
        self.video_index = None
        self.user_index = None
        self.video_metadata = None
        self.user_metadata = None
        
    def load_features(self):
        """Load pre-computed features."""
        print("Loading features...")
        
        # Load video features
        videos_features = pd.read_parquet(self.features_dir / "videos_features.parquet")
        self.video_metadata = videos_features[['video_id', 'category', 'duration', 
                                              'engagement_rate', 'view_count']].copy()
        
        # Extract video embeddings
        self.video_embeddings = np.stack(videos_features['video_embedding_processed'].values)
        self.video_embeddings = normalize(self.video_embeddings, axis=1)
        
        # Load user features
        users_features = pd.read_parquet(self.features_dir / "users_features.parquet")
        self.user_metadata = users_features[['user_id', 'interests', 'activity_level']].copy()
        
        # Extract user embeddings
        self.user_embeddings = np.stack(users_features['user_embedding'].values)
        self.user_embeddings = normalize(self.user_embeddings, axis=1)
        
        print(f"  Loaded {len(self.video_embeddings)} video embeddings")
        print(f"  Loaded {len(self.user_embeddings)} user embeddings")
        
    def build_video_index(self):
        """Build FAISS index for video embeddings."""
        print("Building FAISS video index...")
        
        d = self.video_embeddings.shape[1]  # Dimension
        
        if self.index_type == "Flat":
            # Exact search (brute force)
            self.video_index = faiss.IndexFlatL2(d)
        elif self.index_type.startswith("IVF"):
            # Inverted file index for faster search
            nlist = int(self.index_type.split(",")[0].replace("IVF", ""))
            quantizer = faiss.IndexFlatL2(d)
            self.video_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            
            # Train the index
            print(f"  Training IVF index with {nlist} clusters...")
            self.video_index.train(self.video_embeddings.astype('float32'))
            self.video_index.nprobe = self.nprobe
        else:
            # Default to flat index
            self.video_index = faiss.IndexFlatL2(d)
        
        # Add vectors to index
        self.video_index.add(self.video_embeddings.astype('float32'))
        
        print(f"  Index built with {self.video_index.ntotal} vectors")
        
    def build_user_index(self):
        """Build FAISS index for user embeddings (for similar user retrieval)."""
        print("Building FAISS user index...")
        
        d = self.user_embeddings.shape[1]
        self.user_index = faiss.IndexFlatL2(d)
        self.user_index.add(self.user_embeddings.astype('float32'))
        
        print(f"  User index built with {self.user_index.ntotal} vectors")
    
    def vector_search(self, query_embedding: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar videos using vector similarity."""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        query_embedding = normalize(query_embedding, axis=1)
        
        distances, indices = self.video_index.search(query_embedding, k)
        return indices[0], distances[0]
    
    def popularity_based_retrieval(self, k: int = 100) -> List[str]:
        """Get most popular videos based on views and engagement."""
        # Calculate popularity score
        self.video_metadata['popularity_score'] = (
            self.video_metadata['view_count'] * 0.7 + 
            self.video_metadata['engagement_rate'] * 1000 * 0.3
        )
        
        # Get top k popular videos
        top_videos = self.video_metadata.nlargest(k, 'popularity_score')
        return top_videos['video_id'].tolist()
    
    def collaborative_filtering(self, user_id: str, interaction_history: pd.DataFrame, 
                              k: int = 100) -> List[str]:
        """Get recommendations based on similar users' preferences."""
        # Get user index
        user_idx = self.user_metadata[self.user_metadata['user_id'] == user_id].index
        
        if len(user_idx) == 0:
            return []
        
        user_idx = user_idx[0]
        user_embedding = self.user_embeddings[user_idx].reshape(1, -1).astype('float32')
        
        # Find similar users
        distances, similar_user_indices = self.user_index.search(user_embedding, 20)
        similar_user_indices = similar_user_indices[0][1:]  # Exclude self
        
        # Get videos watched by similar users
        similar_users = self.user_metadata.iloc[similar_user_indices]['user_id'].values
        similar_user_videos = interaction_history[
            interaction_history['user_id'].isin(similar_users)
        ]['video_id'].value_counts().head(k)
        
        return similar_user_videos.index.tolist()
    
    def content_based_filtering(self, user_interests: List[str], k: int = 100) -> List[str]:
        """Get videos based on user interests and content similarity."""
        # Map interests to video categories
        interest_to_category = {
            'sports': ['sports', 'fitness'],
            'music': ['music', 'entertainment'],
            'gaming': ['gaming', 'entertainment'],
            'cooking': ['howto', 'education'],
            'travel': ['travel', 'entertainment'],
            'tech': ['tech', 'education'],
            'fashion': ['fashion', 'entertainment'],
            'fitness': ['sports', 'howto'],
            'education': ['education', 'howto'],
            'comedy': ['comedy', 'entertainment']
        }
        
        relevant_categories = []
        for interest in user_interests:
            relevant_categories.extend(interest_to_category.get(interest, ['entertainment']))
        
        relevant_categories = list(set(relevant_categories))
        
        # Filter videos by category
        relevant_videos = self.video_metadata[
            self.video_metadata['category'].isin(relevant_categories)
        ].nlargest(k, 'engagement_rate')
        
        return relevant_videos['video_id'].tolist()
    
    def hybrid_retrieval(self, user_id: str, user_embedding: Optional[np.ndarray] = None,
                        interaction_history: Optional[pd.DataFrame] = None) -> List[str]:
        """Combine multiple retrieval strategies for candidate generation."""
        candidates = {}
        
        # Get user metadata
        user_data = self.user_metadata[self.user_metadata['user_id'] == user_id]
        
        if len(user_data) == 0:
            # Fallback to popularity-based for new users
            return self.popularity_based_retrieval(self.n_candidates)
        
        user_interests = user_data.iloc[0]['interests']
        
        # 1. Vector similarity search
        if user_embedding is not None:
            vector_k = int(self.n_candidates * self.strategy_weights['vector_similarity'])
            video_indices, _ = self.vector_search(user_embedding, vector_k)
            vector_videos = self.video_metadata.iloc[video_indices]['video_id'].tolist()
            for i, video_id in enumerate(vector_videos):
                candidates[video_id] = candidates.get(video_id, 0) + (vector_k - i) * self.strategy_weights['vector_similarity']
        
        # 2. Popularity-based
        pop_k = int(self.n_candidates * self.strategy_weights['popularity'])
        pop_videos = self.popularity_based_retrieval(pop_k)
        for i, video_id in enumerate(pop_videos):
            candidates[video_id] = candidates.get(video_id, 0) + (pop_k - i) * self.strategy_weights['popularity']
        
        # 3. Content-based
        content_k = int(self.n_candidates * self.strategy_weights['content_based'])
        content_videos = self.content_based_filtering(user_interests, content_k)
        for i, video_id in enumerate(content_videos):
            candidates[video_id] = candidates.get(video_id, 0) + (content_k - i) * self.strategy_weights['content_based']
        
        # 4. Collaborative filtering (if interaction history available)
        if interaction_history is not None and len(interaction_history) > 0:
            collab_k = int(self.n_candidates * self.strategy_weights['collaborative'])
            collab_videos = self.collaborative_filtering(user_id, interaction_history, collab_k)
            for i, video_id in enumerate(collab_videos):
                candidates[video_id] = candidates.get(video_id, 0) + (collab_k - i) * self.strategy_weights['collaborative']
        
        # Sort by combined score and return top candidates
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [video_id for video_id, _ in sorted_candidates[:self.n_candidates]]
    
    def batch_retrieval(self, user_ids: List[str], 
                       user_embeddings: Optional[np.ndarray] = None,
                       interaction_history: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
        """Generate candidates for multiple users."""
        print(f"Generating candidates for {len(user_ids)} users...")
        
        results = {}
        for i, user_id in enumerate(tqdm(user_ids, desc="Retrieving candidates")):
            user_emb = user_embeddings[i] if user_embeddings is not None else None
            user_history = interaction_history[interaction_history['user_id'] == user_id] if interaction_history is not None else None
            
            candidates = self.hybrid_retrieval(user_id, user_emb, user_history)
            results[user_id] = candidates
        
        return results
    
    def save_index(self, path: Optional[Path] = None):
        """Save FAISS indices and metadata."""
        if path is None:
            path = self.features_dir / "faiss_indices"
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indices
        faiss.write_index(self.video_index, str(path / "video_index.faiss"))
        faiss.write_index(self.user_index, str(path / "user_index.faiss"))
        
        # Save metadata
        self.video_metadata.to_parquet(path / "video_metadata.parquet")
        self.user_metadata.to_parquet(path / "user_metadata.parquet")
        
        # Save embeddings
        np.save(path / "video_embeddings.npy", self.video_embeddings)
        np.save(path / "user_embeddings.npy", self.user_embeddings)
        
        print(f"Indices saved to {path}")
    
    def load_index(self, path: Optional[Path] = None):
        """Load FAISS indices and metadata."""
        if path is None:
            path = self.features_dir / "faiss_indices"
        
        # Load FAISS indices
        self.video_index = faiss.read_index(str(path / "video_index.faiss"))
        self.user_index = faiss.read_index(str(path / "user_index.faiss"))
        
        # Load metadata
        self.video_metadata = pd.read_parquet(path / "video_metadata.parquet")
        self.user_metadata = pd.read_parquet(path / "user_metadata.parquet")
        
        # Load embeddings
        self.video_embeddings = np.load(path / "video_embeddings.npy")
        self.user_embeddings = np.load(path / "user_embeddings.npy")
        
        print(f"Indices loaded from {path}")


def main():
    """Main function to build and test FAISS retrieval."""
    # Initialize retriever
    retriever = FAISSRetriever()
    
    # Load features
    retriever.load_features()
    
    # Build indices
    retriever.build_video_index()
    retriever.build_user_index()
    
    # Save indices
    retriever.save_index()
    
    # Test retrieval for a sample user
    sample_user = retriever.user_metadata.iloc[0]['user_id']
    sample_embedding = retriever.user_embeddings[0]
    
    print(f"\nTesting retrieval for user: {sample_user}")
    candidates = retriever.hybrid_retrieval(sample_user, sample_embedding)
    
    print(f"Generated {len(candidates)} candidates")
    print(f"Sample candidates: {candidates[:10]}")
    
    # Test batch retrieval
    test_users = retriever.user_metadata['user_id'].head(10).tolist()
    test_embeddings = retriever.user_embeddings[:10]
    
    batch_results = retriever.batch_retrieval(test_users, test_embeddings)
    print(f"\nBatch retrieval completed for {len(batch_results)} users")


if __name__ == "__main__":
    main()