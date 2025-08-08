"""
Batch synthetic data generator for YouTube Shorts recommendation system.
Generates data iteratively to handle large datasets without timeouts.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm
import pyarrow.parquet as pq
from pathlib import Path
import pickle
import click


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class BatchSyntheticDataGenerator:
    """Generate synthetic data in batches for recommendation system."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.seed = config.get('seed', 42)
        
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create output directories
        self.output_dir = Path(config['paths']['data']['raw'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State file for resuming
        self.state_file = self.output_dir / ".generation_state.pkl"
        
    def save_state(self, state: Dict):
        """Save generation state for resuming."""
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self) -> Optional[Dict]:
        """Load generation state if exists."""
        if self.state_file.exists():
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def generate_users_batch(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Generate a batch of synthetic user profiles."""
        interests_list = self.data_config['user_features']['interests']
        age_range = self.data_config['user_features']['age_range']
        
        users = []
        for user_id in range(start_idx, end_idx):
            # Generate user interests (each user has 1-5 interests)
            n_interests = random.randint(1, 5)
            user_interests = random.sample(interests_list, n_interests)
            
            # Generate demographics
            age = random.randint(*age_range)
            
            # Determine user activity level
            activity_level = np.random.choice(
                ['low', 'medium', 'high'],
                p=[0.3, 0.5, 0.2]
            )
            
            # User embedding (will be computed from interactions later)
            initial_embedding = np.random.randn(128).tolist()
            
            users.append({
                'user_id': f'user_{user_id:06d}',
                'age': age,
                'interests': user_interests,
                'activity_level': activity_level,
                'registration_date': datetime.now() - timedelta(days=random.randint(1, 365)),
                'initial_embedding': initial_embedding
            })
        
        return pd.DataFrame(users)
    
    def generate_videos_batch(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Generate a batch of synthetic video metadata."""
        categories = self.data_config['video_features']['categories']
        duration_range = self.data_config['video_features']['duration_range']
        tags_range = self.data_config['video_features']['tags_per_video']
        
        # Generate a pool of tags
        all_tags = [
            "funny", "tutorial", "vlog", "challenge", "prank", "review",
            "unboxing", "reaction", "compilation", "highlights", "tips",
            "tricks", "diy", "lifestyle", "motivation", "storytime",
            "haul", "routine", "transformation", "experiment", "trending"
        ]
        
        videos = []
        for video_id in range(start_idx, end_idx):
            # Video metadata
            category = random.choice(categories)
            duration = random.randint(*duration_range)
            n_tags = random.randint(*tags_range)
            tags = random.sample(all_tags, min(n_tags, len(all_tags)))
            
            # Video quality scores (for ranking)
            engagement_score = np.random.beta(2, 5)  # Skewed towards lower values
            quality_score = np.random.beta(3, 3)  # Normal distribution
            
            # Upload time (videos uploaded over past year)
            upload_date = datetime.now() - timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # View count (power law distribution)
            view_count = int(np.random.pareto(1.5) * 1000)
            
            # Video embedding (pre-computed for similarity)
            video_embedding = np.random.randn(768).tolist()
            
            videos.append({
                'video_id': f'video_{video_id:08d}',
                'category': category,
                'duration': duration,
                'tags': tags,
                'engagement_score': engagement_score,
                'quality_score': quality_score,
                'upload_date': upload_date,
                'view_count': view_count,
                'video_embedding': video_embedding
            })
        
        return pd.DataFrame(videos)
    
    def generate_interactions_batch(self, users_df: pd.DataFrame, videos_df: pd.DataFrame, 
                                   n_interactions: int, interaction_offset: int = 0) -> pd.DataFrame:
        """Generate a batch of user-video interactions."""
        interaction_config = self.data_config['interaction']
        temporal_config = self.data_config['temporal']
        
        interactions = []
        
        # Create user interest to category mapping
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
            'comedy': ['comedy', 'entertainment'],
            'news': ['news'],
            'science': ['education', 'tech'],
            'art': ['entertainment', 'education'],
            'diy': ['howto', 'education'],
            'pets': ['entertainment']
        }
        
        # Generate interactions
        interaction_id = interaction_offset
        
        while len(interactions) < n_interactions:
            # Sample user
            user = users_df.sample(1).iloc[0]
            user_interests = user['interests']
            activity_level = user['activity_level']
            
            # Determine session length based on activity level
            session_length_map = {'low': (3, 8), 'medium': (5, 15), 'high': (10, 30)}
            session_length = random.randint(*session_length_map[activity_level])
            
            # Generate session timestamp
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, temporal_config['days_of_history']),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Create session
            session_id = f"session_{interaction_id:08d}"
            
            for position in range(min(session_length, n_interactions - len(interactions))):
                # Select video based on user interests (70% relevant, 30% exploration)
                if random.random() < 0.7 and len(user_interests) > 0:
                    # Select relevant video
                    interest = random.choice(user_interests)
                    relevant_categories = interest_to_category.get(interest, ['entertainment'])
                    relevant_videos = videos_df[videos_df['category'].isin(relevant_categories)]
                    
                    if len(relevant_videos) > 0:
                        # Bias towards popular videos
                        weights = np.power(relevant_videos['view_count'].values, 0.5)
                        weights = weights / weights.sum()
                        video_idx = np.random.choice(len(relevant_videos), p=weights)
                        video = relevant_videos.iloc[video_idx]
                    else:
                        video = videos_df.sample(1).iloc[0]
                else:
                    # Explore random video
                    video = videos_df.sample(1).iloc[0]
                
                # Generate interaction signals
                watch_time_ratio = np.random.beta(3, 2)  # Skewed towards completion
                watch_time = int(video['duration'] * watch_time_ratio)
                
                # Like probability increases with watch time
                like_prob = interaction_config['like_probability'] * (watch_time_ratio ** 2)
                liked = random.random() < like_prob
                
                # Share probability
                share_prob = interaction_config['share_probability'] * (1.5 if liked else 0.5)
                shared = random.random() < share_prob
                
                # Skip detection
                skipped = watch_time_ratio < 0.3
                
                interactions.append({
                    'interaction_id': f"int_{interaction_id:010d}",
                    'user_id': user['user_id'],
                    'video_id': video['video_id'],
                    'session_id': session_id,
                    'timestamp': timestamp + timedelta(seconds=position * 30),
                    'watch_time': watch_time,
                    'watch_time_ratio': watch_time_ratio,
                    'liked': liked,
                    'shared': shared,
                    'skipped': skipped,
                    'position_in_session': position,
                    'video_duration': video['duration'],
                    'video_category': video['category']
                })
                
                interaction_id += 1
        
        return pd.DataFrame(interactions)
    
    def generate_all(self, batch_size: int = 1000):
        """Generate all data in batches."""
        print("Starting batch synthetic data generation...")
        
        # Check if we're resuming
        state = self.load_state()
        if state:
            print(f"Resuming from previous state...")
            users_generated = state.get('users_generated', 0)
            videos_generated = state.get('videos_generated', 0)
            interactions_generated = state.get('interactions_generated', 0)
        else:
            users_generated = 0
            videos_generated = 0
            interactions_generated = 0
        
        n_users = self.data_config['n_users']
        n_videos = self.data_config['n_videos']
        n_interactions = self.data_config['n_interactions']
        
        # Generate users in batches
        if users_generated < n_users:
            print(f"\nGenerating users ({users_generated}/{n_users})...")
            users_list = []
            
            with tqdm(total=n_users - users_generated, desc="Users") as pbar:
                while users_generated < n_users:
                    batch_end = min(users_generated + batch_size, n_users)
                    batch_df = self.generate_users_batch(users_generated, batch_end)
                    users_list.append(batch_df)
                    
                    pbar.update(batch_end - users_generated)
                    users_generated = batch_end
                    
                    # Save state
                    self.save_state({
                        'users_generated': users_generated,
                        'videos_generated': videos_generated,
                        'interactions_generated': interactions_generated
                    })
            
            # Combine and save all users
            users_df = pd.concat(users_list, ignore_index=True)
            users_path = self.output_dir / "users.parquet"
            users_df.to_parquet(users_path, engine='pyarrow')
            print(f"Saved {len(users_df)} users")
        else:
            users_path = self.output_dir / "users.parquet"
            users_df = pd.read_parquet(users_path)
            print(f"Users already generated: {len(users_df)}")
        
        # Generate videos in batches
        if videos_generated < n_videos:
            print(f"\nGenerating videos ({videos_generated}/{n_videos})...")
            videos_list = []
            
            with tqdm(total=n_videos - videos_generated, desc="Videos") as pbar:
                while videos_generated < n_videos:
                    batch_end = min(videos_generated + batch_size * 2, n_videos)  # Larger batches for videos
                    batch_df = self.generate_videos_batch(videos_generated, batch_end)
                    videos_list.append(batch_df)
                    
                    pbar.update(batch_end - videos_generated)
                    videos_generated = batch_end
                    
                    # Save state
                    self.save_state({
                        'users_generated': users_generated,
                        'videos_generated': videos_generated,
                        'interactions_generated': interactions_generated
                    })
            
            # Combine and save all videos
            videos_df = pd.concat(videos_list, ignore_index=True)
            videos_path = self.output_dir / "videos.parquet"
            videos_df.to_parquet(videos_path, engine='pyarrow')
            print(f"Saved {len(videos_df)} videos")
        else:
            videos_path = self.output_dir / "videos.parquet"
            videos_df = pd.read_parquet(videos_path)
            print(f"Videos already generated: {len(videos_df)}")
        
        # Generate interactions in batches
        if interactions_generated < n_interactions:
            print(f"\nGenerating interactions ({interactions_generated}/{n_interactions})...")
            interactions_list = []
            
            # Load existing interactions if any
            interactions_path = self.output_dir / "interactions.parquet"
            if interactions_path.exists() and interactions_generated > 0:
                existing_interactions = pd.read_parquet(interactions_path)
                interactions_list.append(existing_interactions)
            
            interactions_batch_size = 10000  # Smaller batches for interactions
            
            with tqdm(total=n_interactions - interactions_generated, desc="Interactions") as pbar:
                while interactions_generated < n_interactions:
                    batch_size_actual = min(interactions_batch_size, n_interactions - interactions_generated)
                    batch_df = self.generate_interactions_batch(
                        users_df, videos_df, batch_size_actual, interactions_generated
                    )
                    interactions_list.append(batch_df)
                    
                    pbar.update(batch_size_actual)
                    interactions_generated += batch_size_actual
                    
                    # Save periodically
                    if interactions_generated % 50000 == 0 or interactions_generated >= n_interactions:
                        # Combine and save
                        interactions_df = pd.concat(interactions_list, ignore_index=True)
                        interactions_df.to_parquet(interactions_path, engine='pyarrow')
                        print(f"  Checkpoint: Saved {len(interactions_df)} interactions")
                        
                        # Clear list but keep the dataframe reference
                        interactions_list = [interactions_df]
                    
                    # Save state
                    self.save_state({
                        'users_generated': users_generated,
                        'videos_generated': videos_generated,
                        'interactions_generated': interactions_generated
                    })
            
            # Final save
            interactions_df = pd.concat(interactions_list, ignore_index=True)
            interactions_df.to_parquet(interactions_path, engine='pyarrow')
            print(f"Saved {len(interactions_df)} interactions")
        else:
            interactions_path = self.output_dir / "interactions.parquet"
            interactions_df = pd.read_parquet(interactions_path)
            print(f"Interactions already generated: {len(interactions_df)}")
        
        # Save statistics
        self._save_statistics(users_df, videos_df, interactions_df)
        
        # Clean up state file
        if self.state_file.exists():
            os.remove(self.state_file)
        
        print("\nBatch data generation complete!")
        return users_df, videos_df, interactions_df
    
    def _save_statistics(self, users_df: pd.DataFrame, videos_df: pd.DataFrame,
                        interactions_df: pd.DataFrame):
        """Save summary statistics of generated data."""
        stats = {
            'n_users': len(users_df),
            'n_videos': len(videos_df),
            'n_interactions': len(interactions_df),
            'avg_interactions_per_user': len(interactions_df) / len(users_df),
            'avg_interactions_per_video': len(interactions_df) / len(videos_df),
            'unique_sessions': interactions_df['session_id'].nunique(),
            'date_range': {
                'start': str(interactions_df['timestamp'].min()),
                'end': str(interactions_df['timestamp'].max())
            },
            'engagement_stats': {
                'like_rate': interactions_df['liked'].mean(),
                'share_rate': interactions_df['shared'].mean(),
                'skip_rate': interactions_df['skipped'].mean(),
                'avg_watch_time_ratio': interactions_df['watch_time_ratio'].mean()
            }
        }
        
        stats_path = self.output_dir / "data_statistics.yaml"
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        print(f"\nData Statistics:")
        print(f"  Users: {stats['n_users']:,}")
        print(f"  Videos: {stats['n_videos']:,}")
        print(f"  Interactions: {stats['n_interactions']:,}")
        print(f"  Avg interactions/user: {stats['avg_interactions_per_user']:.1f}")
        print(f"  Like rate: {stats['engagement_stats']['like_rate']:.2%}")
        print(f"  Skip rate: {stats['engagement_stats']['skip_rate']:.2%}")


@click.command()
@click.option('--config', default='configs/config.yaml', help='Path to configuration file')
@click.option('--batch-size', default=1000, type=int, help='Batch size for generation')
@click.option('--reset', is_flag=True, help='Reset and start from beginning')
def main(config: str, batch_size: int, reset: bool):
    """Main function to run batch data generation."""
    # Load configuration
    config_data = load_config(config)
    
    # Initialize generator
    generator = BatchSyntheticDataGenerator(config_data)
    
    # Reset if requested
    if reset and generator.state_file.exists():
        os.remove(generator.state_file)
        print("Reset generation state")
    
    # Generate data
    users_df, videos_df, interactions_df = generator.generate_all(batch_size=batch_size)
    
    return users_df, videos_df, interactions_df


if __name__ == "__main__":
    main()