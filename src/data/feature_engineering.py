"""
Feature engineering pipeline for recommendation system.
Processes raw data into features for model training and inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path
import yaml
from datetime import datetime
# Removed unused torch import
from tqdm import tqdm


class FeatureEngineer:
    """Feature engineering for user and video data."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize feature engineer with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(self.config["paths"]["data"]["raw"])
        self.features_dir = Path(self.config["paths"]["data"]["features"])
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encoders and scalers
        self.user_scaler = StandardScaler()
        self.video_scaler = StandardScaler()
        # OrdinalEncoder for categorical features (not target labels)
        self.category_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        # Activity level has natural ordering: low < medium < high < very_high
        self.activity_encoder = OrdinalEncoder(
            categories=[["low", "medium", "high", "very_high"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        
        # Store top interests (will be set during fit)
        self.top_interests = []

        # Feature dimensions from config
        self.user_emb_dim = self.config["model"]["embedding"]["user_dim"]
        self.video_emb_dim = self.config["model"]["embedding"]["video_dim"]
        self.context_dim = self.config["model"]["embedding"]["context_dim"]

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data from parquet files."""
        users_df = pd.read_parquet(self.data_dir / "users.parquet")
        videos_df = pd.read_parquet(self.data_dir / "videos.parquet")
        interactions_df = pd.read_parquet(self.data_dir / "interactions.parquet")

        print(f"Loaded data:")
        print(f"  Users: {len(users_df):,}")
        print(f"  Videos: {len(videos_df):,}")
        print(f"  Interactions: {len(interactions_df):,}")

        return users_df, videos_df, interactions_df

    def process_user_features(
        self, users_df: pd.DataFrame, interactions_df: pd.DataFrame, fit_scalers: bool = True
    ) -> pd.DataFrame:
        """Process user features including behavioral patterns."""
        print("\nProcessing user features...")

        # Calculate user statistics from interactions
        user_stats = (
            interactions_df.groupby("user_id")
            .agg(
                {
                    "interaction_id": "count",
                    "watch_time_ratio": "mean",
                    "liked": "mean",
                    "shared": "mean",
                    "skipped": "mean",
                    "video_duration": "mean",
                }
            )
            .rename(
                columns={
                    "interaction_id": "total_interactions",
                    "watch_time_ratio": "avg_watch_ratio",
                    "liked": "like_rate",
                    "shared": "share_rate",
                    "skipped": "skip_rate",
                    "video_duration": "avg_video_duration",
                }
            )
        )

        # Calculate session statistics
        session_stats = (
            interactions_df.groupby("user_id")["session_id"]
            .nunique()
            .rename("num_sessions")
        )
        user_stats = user_stats.join(session_stats)

        # Merge with user data
        users_features = users_df.merge(
            user_stats, left_on="user_id", right_index=True, how="left"
        )

        # Fill missing values with explicit defaults
        # Using 0 for counts/rates is reasonable for users with no interactions
        fill_values = {
            'total_interactions': 0,
            'avg_watch_ratio': 0,
            'like_rate': 0,
            'share_rate': 0,
            'skip_rate': 0,
            'avg_video_duration': 0,
            'num_sessions': 0
        }
        users_features = users_features.fillna(value=fill_values)

        # Process age
        users_features["age_group"] = pd.cut(
            users_features["age"],
            bins=[0, 18, 25, 35, 50, 100],
            labels=["<18", "18-25", "25-35", "35-50", "50+"],
        )

        # Encode activity level (ordinal feature with natural ordering)
        if fit_scalers:
            users_features["activity_level_encoded"] = self.activity_encoder.fit_transform(
                users_features[["activity_level"]]
            ).ravel()
        else:
            users_features["activity_level_encoded"] = self.activity_encoder.transform(
                users_features[["activity_level"]]
            ).ravel()

        # Process interests (one-hot encoding for top interests)
        # ISSUE: Should determine top interests from training data only
        if fit_scalers:
            # Collect all interests from training data
            all_interests_counter = {}
            for interests in users_features["interests"]:
                for interest in interests:
                    all_interests_counter[interest] = all_interests_counter.get(interest, 0) + 1
            # Store top 15 interests for later use
            self.top_interests = sorted(all_interests_counter.keys(), 
                                       key=lambda x: all_interests_counter[x], 
                                       reverse=True)[:15]
        
        # Apply one-hot encoding for selected interests
        for interest in self.top_interests:
            users_features[f"interest_{interest}"] = users_features["interests"].apply(
                lambda x: 1 if interest in x else 0
            )

        # Calculate user embedding based on behavior
        behavior_features = users_features[
            [
                "total_interactions",
                "avg_watch_ratio",
                "like_rate",
                "share_rate",
                "skip_rate",
                "num_sessions",
            ]
        ].values

        # Normalize behavioral features
        if fit_scalers:
            behavior_features_scaled = self.user_scaler.fit_transform(behavior_features)
        else:
            behavior_features_scaled = self.user_scaler.transform(behavior_features)

        # Create user embeddings (combine initial embeddings with behavioral features)
        user_embeddings = []
        for i, (idx, row) in enumerate(users_features.iterrows()):
            # Get initial embedding
            initial_emb = np.array(row["initial_embedding"])[: self.user_emb_dim // 2]

            # Create behavioral embedding - use row index i for behavior_features_scaled
            behavior_emb = np.zeros(self.user_emb_dim // 2)
            if i < len(behavior_features_scaled):
                behavior_emb[
                    : min(len(behavior_features_scaled[i]), self.user_emb_dim // 2)
                ] = behavior_features_scaled[i][: self.user_emb_dim // 2]

            # Combine embeddings
            user_emb = np.concatenate([initial_emb, behavior_emb])
            user_embeddings.append(user_emb)

        users_features["user_embedding"] = user_embeddings

        print(f"  Processed {len(users_features)} users")
        print(f"  User feature dimensions: {users_features.shape[1]}")

        return users_features

    def process_video_features(
        self, videos_df: pd.DataFrame, interactions_df: pd.DataFrame, fit_scalers: bool = True
    ) -> pd.DataFrame:
        """Process video features including engagement metrics."""
        print("\nProcessing video features...")

        # Calculate video statistics from interactions
        video_stats = (
            interactions_df.groupby("video_id")
            .agg(
                {
                    "interaction_id": "count",
                    "watch_time_ratio": "mean",
                    "liked": "sum",
                    "shared": "sum",
                    "skipped": "sum",
                }
            )
            .rename(
                columns={
                    "interaction_id": "total_views",
                    "watch_time_ratio": "avg_completion_rate",
                    "liked": "total_likes",
                    "shared": "total_shares",
                    "skipped": "total_skips",
                }
            )
        )

        # Calculate engagement rate
        video_stats["engagement_rate"] = (
            video_stats["total_likes"] + video_stats["total_shares"]
        ) / video_stats["total_views"].clip(lower=1)

        # Merge with video data
        videos_features = videos_df.merge(
            video_stats, left_on="video_id", right_index=True, how="left"
        )

        # Fill missing values with explicit defaults
        # Using 0 for engagement metrics is reasonable for new/unwatched videos
        fill_values = {
            'total_views': 0,
            'avg_completion_rate': 0,
            'total_likes': 0,
            'total_shares': 0,
            'total_skips': 0,
            'engagement_rate': 0
        }
        videos_features = videos_features.fillna(value=fill_values)

        # Encode category (nominal feature - no natural ordering)
        if fit_scalers:
            videos_features["category_encoded"] = self.category_encoder.fit_transform(
                videos_features[["category"]]
            ).ravel()
        else:
            videos_features["category_encoded"] = self.category_encoder.transform(
                videos_features[["category"]]
            ).ravel()

        # Process tags (TF-IDF on concatenated tags)
        videos_features["tags_text"] = videos_features["tags"].apply(
            lambda x: " ".join(x)
        )

        # Create sparse tag features
        if fit_scalers:
            tag_features_sparse = self.tfidf_vectorizer.fit_transform(
                videos_features["tags_text"]
            )
        else:
            tag_features_sparse = self.tfidf_vectorizer.transform(
                videos_features["tags_text"]
            )
        tag_features_dense = tag_features_sparse.toarray()

        # Process video embeddings
        video_embeddings = []
        for i, (idx, row) in enumerate(videos_features.iterrows()):
            # Get base embedding
            base_emb = np.array(row["video_embedding"])[
                : self.video_emb_dim - tag_features_dense.shape[1]
            ]

            # Add tag features - use row index i for tag_features_dense
            tag_emb = (
                tag_features_dense[i]
                if i < len(tag_features_dense)
                else np.zeros(tag_features_dense.shape[1] if tag_features_dense.shape[0] > 0 else 0)
            )

            # Combine embeddings
            video_emb = np.concatenate([base_emb, tag_emb])

            # Ensure correct dimension
            if len(video_emb) < self.video_emb_dim:
                video_emb = np.pad(video_emb, (0, self.video_emb_dim - len(video_emb)))
            else:
                video_emb = video_emb[: self.video_emb_dim]

            video_embeddings.append(video_emb)

        videos_features["video_embedding_processed"] = video_embeddings

        # Normalize numerical features
        numerical_features = [
            "duration",
            "view_count",
            "engagement_score",
            "quality_score",
            "total_views",
            "engagement_rate",
        ]
        
        # Stack all numerical features for a single fit/transform
        available_features = [f for f in numerical_features if f in videos_features.columns]
        if available_features:
            numerical_data = videos_features[available_features].values
            if fit_scalers:
                scaled_data = self.video_scaler.fit_transform(numerical_data)
            else:
                scaled_data = self.video_scaler.transform(numerical_data)
            
            for i, feat in enumerate(available_features):
                videos_features[f"{feat}_scaled"] = scaled_data[:, i]

        print(f"  Processed {len(videos_features)} videos")
        print(f"  Video feature dimensions: {videos_features.shape[1]}")

        return videos_features

    def create_interaction_features(
        self,
        interactions_df: pd.DataFrame,
        users_features: pd.DataFrame,
        videos_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create interaction-level features for training."""
        print("\nCreating interaction features...")

        # Merge user and video features
        interactions_features = interactions_df.merge(
            users_features[
                ["user_id", "user_embedding", "age_group", "activity_level_encoded"]
            ],
            on="user_id",
            how="left",
        )

        interactions_features = interactions_features.merge(
            videos_features[
                [
                    "video_id",
                    "video_embedding_processed",
                    "category_encoded",
                    "duration",
                    "engagement_rate",
                ]
            ],
            on="video_id",
            how="left",
        )

        # Create temporal features
        interactions_features["hour"] = pd.to_datetime(
            interactions_features["timestamp"]
        ).dt.hour
        interactions_features["day_of_week"] = pd.to_datetime(
            interactions_features["timestamp"]
        ).dt.dayofweek
        interactions_features["is_weekend"] = (
            interactions_features["day_of_week"].isin([5, 6]).astype(int)
        )

        # Create context features
        context_features = []
        for _, row in tqdm(
            interactions_features.iterrows(),
            total=len(interactions_features),
            desc="Creating context features",
        ):
            context = np.zeros(self.context_dim)
            context[0] = row["hour"] / 24.0  # Normalized hour
            context[1] = row["day_of_week"] / 7.0  # Normalized day
            context[2] = row["is_weekend"]
            context[3] = row["position_in_session"] / 20.0  # Normalized position
            context[4] = row["watch_time_ratio"]

            context_features.append(context)

        interactions_features["context_features"] = context_features

        # Create labels for training
        interactions_features["label_ctr"] = interactions_features["liked"].astype(
            float
        )
        interactions_features["label_watch_time"] = interactions_features[
            "watch_time_ratio"
        ]
        interactions_features["label_skip"] = interactions_features["skipped"].astype(
            float
        )

        print(f"  Created features for {len(interactions_features)} interactions")

        return interactions_features

    def create_training_data(
        self, interactions_features: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[Dict, Dict]:
        """Create training and validation datasets."""
        print("\nCreating training and validation sets...")

        # Sort by timestamp for temporal split
        interactions_features = interactions_features.sort_values("timestamp")

        # Split data
        split_idx = int(len(interactions_features) * (1 - test_size))
        train_df = interactions_features.iloc[:split_idx]
        val_df = interactions_features.iloc[split_idx:]

        print(f"  Training samples: {len(train_df):,}")
        print(f"  Validation samples: {len(val_df):,}")

        # Create feature dictionaries
        def create_feature_dict(df):
            return {
                "user_embeddings": np.stack(df["user_embedding"].values),
                "video_embeddings": np.stack(df["video_embedding_processed"].values),
                "context_features": np.stack(df["context_features"].values),
                "labels_ctr": df["label_ctr"].values,
                "labels_watch_time": df["label_watch_time"].values,
                "user_ids": df["user_id"].values,
                "video_ids": df["video_id"].values,
            }

        train_data = create_feature_dict(train_df)
        val_data = create_feature_dict(val_df)

        return train_data, val_data

    def save_features(
        self,
        users_features: pd.DataFrame,
        videos_features: pd.DataFrame,
        train_data: Dict,
        val_data: Dict,
    ):
        """Save processed features to disk."""
        print("\nSaving features...")

        # Save user features
        users_features.to_parquet(self.features_dir / "users_features.parquet")

        # Save video features
        videos_features.to_parquet(self.features_dir / "videos_features.parquet")

        # Save training data
        with open(self.features_dir / "train_data.pkl", "wb") as f:
            pickle.dump(train_data, f)

        with open(self.features_dir / "val_data.pkl", "wb") as f:
            pickle.dump(val_data, f)

        # Save encoders and scalers
        with open(self.features_dir / "encoders.pkl", "wb") as f:
            pickle.dump(
                {
                    "user_scaler": self.user_scaler,
                    "video_scaler": self.video_scaler,
                    "category_encoder": self.category_encoder,
                    "activity_encoder": self.activity_encoder,
                    "tfidf_vectorizer": self.tfidf_vectorizer,
                },
                f,
            )

        print(f"  Features saved to {self.features_dir}")

    def process_all(self):
        """Run complete feature engineering pipeline."""
        print("Starting feature engineering pipeline...")

        # Load raw data
        users_df, videos_df, interactions_df = self.load_raw_data()
        
        # Split interactions first for proper train/val split
        interactions_df = interactions_df.sort_values("timestamp")
        split_idx = int(len(interactions_df) * 0.8)
        train_interactions = interactions_df.iloc[:split_idx]
        val_interactions = interactions_df.iloc[split_idx:]
        
        print(f"\nSplit interactions:")
        print(f"  Training interactions: {len(train_interactions):,}")
        print(f"  Validation interactions: {len(val_interactions):,}")

        # Process features - fit scalers only on training data
        users_features_train = self.process_user_features(users_df, train_interactions, fit_scalers=True)
        videos_features_train = self.process_video_features(videos_df, train_interactions, fit_scalers=True)
        
        # Process validation data using fitted scalers
        users_features_val = self.process_user_features(users_df, interactions_df, fit_scalers=False)
        videos_features_val = self.process_video_features(videos_df, interactions_df, fit_scalers=False)
        
        # Create interaction features for both sets
        train_interactions_features = self.create_interaction_features(
            train_interactions, users_features_train, videos_features_train
        )
        val_interactions_features = self.create_interaction_features(
            val_interactions, users_features_val, videos_features_val
        )
        
        # Create training data from already split interactions
        def create_feature_dict(df):
            return {
                "user_embeddings": np.stack(df["user_embedding"].values),
                "video_embeddings": np.stack(df["video_embedding_processed"].values),
                "context_features": np.stack(df["context_features"].values),
                "labels_ctr": df["label_ctr"].values,
                "labels_watch_time": df["label_watch_time"].values,
                "user_ids": df["user_id"].values,
                "video_ids": df["video_id"].values,
            }
        
        train_data = create_feature_dict(train_interactions_features)
        val_data = create_feature_dict(val_interactions_features)
        
        print(f"\nFinal dataset sizes:")
        print(f"  Training samples: {len(train_data['user_ids']):,}")
        print(f"  Validation samples: {len(val_data['user_ids']):,}")

        # Save everything (use the full processed features for saving)
        self.save_features(users_features_val, videos_features_val, train_data, val_data)

        print("\nFeature engineering complete!")
        return users_features_val, videos_features_val, train_data, val_data


def main():
    """Main function to run feature engineering."""
    engineer = FeatureEngineer()
    _, _, train_data, val_data = engineer.process_all()

    # Print summary statistics
    print("\nFeature Summary:")
    print(f"User embedding shape: {train_data['user_embeddings'].shape}")
    print(f"Video embedding shape: {train_data['video_embeddings'].shape}")
    print(f"Context features shape: {train_data['context_features'].shape}")
    print(f"CTR rate in training: {train_data['labels_ctr'].mean():.2%}")
    print(f"Avg watch time ratio: {train_data['labels_watch_time'].mean():.2f}")
    
    # Validation set statistics for comparison
    print(f"\nValidation set:")
    print(f"CTR rate in validation: {val_data['labels_ctr'].mean():.2%}")
    print(f"Avg watch time ratio: {val_data['labels_watch_time'].mean():.2f}")


if __name__ == "__main__":
    main()
