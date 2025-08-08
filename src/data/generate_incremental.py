#!/usr/bin/env python
"""
Incremental data generator that generates data in small chunks.
Run this multiple times to build up the full dataset.
"""

import click
import pandas as pd
from pathlib import Path
from generate_synthetic_batch import BatchSyntheticDataGenerator, load_config
import time


@click.command()
@click.option('--config', default='configs/config.yaml', help='Config file path')
@click.option('--chunk-size', default=50000, type=int, help='Interactions per chunk')
@click.option('--max-time', default=25, type=int, help='Max seconds per run')
def generate_chunk(config: str, chunk_size: int, max_time: int):
    """Generate data in chunks with time limit."""
    
    # Load config
    config_data = load_config(config)
    generator = BatchSyntheticDataGenerator(config_data)
    
    # Check current state
    state = generator.load_state()
    if not state:
        state = {
            'users_generated': 0,
            'videos_generated': 0,
            'interactions_generated': 0
        }
    
    n_users = config_data['data']['n_users']
    n_videos = config_data['data']['n_videos']
    n_interactions = config_data['data']['n_interactions']
    
    print(f"Current progress:")
    print(f"  Users: {state['users_generated']}/{n_users} ({state['users_generated']/n_users*100:.1f}%)")
    print(f"  Videos: {state['videos_generated']}/{n_videos} ({state['videos_generated']/n_videos*100:.1f}%)")
    print(f"  Interactions: {state['interactions_generated']}/{n_interactions} ({state['interactions_generated']/n_interactions*100:.1f}%)")
    print()
    
    start_time = time.time()
    
    # Generate users if needed
    if state['users_generated'] < n_users:
        print("Generating users...")
        users_list = []
        
        while state['users_generated'] < n_users and time.time() - start_time < max_time:
            batch_end = min(state['users_generated'] + 1000, n_users)
            batch_df = generator.generate_users_batch(state['users_generated'], batch_end)
            users_list.append(batch_df)
            state['users_generated'] = batch_end
            
            if state['users_generated'] % 2000 == 0:
                print(f"  Generated {state['users_generated']}/{n_users} users")
        
        if users_list:
            # Save users
            users_path = generator.output_dir / "users.parquet"
            if users_path.exists():
                existing = pd.read_parquet(users_path)
                users_df = pd.concat([existing] + users_list, ignore_index=True)
            else:
                users_df = pd.concat(users_list, ignore_index=True)
            
            users_df.to_parquet(users_path, engine='pyarrow')
            print(f"  Saved {len(users_df)} total users")
    
    # Generate videos if needed
    if state['videos_generated'] < n_videos and time.time() - start_time < max_time - 5:
        print("Generating videos...")
        videos_list = []
        
        while state['videos_generated'] < n_videos and time.time() - start_time < max_time - 5:
            batch_end = min(state['videos_generated'] + 2000, n_videos)
            batch_df = generator.generate_videos_batch(state['videos_generated'], batch_end)
            videos_list.append(batch_df)
            state['videos_generated'] = batch_end
            
            if state['videos_generated'] % 5000 == 0:
                print(f"  Generated {state['videos_generated']}/{n_videos} videos")
        
        if videos_list:
            # Save videos
            videos_path = generator.output_dir / "videos.parquet"
            if videos_path.exists():
                existing = pd.read_parquet(videos_path)
                videos_df = pd.concat([existing] + videos_list, ignore_index=True)
            else:
                videos_df = pd.concat(videos_list, ignore_index=True)
            
            videos_df.to_parquet(videos_path, engine='pyarrow')
            print(f"  Saved {len(videos_df)} total videos")
    
    # Generate interactions if users and videos are ready
    if (state['users_generated'] >= n_users and 
        state['videos_generated'] >= n_videos and 
        state['interactions_generated'] < n_interactions and
        time.time() - start_time < max_time - 2):
        
        print("Generating interactions...")
        
        # Load users and videos
        users_df = pd.read_parquet(generator.output_dir / "users.parquet")
        videos_df = pd.read_parquet(generator.output_dir / "videos.parquet")
        
        # Generate interactions
        interactions_to_generate = min(chunk_size, n_interactions - state['interactions_generated'])
        
        batch_df = generator.generate_interactions_batch(
            users_df, videos_df, 
            interactions_to_generate, 
            state['interactions_generated']
        )
        
        # Save interactions
        interactions_path = generator.output_dir / "interactions.parquet"
        if interactions_path.exists():
            existing = pd.read_parquet(interactions_path)
            interactions_df = pd.concat([existing, batch_df], ignore_index=True)
        else:
            interactions_df = batch_df
        
        interactions_df.to_parquet(interactions_path, engine='pyarrow')
        state['interactions_generated'] += interactions_to_generate
        
        print(f"  Generated {interactions_to_generate} interactions")
        print(f"  Total interactions: {state['interactions_generated']}/{n_interactions}")
    
    # Save state
    generator.save_state(state)
    
    # Check if complete
    if (state['users_generated'] >= n_users and 
        state['videos_generated'] >= n_videos and 
        state['interactions_generated'] >= n_interactions):
        
        print("\n✅ Data generation complete!")
        
        # Load all data for statistics
        users_df = pd.read_parquet(generator.output_dir / "users.parquet")
        videos_df = pd.read_parquet(generator.output_dir / "videos.parquet")
        interactions_df = pd.read_parquet(generator.output_dir / "interactions.parquet")
        
        # Save statistics
        generator._save_statistics(users_df, videos_df, interactions_df)
        
        # Clean up state file
        if generator.state_file.exists():
            generator.state_file.unlink()
    else:
        print(f"\n⏳ Generation in progress. Run again to continue.")
        print(f"   Elapsed time: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    generate_chunk()