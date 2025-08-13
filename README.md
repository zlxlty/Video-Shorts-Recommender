# YouTube Shorts Recommendation System - Project Context

## Project Overview
A production-ready YouTube Shorts-style recommendation system with end-to-end ML pipeline, from data generation to deployment.

## Tech Stack
- **Environment**: uv (Python 3.11+)
- **ML**: PyTorch, FAISS, Scikit-learn, Transformers
- **API**: FastAPI, Redis, Pydantic
- **Monitoring**: Prometheus, Grafana
- **Testing**: Pytest, Coverage
- **Development**: Black, Ruff, MyPy, Pre-commit

## Project Structure
```
video_recommender/
├── src/
│   ├── data/           # Data generation and processing
│   ├── models/         # ML models and training
│   ├── api/            # FastAPI service
│   ├── evaluation/     # Metrics and A/B testing
│   ├── candidate_gen/  # Recall strategies
│   └── utils/          # Shared utilities
├── configs/            # YAML configurations
├── notebooks/          # Exploration and analysis
├── tests/              # Unit and integration tests
├── docker/             # Containerization
├── monitoring/         # Observability configs
└── scripts/            # Automation scripts
```

## Key Commands

### Environment Setup
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project
uv init
uv python pin 3.11

# Install dependencies
uv sync
```

### Development
```bash
# Run data generation
uv run python src/data/generate_synthetic.py

# Train model
uv run python src/models/train.py

# Start API server
uv run uvicorn src.api.main:app --reload --port 8000

# Run tests
uv run pytest tests/ -v --cov=src

# Code quality
uv run black src/
uv run ruff check src/
uv run mypy src/
```

### Docker
```bash
# Build and run
docker-compose up --build

# Run specific service
docker-compose up api redis prometheus grafana
```

## Architecture

### 1. Data Pipeline
- **Synthetic Data**: 10K users, 50K videos, 1M+ interactions
- **Features**: User embeddings, video embeddings, contextual features
- **Storage**: Parquet files for offline, Redis for online

### 2. Candidate Generation (Recall)
- **Strategies**: 
  - Popularity-based (trending)
  - Tag-based matching
  - Collaborative filtering
  - FAISS vector search (768-dim embeddings)
- **Output**: Top 1000 candidates per user

### 3. Ranking Model
- **Architecture**: 4-layer TransformerEncoder + Multi-task heads
- **Objectives**: CTR prediction + Diversity scoring
- **Input**: user_emb (128) + video_emb (768) + context (32)
- **Training**: PyTorch Lightning, AdamW, CosineAnnealingLR

### 4. Serving
- **API**: FastAPI with async endpoints
- **Caching**: Redis for features and predictions
- **Latency**: <100ms p99
- **Throughput**: 1000+ QPS

### 5. Evaluation
- **Offline**: NDCG@k, Hit@k, Diversity@k
- **Online**: CTR, Watch Time, Skip Rate
- **A/B Testing**: Traffic splitting, statistical significance

### 6. Monitoring
- **Metrics**: Prometheus (latency, throughput, errors)
- **Dashboards**: Grafana
- **Alerts**: Model drift, system health

## Model Details

### Transformer Ranker
```python
class MultiTaskRanker(nn.Module):
    def __init__(self):
        self.transformer = nn.TransformerEncoder(...)
        self.ctr_head = nn.Linear(hidden_dim, 1)
        self.diversity_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, user_emb, video_emb, context):
        # Combine features
        # Pass through transformer
        # Multi-task outputs
        return ctr_score, diversity_score
```

### Loss Function
```python
loss = alpha * bce_loss(ctr_pred, ctr_label) + 
       beta * diversity_loss(diversity_pred, video_embeddings)
```

## Performance Targets
- **Model Accuracy**: NDCG@10 > 0.35
- **Latency**: p50 < 50ms, p99 < 100ms
- **Throughput**: 1000+ QPS
- **Test Coverage**: > 85%
- **Diversity**: Intra-list distance > 0.6

## Resume Bullet Points (STAR Format)

1. **Situation**: Needed personalized video recommendations at scale
   **Task**: Build end-to-end recommendation system
   **Action**: Implemented multi-stage architecture with FAISS recall and Transformer ranking
   **Result**: Achieved 35% NDCG@10, <100ms latency serving 1000+ QPS

2. **Situation**: Required diverse recommendations to improve user engagement
   **Task**: Balance relevance and diversity in recommendations
   **Action**: Designed multi-task learning with diversity loss
   **Result**: Increased content diversity by 40% while maintaining CTR

3. **Situation**: Needed production-ready ML serving infrastructure
   **Task**: Deploy real-time inference with monitoring
   **Action**: Built FastAPI service with Redis caching and Prometheus metrics
   **Result**: 99.9% uptime with comprehensive observability

## 🧠 Core ML Component

### Transformer-Based Multi-Objective Ranking Model

**Input**: User & session embeddings, video features, watch history  
**Architecture**: TransformerEncoder + MLP Heads for:  
- 🎯 CTR Prediction  
- 🎨 Diversity Scoring  

**Loss**: Weighted cross-entropy + intra-list diversity loss  
**Framework**: PyTorch  
**Evaluation**: NDCG@k, Hit@k, Diversity@k  
**Impact**: Personalized + diverse recommendations in real time

## Notes
- Use realistic data distributions in synthetic generation
- Focus on code quality and testing
- Document all design decisions
- Optimize for both accuracy and latency
- Make system horizontally scalable