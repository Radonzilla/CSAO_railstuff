# System Architecture and Design Decisions

## Overview
The CSAO Recommendation System is designed as a real-time, sequential recommendation engine using an LSTM-based model to predict add-on items based on cart state, user behavior, restaurant details, and context. It handles cold starts via content-based similarity and ensures low latency through pre-computed embeddings.

## Components
- **Data Pipeline**: ETL with Pandas for feature engineering; Airflow for scheduling refreshes.
- **Model**: LSTM for sequences + FC layer for fused features. Input: Cart seq embeddings + user/rest/context/cart agg. Output: Item probabilities.
- **Inference Service**: TorchServe or FastAPI for <200ms predictions. Caching with Redis for frequent queries.
- **Fallback**: Cosine similarity on item features for new users/items.

## Design Decisions
- **Sequential Modeling**: LSTM chosen over Transformer for efficiency on variable cart lengths (common in food delivery).
- **Scalability**: Horizontal scaling with Kubernetes; batch inference for peaks.
- **Latency Trade-offs**: Reduced hidden_dim to 64 for speed; GPU acceleration.
- **Limitations**: Assumes fixed menu; extend with online learning for evolving prefs.

## Diagram
[Insert ASCII or Mermaid diagram here, e.g.]
graph TD
    A[User Cart] --> B[Feature Extractor]
    B --> C[LSTM Model]
    C --> D[Ranker & Diversity Filter]
    D --> E[Recommendations]