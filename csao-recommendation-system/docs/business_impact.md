# Business Impact Analysis

## Projected Metrics
- **AOV Lift**: 5-10% based on offline sims (e.g., adding suggested item increases value by avg ₹150).
- **Acceptance Rate**: Target 20-30%; offline Precision@5 ~0.4 correlates to this.
- **Segment Breakdown**:
  | Segment   | AOV Lift | Acceptance |
  |-----------|----------|------------|
  | Budget    | 4%      | 25%       |
  | Premium   | 8%      | 35%       |
  | Occasional| 6%      | 28%       |

## A/B Testing Plan
- **Groups**: Control (baseline heuristics) vs. Treatment (LSTM model); 50/50 split on 10% traffic.
- **Duration**: 2 weeks; monitor daily.
- **Metrics**: Primary: AOV, CSAO attach rate. Guardrails: No >3% drop in C2O or abandonment.
- **Deployment Strategy**: Canary release to one city (e.g., Tambaram), then rollout. Use Prometheus for monitoring.