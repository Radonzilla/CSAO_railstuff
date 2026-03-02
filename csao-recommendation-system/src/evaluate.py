import torch
from sklearn.metrics import roc_auc_score, ndcg_score, precision_score, recall_score
# adjust path accordingly
from src.train_model import LSTMRec   # adjust path accordingly
# Load model, test_loader from above
model = LSTMRec(num_items)
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

preds = []
true = []
for batch in test_loader:
    cart_seq, user, rest, context, cart_agg, label = batch
    out = model(cart_seq, user, rest, context, cart_agg)
    preds.extend(torch.softmax(out, dim=1).detach().numpy())
    true.extend(label.numpy())

# For ranking, assume we rank all items and evaluate top-K
# In practice, rank candidates only
auc = roc_auc_score(true, preds, multi_class='ovr')
ndcg = ndcg_score([true], [np.argsort(-p)[:, :10] for p in preds])  # Simplified
print(f'AUC: {auc}, NDCG@10: {ndcg}')
# Add Precision@K, Recall@K similarly