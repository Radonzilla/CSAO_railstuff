import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score, precision_score, recall_score

# Load processed data
features = np.load('data/features.npy', allow_pickle=True)
labels = np.load('data/labels.npy')

# Load items to get FULL vocabulary size
items = pd.read_csv('data/items.csv')
num_items = len(items) + 1  # +1 for padding

# Dataset
class CartDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        f = self.features[idx]
        cart_seq = torch.tensor(f['cart_seq'], dtype=torch.long)
        user = torch.tensor(f['user_feat'], dtype=torch.float)
        rest = torch.tensor(f['rest_feat'], dtype=torch.float)
        context = torch.tensor(f['context_feat'], dtype=torch.float)
        cart_agg = torch.tensor(f['cart_feat'], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cart_seq, user, rest, context, cart_agg, label

# Custom collate to pad variable-length cart_seq
def collate_fn(batch):
    cart_seqs, users, rests, contexts, cart_aggs, labels = zip(*batch)
    
    max_len = max(len(seq) for seq in cart_seqs)
    padded_seqs = [torch.cat([seq, torch.full((max_len - len(seq),), num_items-1, dtype=torch.long)]) 
                   for seq in cart_seqs]
    padded_seqs = torch.stack(padded_seqs)
    
    users = torch.stack(users)
    rests = torch.stack(rests)
    contexts = torch.stack(contexts)
    cart_aggs = torch.stack(cart_aggs)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_seqs, users, rests, contexts, cart_aggs, labels

# Re-create test split (match training's random_state)
_, test_features, _, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

test_ds = CartDataset(test_features, test_labels)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)  # Small batch for tiny data

# Model class
class LSTMRec(nn.Module):
    def __init__(self, num_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_items, embed_dim, padding_idx=num_items-1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 5 + 4 + 2 + 4, num_items)  # Adjust dims as in training
        self.num_items = num_items

    def forward(self, cart_seq, user, rest, context, cart_agg):
        embeds = self.embed(cart_seq)
        _, (hn, _) = self.lstm(embeds)
        hn = hn.squeeze(0)
        concat = torch.cat([hn, user, rest, context, cart_agg], dim=1)
        out = self.fc(concat)
        return out

# Load model
model = LSTMRec(num_items)
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

# Collect predictions
preds = []
true = []
with torch.no_grad():
    for batch in test_loader:
        cart_seq, user, rest, context, cart_agg, label = batch
        out = model(cart_seq, user, rest, context, cart_agg)
        probs = torch.softmax(out, dim=1).detach().numpy()
        preds.extend(probs)
        true.extend(label.numpy())

# ====================== METRICS ======================
print("\n=== Evaluation Results ===")

# AUC
unique_true = len(np.unique(true))
if unique_true > 1 and unique_true == num_items:
    auc = roc_auc_score(true, preds, multi_class='ovr')
    print(f"AUC                  : {auc:.4f}")
else:
    print(f"AUC                  : N/A (only {unique_true} classes in test set out of {num_items} possible)")

# Precision@5 & Recall@5
K = 5
pred_topk = [np.argsort(-p)[:K] for p in preds]
precision_scores = [1/K if true_label in topk else 0 for true_label, topk in zip(true, pred_topk)]
recall_scores     = [1.0 if true_label in topk else 0 for true_label, topk in zip(true, pred_topk)]

print(f"Precision@{K}          : {np.mean(precision_scores):.4f}")
print(f"Recall@{K}             : {np.mean(recall_scores):.4f}")

# NDCG@5 (fixed & clean version)
ndcg_scores = []
for i in range(len(true)):
    y_true = np.zeros(num_items, dtype=float)
    y_true[true[i]] = 1.0
    ndcg = ndcg_score([y_true], [preds[i]], k=K)
    ndcg_scores.append(ndcg)

print(f"NDCG@{K}               : {np.mean(ndcg_scores):.4f}")

print("Evaluation completed successfully!")
