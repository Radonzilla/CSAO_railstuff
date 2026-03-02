import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ====================== LOAD DATA ======================
features = np.load('data/features.npy', allow_pickle=True)
labels = np.load('data/labels.npy')

# Use ALL items from items.csv → this is the correct vocabulary size
items = pd.read_csv('data/items.csv')
num_items = len(items) + 1                    # +1 for padding token (index 0)

# ====================== DATASET ======================
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


# ====================== COLLATE FN (padding) ======================
def collate_fn(batch):
    cart_seqs, users, rests, contexts, cart_aggs, labels = zip(*batch)

    # Pad sequences to same length
    max_len = max(len(seq) for seq in cart_seqs)
    padded_seqs = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) 
                   for seq in cart_seqs]
    padded_seqs = torch.stack(padded_seqs)

    users = torch.stack(users)
    rests = torch.stack(rests)
    contexts = torch.stack(contexts)
    cart_aggs = torch.stack(cart_aggs)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_seqs, users, rests, contexts, cart_aggs, labels


# ====================== MODEL ======================
class LSTMRec(nn.Module):
    def __init__(self, num_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_items, embed_dim, padding_idx=0)   # ← IMPORTANT
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 5 + 4 + 2 + 4, num_items)
        self.num_items = num_items

    def forward(self, cart_seq, user, rest, context, cart_agg):
        embeds = self.embed(cart_seq)                    # now safe
        _, (hn, _) = self.lstm(embeds)
        hn = hn.squeeze(0)
        concat = torch.cat([hn, user, rest, context, cart_agg], dim=1)
        out = self.fc(concat)
        return out


# ====================== TRAINING ======================
model = LSTMRec(num_items)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

train_ds = CartDataset(train_features, train_labels)
test_ds = CartDataset(test_features, test_labels)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

print(f"Training with {len(train_features)} samples | Vocabulary size: {num_items}")

for epoch in range(15):
    model.train()
    total_loss = 0
    for batch in train_loader:
        cart_seq, user, rest, context, cart_agg, label = batch

        optimizer.zero_grad()
        out = model(cart_seq, user, rest, context, cart_agg)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/15 - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'models/lstm_model.pth')
print("Model saved to models/lstm_model.pth ✅")