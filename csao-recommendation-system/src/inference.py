import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
from src.preprocess import get_cart_features  # Assume preprocess.py is in src/

# Load data for encoding and features
users = pd.read_csv('data/users.csv')
restaurants = pd.read_csv('data/restaurants.csv')
items = pd.read_csv('data/items.csv')

# Encoders (must match training)
le_user = LabelEncoder().fit(users['user_id'])
le_item = LabelEncoder().fit(items['item_id'])
le_rest = LabelEncoder().fit(restaurants['restaurant_id'])
le_cat = LabelEncoder().fit(items['category'])
le_cuis = LabelEncoder().fit(users['preferred_cuisine'].append(restaurants['cuisine']))
le_seg = LabelEncoder().fit(users['segment'])
le_price_range = LabelEncoder().fit(restaurants['price_range'])

# Model class (copy from train_model.py)
class LSTMRec(torch.nn.Module):
    def __init__(self, num_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = torch.nn.Embedding(num_items, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim + 5 + 4 + 2 + 4, num_items)  # Adjust based on feature dims
        self.num_items = num_items

    def forward(self, cart_seq, user, rest, context, cart_agg):
        embeds = self.embed(cart_seq)
        _, (hn, _) = self.lstm(embeds)
        hn = hn.squeeze(0)
        concat = torch.cat([hn, user, rest, context, cart_agg], dim=1)
        out = self.fc(concat)
        return out

# Load model
num_items = len(le_item.classes_) + 1  # +1 for padding
model = LSTMRec(num_items)
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

def infer(user_id, restaurant_id, current_cart_str, hour=None, day_of_week=None):
    # Default context if not provided
    if hour is None:
        hour = np.random.randint(0, 23)
    if day_of_week is None:
        day_of_week = np.random.randint(0, 6)

    # Features
    user_feat = users[users['user_id'] == user_id][['frequency', 'recency', 'monetary', 'preferred_cuisine', 'segment']].values[0]
    user_feat[3] = le_cuis.transform([user_feat[3]])[0]  # Encode
    user_feat[4] = le_seg.transform([user_feat[4]])[0]

    rest_feat = restaurants[restaurants['restaurant_id'] == restaurant_id][['cuisine', 'price_range', 'ratings', 'is_chain']].values[0]
    rest_feat[0] = le_cuis.transform([rest_feat[0]])[0]
    rest_feat[1] = le_price_range.transform([rest_feat[1]])[0]

    context_feat = np.array([hour, day_of_week])

    cart_ids = [int(x) for x in current_cart_str.split(',')]
    cart_feat = get_cart_features(current_cart_str, items)
    cart_seq = torch.tensor([le_item.transform(cart_ids)], dtype=torch.long)

    # To tensors
    user_t = torch.tensor(user_feat, dtype=torch.float).unsqueeze(0)
    rest_t = torch.tensor(rest_feat, dtype=torch.float).unsqueeze(0)
    context_t = torch.tensor(context_feat, dtype=torch.float).unsqueeze(0)
    cart_agg_t = torch.tensor(list(cart_feat.values()), dtype=torch.float).unsqueeze(0)

    with torch.no_grad():
        out = model(cart_seq, user_t, rest_t, context_t, cart_agg_t)
        probs = torch.softmax(out, dim=1)[0].numpy()

    top_indices = np.argsort(-probs)[:5]  # Top 5
    suggestions = [le_item.inverse_transform([idx])[0] for idx in top_indices if idx < len(le_item.classes_)]

    return suggestions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_id', type=int, required=True)
    parser.add_argument('--restaurant_id', type=int, required=True)
    parser.add_argument('--current_cart', type=str, required=True)  # e.g., "1,2"
    args = parser.parse_args()

    suggestions = infer(args.user_id, args.restaurant_id, args.current_cart)
    print(f"Suggested add-ons: {suggestions}")