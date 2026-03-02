import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load data
users = pd.read_csv('data/users.csv')
restaurants = pd.read_csv('data/restaurants.csv')
items = pd.read_csv('data/items.csv')
interactions = pd.read_csv('data/interactions.csv')

# Fit encoders but DON'T transform ID columns (keep originals for lookups)
le_user = LabelEncoder().fit(users['user_id'])
le_item = LabelEncoder().fit(items['item_id'])
le_rest = LabelEncoder().fit(restaurants['restaurant_id'])
le_cat = LabelEncoder()
le_cuis = LabelEncoder()
le_seg = LabelEncoder()
le_price_range = LabelEncoder()

# Encode only the categorical columns
items['category'] = le_cat.fit_transform(items['category'])
users['preferred_cuisine'] = le_cuis.fit_transform(users['preferred_cuisine'])
users['segment'] = le_seg.fit_transform(users['segment'])
restaurants['cuisine'] = le_cuis.transform(restaurants['cuisine'])
restaurants['price_range'] = le_price_range.fit_transform(restaurants['price_range'])

# Create item embeddings (simple one-hot for content-based fallback)
item_features = pd.get_dummies(items[['category', 'veg', 'price']], columns=['category', 'veg'])
item_matrix = item_features.to_numpy()

# Cart features
def get_cart_features(current_cart, items):
    cart_items = [int(x) for x in current_cart.split(',')]
    cart_df = items[items['item_id'].isin(cart_items)]
    agg = {
        'total_price': cart_df['price'].sum(),
        'item_count': len(cart_items),
        'has_main': 1 if 'Main Dish' in cart_df['category'].values else 0,  # Note: category is now encoded, so adjust if needed (or use original strings before encoding)
        'has_side': 1 if 'Side Dish' in cart_df['category'].values else 0,
        # Add more (e.g., veg ratio)
    }
    return agg

# Generate training data
features = []
labels = []
for row in interactions.itertuples():
    user_feat = users[users['user_id'] == row.user_id].iloc[0][['frequency', 'recency', 'monetary', 'preferred_cuisine', 'segment']].values
    rest_feat = restaurants[restaurants['restaurant_id'] == row.restaurant_id].iloc[0][['cuisine', 'price_range', 'ratings', 'is_chain']].values
    context_feat = [row.hour, row.day_of_week]
    cart_feat = get_cart_features(row.current_cart, items)
    cart_seq = [le_item.transform([int(x)])[0] for x in row.current_cart.split(',')]  # Transform here for model input
    features.append({
        'user_feat': user_feat,
        'rest_feat': rest_feat,
        'context_feat': context_feat,
        'cart_feat': list(cart_feat.values()),
        'cart_seq': cart_seq
    })
    labels.append(le_item.transform([row.added_item])[0])

# Save processed data (e.g., np.save for features)
np.save('data/features.npy', features)
np.save('data/labels.npy', labels)
np.save('data/item_matrix.npy', item_matrix)