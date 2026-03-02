import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

num_users = 200
num_restaurants = 50
num_items = 500
num_orders = 1000

categories = ['Main Dish', 'Side Dish', 'Beverage', 'Dessert', 'Appetizer']
cuisines = ['Indian', 'Chinese', 'Italian', 'Mexican', 'American', 'Continental']

users = pd.DataFrame({
    'user_id': range(1, num_users + 1),
    'frequency': np.random.poisson(5, num_users) + 1,
    'recency': [random.randint(1, 30) for _ in range(num_users)],
    'monetary': np.random.uniform(100, 1000, num_users),
    'preferred_cuisine': [random.choice(cuisines) for _ in range(num_users)],
    'segment': [random.choice(['budget', 'premium', 'occasional']) for _ in range(num_users)]
})

restaurants = pd.DataFrame({
    'restaurant_id': range(1, num_restaurants + 1),
    'cuisine': [random.choice(cuisines) for _ in range(num_restaurants)],
    'price_range': [random.choice(['low', 'medium', 'high']) for _ in range(num_restaurants)],
    'ratings': np.random.uniform(3.0, 5.0, num_restaurants),
    'is_chain': [random.choice([0, 1]) for _ in range(num_restaurants)]
})

items_per_rest = num_items // num_restaurants
extra = num_items % num_restaurants
restaurant_ids = []
for r in range(1, num_restaurants + 1):
    restaurant_ids.extend([r] * items_per_rest)
for r in range(1, extra + 1):
    restaurant_ids.append(r)
random.shuffle(restaurant_ids)

items = pd.DataFrame({
    'item_id': range(1, num_items + 1),
    'name': [f'Item_{i}' for i in range(1, num_items + 1)],
    'category': [random.choice(categories) for _ in range(num_items)],
    'veg': [random.choice([0, 1]) for _ in range(num_items)],
    'price': np.random.uniform(50, 500, num_items),
    'restaurant_id': restaurant_ids
})

category_flow = {
    'Main Dish': ['Side Dish', 'Beverage', 'Dessert'],
    'Side Dish': ['Beverage', 'Dessert'],
    'Beverage': ['Dessert'],
    'Appetizer': ['Main Dish', 'Beverage'],
    'Dessert': []
}

def simulate_sequential_cart(restaurant_id):
    rest_items = items[items['restaurant_id'] == restaurant_id]
    if len(rest_items) < 2:
        return []
    available_cats = rest_items['category'].unique()
    if len(available_cats) < 2:
        return []
    start_cat = random.choice(available_cats)
    cart = [rest_items[rest_items['category'] == start_cat].sample(1)['item_id'].iloc[0]]
    current_cat = start_cat
    max_length = random.randint(2, 5)
    while len(cart) < max_length:
        possible_next = category_flow.get(current_cat, list(available_cats))
        next_cats = [cat for cat in possible_next if cat in available_cats]
        if not next_cats:
            next_cats = [cat for cat in available_cats if cat != current_cat]
            if not next_cats:
                break
        next_cat = random.choice(next_cats)
        candidates = rest_items[(rest_items['category'] == next_cat) & ~rest_items['item_id'].isin(cart)]
        if candidates.empty:
            break
        added = candidates.sample(1)['item_id'].iloc[0]
        cart.append(added)
        current_cat = next_cat
    return cart

orders = []
for _ in range(num_orders):
    user_id = random.randint(1, num_users)
    restaurant_id = random.randint(1, num_restaurants)
    cart_items = simulate_sequential_cart(restaurant_id)
    if len(cart_items) < 2:
        continue
    timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
    hour = random.randint(0, 23)
    day_of_week = timestamp.weekday()
    total_value = items[items['item_id'].isin(cart_items)]['price'].sum()
    orders.append({
        'order_id': len(orders) + 1,
        'user_id': user_id,
        'restaurant_id': restaurant_id,
        'timestamp': timestamp,
        'hour': hour,
        'day_of_week': day_of_week,
        'items': cart_items,
        'total_value': total_value
    })

orders_df = pd.DataFrame(orders)

interactions = []
for order in orders:
    cart = order['items']
    for i in range(1, len(cart)):
        current_cart = cart[:i]
        added = cart[i]
        interactions.append({
            'user_id': order['user_id'],
            'restaurant_id': order['restaurant_id'],
            'timestamp': order['timestamp'],
            'hour': order['hour'],
            'day_of_week': order['day_of_week'],
            'current_cart': ','.join(map(str, current_cart)),
            'added_item': added
        })

interactions_df = pd.DataFrame(interactions)

# Save
users.to_csv('data/users.csv', index=False)
restaurants.to_csv('data/restaurants.csv', index=False)
items.to_csv('data/items.csv', index=False)
orders_df.to_csv('data/orders.csv', index=False)
interactions_df.to_csv('data/interactions.csv', index=False)