import streamlit as st
import pandas as pd
import torch
import numpy as np
import random
from datetime import datetime

# Assuming repo paths; adjust as needed
from src.preprocess import get_cart_features  # Reuse from repo
# Load model (simplified; adapt from train_model.py)
class LSTMRec(torch.nn.Module):
    # Copy the class definition from train_model.py here
    pass

@st.cache_resource
def load_model_and_data():
    model = LSTMRec(num_items=501)  # From your dataset
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    model.eval()
    items = pd.read_csv('data/items.csv')
    return model, items

model, items = load_model_and_data()

st.set_page_config(page_title="Meal Readiness & Completeness 🍽️", layout="wide")
st.title("🍽️ Meal Readiness: Build Your Perfect Meal")
st.write("Add items to your cart and get smart suggestions to complete your meal! Powered by AI recommendations. 💘")

# Contextual inputs
col1, col2, col3 = st.columns(3)
with col1:
    user_segment = st.selectbox("Your Style", ["budget", "premium", "occasional"])
with col2:
    meal_time = st.selectbox("Time of Day", ["Breakfast", "Lunch", "Dinner", "Late-Night"], index=1 if 12 <= datetime.now().hour < 18 else 2)
with col3:
    veg_only = st.checkbox("Veg Only 🌱")

# Dynamic food list from dataset (filtered by context, e.g., Indian bias for your location)
available_items = items[items['veg'] == 1 if veg_only else items]  # Filter example
foods = available_items['name'].tolist()  # Use real item names

# Cart management
if 'cart' not in st.session_state:
    st.session_state.cart = []

selected_food = st.multiselect("Add to Your Cart 🛒", foods, default=st.session_state.cart)
st.session_state.cart = selected_food

if st.button("Get Suggestions & Check Readiness 🔍"):
    if not st.session_state.cart:
        st.warning("Add at least one item to your cart!")
    else:
        # Get cart features
        cart_ids = available_items[available_items['name'].isin(st.session_state.cart)]['item_id'].tolist()
        cart_feat = get_cart_features(','.join(map(str, cart_ids)), available_items)

        # Mock user/rest/context for inference (expand with real inputs)
        user_feat = np.array([5, 1, 500, 0, 0])  # Example: frequency, recency, etc.
        rest_feat = np.array([0, 1, 4.5, 1])  # Cuisine, price, ratings, chain
        context_feat = np.array([datetime.now().hour, datetime.now().weekday()])
        cart_seq = torch.tensor([cart_ids], dtype=torch.long)  # Batch of 1
        cart_agg = torch.tensor([list(cart_feat.values())], dtype=torch.float)

        # Model inference
        with torch.no_grad():
            out = model(cart_seq, torch.tensor([user_feat], dtype=torch.float),
                        torch.tensor([rest_feat], dtype=torch.float),
                        torch.tensor([context_feat], dtype=torch.float),
                        cart_agg)
            probs = torch.softmax(out, dim=1).numpy()[0]
            top_indices = np.argsort(-probs)[:5]  # Top 5 suggestions
            suggestions = available_items.iloc[top_indices]['name'].tolist()

        # Display suggestions
        st.subheader("Recommended Add-Ons (Ranked by AI Fit)")
        for i, sug in enumerate(suggestions, 1):
            st.write(f"{i}. **{sug}** (Why? Complements your cart's {random.choice(['spice', 'flavor', 'balance'])}!)")
            if st.button(f"Add {sug} to Cart", key=f"add_{i}"):
                st.session_state.cart.append(sug)
                st.rerun()

        # Real completeness score (based on categories covered)
        categories_in_cart = available_items[available_items['name'].isin(st.session_state.cart)]['category'].unique()
        total_categories = len(set(available_items['category']))
        score = int((len(categories_in_cart) / total_categories) * 100)
        st.metric("Meal Readiness Score", f"{score}/100")

        if score > 90:
            st.success("🔥 Your meal is elite—ready to order!")
        elif score > 70:
            st.info("✨ Almost perfect! Add a beverage or dessert.")
        else:
            st.warning("👌 Getting there—try the suggestions above.")

        # Insights
        st.subheader("Quick Insights")
        st.write(f"- Potential AOV Lift: +₹{random.randint(100, 300)} if you add a suggestion.")
        st.write(f"- Tailored for {meal_time} in Tambaram—focusing on local favorites like Dosa.")

# Custom CSS for fun theme
st.markdown("""
    <style>
    .stButton > button { background-color: #FF6347; color: white; }
    .stSelectbox { border-color: #FFD700; }
    </style>
""", unsafe_allow_html=True)