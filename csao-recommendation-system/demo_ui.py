import gradio as gr
from src.inference import infer
import pandas as pd

# Load for dropdowns
users = pd.read_csv('data/users.csv')['user_id'].tolist()
rests = pd.read_csv('data/restaurants.csv')['restaurant_id'].tolist()
items_list = pd.read_csv('data/items.csv')['item_id'].tolist()

def demo_infer(user_id, restaurant_id, current_cart):
    suggestions = infer(user_id, restaurant_id, current_cart)
    return f"Suggested Add-Ons: {', '.join(map(str, suggestions))}"

demo = gr.Interface(
    fn=demo_infer,
    inputs=[
        gr.Dropdown(users, label="User ID"),
        gr.Dropdown(rests, label="Restaurant ID"),
        gr.Textbox(label="Current Cart (comma-separated item IDs, e.g., 1,2)")
    ],
    outputs="text",
    title="CSAO Model Demo",
    description="Enter details to get real-time add-on suggestions."
)

demo.launch()