import streamlit as st
import random

st.set_page_config(page_title="Meal Completeness Score 🍽️", layout="centered")

st.title("🍽️ Meal Completeness Score")
st.write("Pick a dish and let us find its perfect food soulmate 💘")

# Food list
foods = [
    "Biryani",
    "Pizza",
    "Burger",
    "Pasta",
    "Dosa",
    "Paneer Butter Masala",
    "Sushi",
    "Fried Rice",
    "Tacos",
    "Chocolate Cake"
]

# Complement pairs
complements = {
    "Biryani": ["Lassi", "Raita"],
    "Pizza": ["Garlic Bread", "Cold Coke"],
    "Burger": ["Fries", "Milkshake"],
    "Pasta": ["Garlic Bread", "Iced Tea"],
    "Dosa": ["Filter Coffee", "Coconut Chutney"],
    "Paneer Butter Masala": ["Butter Naan", "Sweet Lassi"],
    "Sushi": ["Miso Soup", "Green Tea"],
    "Fried Rice": ["Manchurian", "Spring Rolls"],
    "Tacos": ["Nachos", "Lemon Soda"],
    "Chocolate Cake": ["Vanilla Ice Cream", "Hot Coffee"]
}

# Cheesy templates (LLM-style 😌)
templates = [
    "A {comp} would complement your {food} like moonlight hugs the ocean.",
    "Your {food} deserves a {comp} the way stars deserve the night sky.",
    "{comp} and {food} together? That’s a love story better than Romeo & Juliet.",
    "A {comp} beside your {food} is as perfect as silver kissing violet.",
    "{food} without {comp}? That’s like a song without melody!"
]

selected_food = st.selectbox("Choose your main dish 👇", foods)

if st.button("Find My Food Soulmate 💖"):
    comp_choice = random.choice(complements[selected_food])
    message_template = random.choice(templates)
    final_message = message_template.format(food=selected_food, comp=comp_choice)

    st.success(final_message)

    # Fake Meal Completeness Score
    score = random.randint(70, 100)
    st.metric("Meal Completeness Score", f"{score}/100")

    if score > 90:
        st.write("🔥 This combo is elite-tier dining.")
    elif score > 80:
        st.write("✨ A well-balanced and satisfying meal!")
    else:
        st.write("👌 Good choice! But you can explore more combinations.")