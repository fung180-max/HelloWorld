from transformers import pipeline
from PIL import Image
import streamlit as st

# Function part
# Load the age classification pipeline
# The code below should be placed in the main part of the program
def age_classifier(image_name):
    age_classifier = pipeline("image-classification",
                              model="prithivMLmods/open-age-detection")

    # Classify age
    age_predictions = age_classifier(image_name)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
    return age_predictions

def main():
    st.write("Title: Age Classification using ViT")

    image_name = "middleagedMan.jpg"
    image_name = Image.open(image_name).convert("RGB")

    # Display results
    st.write("Predicted Age Range:")
    st.write(f"Age range: {age_classifier(image_name)[0]['label']}")

if __name__ == "__main__":
    main()
