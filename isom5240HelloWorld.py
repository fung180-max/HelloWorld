from transformers import pipeline
from PIL import Image
import streamlit as st

# Load the age classification pipeline
# The code below should be placed in the main part of the program
def age_classifier():
    age_classifier = pipeline("image-classification",
                              model="akashmaggon/vit-base-age-classification")

    image_name = "middleagedMan.jpg"
    image_name = Image.open(image_name).convert("RGB")

    # Classify age
    age_predictions = age_classifier(image_name)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
    return age_predictions

def main():
    st.write("Title: Age Classification using ViT")

    # Display results
    # print("Predicted Age Range:")
    # print(f"Age range: {age_predictions[0]['label']}")
    st.write("Predicted Age Range:")
    st.write(f"Age range: {age_classifier()[0]['label']}")

if __name__ == "__main__":
    main()
