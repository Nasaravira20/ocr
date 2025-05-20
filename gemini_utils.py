# gemini_utils.py
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("gemini_api_key"))

def extract_text_with_gemini(image: Image.Image) -> str:
    model = genai.GenerativeModel()
    response = model.generate_content(
        ["Extract any visible handwritten or printed text from this historical document:", image]
    )
    return response.text

def translate_with_gemini(text):
    model = genai.GenerativeModel()
    response = model.generate_content([f"Translate this tamil language to English:\n{text}"])
    return response.text


if __name__ == "__main__":
    # Example usage
    image_path = "path_to_your_image.jpg"
    image = Image.open(image_path)
    extracted_text = extract_text_with_gemini(image)
    print(extracted_text)