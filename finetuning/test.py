import vertexai
from vertexai.generative_models import GenerativeModel

def generate_images(prompt: str, project_id: str, location: str) -> list:
    """
    Generates images based on the provided text prompt using the Vertex AI API.

    Args:
        prompt: The text prompt to guide the image generation.
        project_id: Your Google Cloud Project ID.
        location: The region you are using (e.g., "us-central1").

    Returns:
        A list of generated image data (bytes).
    """
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Load the image generation model (adjust model name if needed)
    model = GenerativeModel("imagegeneration")  # Or "imagegeneration@001"

   

    #If you want to use the examples, pass them in here.
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,  # Generate one image
        #examples=examples, #Uncomment this line if you want to use the examples defined above
    )

    generated_images_data = []
    if images.count > 0:
        img = images[0]
        generated_images_data = [img.data] # returns a list containing the image data

    return generated_images_data

def main():
    """
    Main function to set parameters and call the image generation function.
    """
    # Replace with your actual Google Cloud Project ID and region.
    project_id = "lawcraftsman"  # <--- CHANGE THIS
    location = "asia-east1"       # <--- CHANGE THIS

    # The text prompt to use for image generation.
    prompt = "A single page of old Tamil literature script"

    # Call the image generation function.
    generated_images = generate_images(prompt, project_id, location)

    if generated_images:
        #  For demonstration, let's save the first generated image to a file.
        #  In a real application, you would likely send this data to a web app,
        #  store it in a database, or process it further.
        image_data = generated_images[0]
        with open("generated_image.jpg", "wb") as f:
            f.write(image_data)
        print("Image saved to generated_image.jpg")
        print("Image generation successful!")
    else:
        print("Image generation failed.")



if __name__ == "__main__":
    main()
    
