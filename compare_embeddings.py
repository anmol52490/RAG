from groq import Groq
from groq import Groq
import os

# Load environment variables from a .env file. This is where API keys and other sensitive info are stored.
# load_dotenv()

# Set the Groq API key for authentication
groq_api_key = os.getenv('GROQ_API_KEY')

# Create a Groq client
client = Groq(api_key=groq_api_key)

def main():
    # This function demonstrates how to use embeddings and evaluators.
    # It is primarily for testing and understanding, not for production use.

    # Create an embedding function to convert text into numerical vectors.
    # Choose a model from the supported models list
    model = 'llama3-70b-8192'

    # Example: Get the vector for the word "apple".
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Get the vector for 'apple'."}
        ]
    )
    # Extract the vector from the response
    vector = response.choices[0].message.content
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Example: Compare the similarity between two words using an evaluator.
    # This is just a demo of how to use embeddings for similarity checks.
    words = ("apple", "iphone")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Compare the similarity between '{words[0]}' and '{words[1]}'."}
        ]
    )
    # Extract the similarity measure from the response
    similarity = response.choices[0].message.content
    print(f"Comparing ({words[0]}, {words[1]}): {similarity}")

if __name__ == "__main__":
    # Run the main function if this script is executed directly.
    main()
