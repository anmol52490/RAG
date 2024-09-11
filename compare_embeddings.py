from groq import Groq
import os

groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

def main():
    model = 'llama3-70b-8192'

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Get the vector for 'apple'."}
        ]
    )
    vector = response.choices[0].message.content
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    words = ("apple", "iphone")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Compare the similarity between '{words[0]}' and '{words[1]}'."}
        ]
    )
    similarity = response.choices[0].message.content
    print(f"Comparing ({words[0]}, {words[1]}): {similarity}")

if __name__ == "__main__":
    main()
