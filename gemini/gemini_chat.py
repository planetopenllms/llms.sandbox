import google.generativeai as genai
import os

api_key = os.environ['GEMINI_API_KEY']

genai.configure(api_key=api_key)


model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    ]
)
response = chat.send_message("I have 2 dogs in my house.")
print(response.text)
#=> That's wonderful!  Two dogs make for a lot of love and fun.
#    What breeds are they? 😊

response = chat.send_message("How many paws are in my house?")
print(response.text)
#=> You have two dogs, and each dog has four paws.
#   So that means there are a total of 8 paws in your house! 😄🐾

print( chat.history )

print( "bye")
