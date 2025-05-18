from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # take environment variables

import os
import sys
import time
from openai import OpenAI


client = OpenAI( api_key  = os.getenv("GEMINI_API_KEY"),
                 base_url = os.getenv("GEMINI_API_BASE") )


## print( client )


def get_response( messages ):
    start_time = time.time()

    completion = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
                # temperature=0.7,
                # stream=True  # stream tokens as they arrive
            )
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    total_tokens = completion.usage.total_tokens
    print(f"\n[Response time: {elapsed_time:.2f}s | Tokens used: {total_tokens}]")

    return completion.choices[0].message.content.strip()
    

print("Welcome to CLI Chat! Type your message and press Enter. Type 'exit' to quit.")
conversation = []  # history of messages for context




while True:
    try:
        prompt = input("You: ")
    except(KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break

    if prompt.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Append user message to conversation history
    conversation.append({"role": "user", "content": prompt})
    response = get_response( conversation )

    print("AI:", end=" ", flush=True)
    print( response )
    print()

    conversation.append({"role": "assistant", "content": response})

