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
    response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
                # temperature=0.7,
                stream=True  # stream tokens as they arrive
            )
    
    assistant_message = ""
    for chunk in response:
        ## chunk.choices[0].delta
        ## ChoiceDelta(content='Hello', function_call=None, refusal=None, role='assistant', tool_calls=None)
        token = chunk.choices[0].delta.content
        print( "(debug) chunk-->  ", token )
        assistant_message += token

    return assistant_message.strip()
    

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

