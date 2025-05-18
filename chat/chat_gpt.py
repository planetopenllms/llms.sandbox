"""
hello. can you write a command-line python chat app script backed with openai api?

thanks. can you add a base_url option for the openai client and use the load_dev 
machinery to get the api key via .env?
"""



import os
import sys
from openai import OpenAI
from dotenv import load_dotenv


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load OpenAI API key and base URL from environment
    api_key = os.getenv("GEMINI_API_KEY")
    api_base = os.getenv("GEMINI_API_BASE")

    if not api_key:
        print("Error: Please set the OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    client = OpenAI( api_key=api_key, base_url=api_base )

    print("Welcome to CLI Chat! Type your message and press Enter. Type 'exit' to quit.")
    conversation = []  # history of messages for context

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Append user message to conversation history
        conversation.append({"role": "user", "content": user_input})

        try:
            # Query the OpenAI ChatCompletion API
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=conversation,
                # temperature=0.7,
                stream=True  # stream tokens as they arrive
            )

            print("Assistant:", end=" ", flush=True)
            assistant_message = ""

            # Stream and display each token
            for chunk in response:
                ## if 'choices' in chunk:
                ##    token = chunk['choices'][0]['delta'].get('content', '')
                token = chunk.choices[0].delta.content
                print( token, end="", flush=True)
                assistant_message += token

            print()  # newline after complete response
            # Add assistant response to conversation history
            conversation.append({"role": "assistant", "content": assistant_message})

        except Exception as e:
            print(f"Error communicating with OpenAI: {e}")

if __name__ == "__main__":
    main()
