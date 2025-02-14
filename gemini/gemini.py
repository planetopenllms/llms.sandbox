import google.generativeai as genai
import os

api_key = os.environ['GEMINI_API_KEY']
## print( api_key )

genai.configure(api_key=api_key)


prompt = "what are some options for positional encoding of word embeddings?"

# "generate a multi-head attention layer in pytorch for a neural network?"

# "generate a (vanilla) self-attention layer in pytorch for a neural network." + \
# "plus wrap the single-head self-attention into a multi-head class."

# "generate a self-attention layer in pytorch for a neural network"
# "can you explain the pytorch Embedding layer / class?"
# "what are the pro and cons of byte pair encoding tokenizers compared to alternatives?"
# "timeline of byte pair encoding (BPE) tokenizers in neural networks"
# "how is the gpt-3 neural network architecture different from gpt-2?"
# "what differences are there in the neural network layers of gpt 2 and llama 3?"
# "generate lenet neural net in pytorch"
# "name some belgian lambic beers and breweries"
# 'name all the belgian trappist beers and monasteries'
# "can large language model think?"
# "generate a multi-layer perceptron in ruby code and train on logical xor"
# "generate a perceptron in python code with numpy and train on logical or"
# "timeline of word embeddings / vectors in a.i. research"
# "what's the difference between gpt and bert neural networks?"
# "what are word embeddings?"

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content( prompt )
print(response.text)
print('---')
print(response.usage_metadata)


print('bye')
