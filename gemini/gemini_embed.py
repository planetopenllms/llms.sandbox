## https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb

import google.generativeai as genai
import os

api_key = os.environ['GEMINI_API_KEY']
## print( api_key )

genai.configure(api_key=api_key)

text = "Hello world"
result = genai.embed_content(model="models/text-embedding-004",
                             content=text)

# Print just a part of the embedding to keep the output manageable
print(str(result['embedding'])[:50], '... TRIMMED]')

print('---')
print(len(result['embedding'])) # The embeddings have 768 dimensions

print('bye')


"""
[0.013168517, -0.00871193, -0.046782672, 0.0006996 ... TRIMMED]
---
768
"""
