#
# see https://ai.google.dev/gemini-api/docs/models/gemini

"""
gemini-1.5-flash
Free:
- 15 RPM
- 1 million TPM
- 1,500 RPD

RPM: Requests per minute
TPM: Tokens per minute
RPD: Requests per day
TPD: Tokens per day
"""



import google.generativeai as genai
import os

api_key = os.environ['GEMINI_API_KEY']
## print( api_key )

genai.configure(api_key=api_key)


model_info = genai.get_model("models/gemini-1.5-flash")
print( model_info )
print( type(model_info) )


"""
Model(name='models/gemini-1.5-flash',
      base_model_id='',
      version='001',
      display_name='Gemini 1.5 Flash',
      description='Fast and versatile multimodal model for scaling across diverse tasks',
      input_token_limit=1000000,
      output_token_limit=8192,
      supported_generation_methods=['generateContent', 'countTokens'],
      temperature=1.0,
      max_temperature=2.0,
      top_p=0.95,
      top_k=40)
<class 'google.generativeai.types.model_types.Model'>
"""

model_info = genai.get_model("models/gemini-1.5-flash-8b")
print( model_info )
print( type(model_info) )

"""
Model(name='models/gemini-1.5-flash-8b',
      base_model_id='',
      version='001',
      display_name='Gemini 1.5 Flash-8B',
      description='Fast and versatile multimodal model for scaling across diverse tasks',
      input_token_limit=1000000,
      output_token_limit=8192,
      supported_generation_methods=['createCachedContent', 'generateContent', 'countTokens'],
      temperature=1.0,
      max_temperature=2.0,
      top_p=0.95,
      top_k=40)
"""

model_info = genai.get_model("models/text-embedding-004")
print( model_info )
print( type(model_info) )

"""
Model(name='models/text-embedding-004',
      base_model_id='',
      version='004',
      display_name='Text Embedding 004',
      description='Obtain a distributed representation of a text.',
      input_token_limit=2048,
      output_token_limit=1,
      supported_generation_methods=['embedContent'],
      temperature=None,
      max_temperature=None,
      top_p=None,
      top_k=None)
<class 'google.generativeai.types.model_types.Model'>
"""


model_info = genai.get_model("models/aqa")
print( model_info )
print( type(model_info) )

"""
Model(name='models/aqa',
      base_model_id='',
      version='001',
      display_name='Model that performs Attributed Question Answering.',
      description=('Model trained to return answers to questions that are grounded in provided '
                   'sources, along with estimating answerable probability.'),
      input_token_limit=7168,
      output_token_limit=1024,
      supported_generation_methods=['generateAnswer'],
      temperature=0.2,
      max_temperature=None,
      top_p=1.0,
      top_k=40)
"""


print('---')
print('generateContent:')
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

print('embedContent:')
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)
print('all:')
models = genai.list_models()
for m in models:
    print(m.name)



print( 'bye' )

