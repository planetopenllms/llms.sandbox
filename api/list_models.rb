###
# query for models - via open ai-compatible api
#          /list.models
#     https://platform.openai.com/docs/api-reference/models/list
#

require 'cocos'

load_env   ## use dotenv (.env)



url = 'https://generativelanguage.googleapis.com/v1beta/openai/models'

headers = {
  ## 'Content-Type'  => 'application/json',
  'Authorization': "Bearer #{ENV['GEMINI_API_KEY']}",  
}

pp headers

res = Webclient.get( url, headers: headers )

puts res.status.code       #=> 200
puts res.status.message    #=> OK
puts res.status.ok?

puts
puts "text:"
puts res.text
puts
puts "json:"
puts res.json


puts "bye"

__END__


{
  "object": "list",
  "data": [
    {
      "id": "models/chat-bison-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/text-bison-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/embedding-gecko-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.0-pro-vision-latest",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-pro-vision",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-pro-latest",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-pro-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-pro-002",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-pro",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-latest",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-001-tuning",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-002",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-8b",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-8b-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-8b-latest",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-8b-exp-0827",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-1.5-flash-8b-exp-0924",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.5-pro-exp-03-25",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.5-pro-preview-03-25",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.5-flash-preview-04-17",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.5-flash-preview-04-17-thinking",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.5-pro-preview-05-06",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-exp",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-lite-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-lite",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-lite-preview-02-05",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-lite-preview",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-pro-exp",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-pro-exp-02-05",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-exp-1206",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-thinking-exp-01-21",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-thinking-exp",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-thinking-exp-1219",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/learnlm-2.0-flash-experimental",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemma-3-1b-it",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemma-3-4b-it",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemma-3-12b-it",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemma-3-27b-it",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/embedding-001",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/text-embedding-004",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-embedding-exp-03-07",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-embedding-exp",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/aqa",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/imagen-3.0-generate-002",
      "object": "model",
      "owned_by": "google"
    },
    {
      "id": "models/gemini-2.0-flash-live-001",
      "object": "model",
      "owned_by": "google"
    }
  ]
}
