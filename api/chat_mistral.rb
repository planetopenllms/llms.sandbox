###
#  see https://docs.mistral.ai/getting-started/quickstart/
=begin
curl --location "https://api.mistral.ai/v1/chat/completions" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "Who is the most renowned French painter?"}]
  }'
=end

require 'cocos'

load_env   ## use dotenv (.env)



url = 'https://api.mistral.ai/v1/chat/completions'

headers = {
  ## 'Content-Type'  => 'application/json',
  'Authorization': "Bearer #{ENV['MISTRAL_API_KEY']}",  
}

body = {  model:  "mistral-medium-latest",
          messages: [{ role: 'user',
                       content:'Hello. What is the answer to life, the universe and everything?'}],
       }  
       

=begin
body = {  'model' =>    "mistral-medium-latest",
          'messages' => [{'role'    => 'user',
                          'content' => 'Hello'}],
       }          
=end

pp headers
pp body
pp JSON.pretty_generate( body )



res = Webclient.post( url, headers: headers, json: body )

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

{"id"=>"c1e8c73196df4529b715182a2a5628b1", 
 "object"=>"chat.completion", 
 "created"=>1747323133, 
 "model"=>"mistral-medium-latest", 
 "choices"=>[{"index"=>0, 
  "message"=>{"role"=>"assistant", 
               "tool_calls"=>nil, 
               "content"=>"The answer to life, the universe, and everything is famously **42**, as revealed in Douglas Adams' science fiction series *The Hitchhiker's Guide to the Galaxy*. In the story, a supercomputer named Deep Thought calculates the answer after 7.5 million years of processing, but the characters are left puzzled because they don't know the actual *question* to which 42 is the answer.\n\nOf course, this is a humorous and philosophical take on the idea that some questions may not have simple or meaningful answers. In reality, the \"answer\" is often interpreted as a commentary on the absurdity of seeking definitive meaning in an unpredictable universe.\n\nWould you like a deeper dive into the philosophical implications, or are you just here for the meme? 😄"}, 
               "finish_reason"=>"stop"}], 
               "usage"=>{"prompt_tokens"=>17, "total_tokens"=>176, "completion_tokens"=>159}}

