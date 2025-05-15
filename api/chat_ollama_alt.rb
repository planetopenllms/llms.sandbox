###
# see https://ollama.com/blog/openai-compatibility
=begin

$ ollama list
NAME            ID              SIZE     
qwen2.5:0.5b    a8b0c5157701    397 MB   
smollm2:135m    9077fe9d2ae1    270 MB   
smollm:135m     b0b2a4617438    91 MB    
phi:latest      e2fd6321a5fe    1.6 GB   
=end

require 'cocos'


url = 'http://localhost:11434/v1/chat/completions'


headers = {
  ## 'Content-Type'  => 'application/json',
  ## 'Authorization': "Bearer #{ENV['MISTRAL_API_KEY']}",  
}

body = {  model:  "qwen2.5:0.5b",
          messages: [{ role: 'user',
                       content:'Hello. What is the answer to life, the universe and everything?'}],
       }  
       

=begin
body = {  'model' =>    "qwen2.5:0.5b",
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

{"id"=>"chatcmpl-917", 
 "object"=>"chat.completion", "created"=>1747329139, 
  "model"=>"qwen2.5:0.5b", "system_fingerprint"=>"fp_ollama", 
  "choices"=>[{"index"=>0, 
  "message"=>{"role"=>"assistant", 
    "content"=>"The answer to the questions \"How long have you been around?\", \"Where do you come from?\" and \"What is your purpose\" is eternity or non-being. However, I can offer a somewhat philosophical perspective on eternal being as infinite time exists - \"Life, the universe, and everything\". This means that what matters most in existence is existence itself. This statement suggests we should not focus on external causes but place greater emphasis on internal qualities like wisdom and virtue to understand how we are formed and function."},
   "finish_reason"=>"stop"}], "usage"=>{"prompt_tokens"=>43, "completion_tokens"=>102, "total_tokens"=>145}}

