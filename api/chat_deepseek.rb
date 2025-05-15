###
#  see https://api-docs.deepseek.com/
#      note - no free tier (payment required for use!)


require 'cocos'

load_env   ## use dotenv (.env)



url = 'https://api.deepseek.com/chat/completions'

headers = {
  ## 'Content-Type'  => 'application/json',
  'Authorization': "Bearer #{ENV['DEEPSEEK_API_KEY']}",  
}

body = {  model:  "deepseek-chat",
          messages: [{ role: 'user',
                       content:'Hello. What is the answer to life, the universe and everything?'}],
       }  
       

=begin
body = {  'model' =>    "deepseek-chat",
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

POST https://api.deepseek.com/chat/completions...
402 Payment Required

{"error"=>{"message"=>"Insufficient Balance", 
           "type"=>"unknown_error", "param"=>nil, "code"=>"invalid_request_error"}}
