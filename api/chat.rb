
require 'cocos'

load_env   ## use dotenv (.env)



url = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'

headers = {
  ## 'Content-Type'  => 'application/json',
  'Authorization': "Bearer #{ENV['GEMINI_API_KEY']}",  
}

body = {  model:  "gemini-2.0-flash",
          messages: [{ role: 'user',
                       content:'Hello. What is the answer to life, the universe and everything?'}],
       }  
       
       
body = {  model:  "gemini-2.0-flash",
          messages: [{ role: 'user',
                       content:'Hello. What is the answer to life, the universe and everything?'},
                      {
                        role: 'assistant',
                        content: "According to the supercomputer Deep Thought in Douglas Adams' *The Hitchhiker's Guide to the Galaxy*, the answer to the ultimate question of life, the universe, and everything is **42**.\n\nHowever, this answer is famously unsatisfying because no one knows what the actual question is!\n"
                      },
                      { role: 'user',
                        content: 'Thanks. What does it mean, any guess?'}
                    ],
       }  

=begin
body = {  'model' =>    "gemini-2.0-flash",
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

{"choices"=>[{"finish_reason"=>"stop", "index"=>0, 
  "message"=>{
    "content"=>"According to the supercomputer Deep Thought in Douglas Adams' *The Hitchhiker's Guide to the Galaxy*, the answer to the ultimate question of life, the universe, and everything is **42**.\n\nHowever, this answer is famously unsatisfying because no one knows what the actual question is!\n",
    "role"=>"assistant"}}],
  "created"=>1747313098, 
  "model"=>"gemini-2.0-flash", 
  "object"=>"chat.completion", 
  "usage"=>{"completion_tokens"=>62, "prompt_tokens"=>14, "total_tokens"=>76}}

  {"choices"=>[{"finish_reason"=>"stop", "index"=>0, 
  "message"=>{"content"=>"According to the supercomputer Deep Thought in Douglas Adams's *The Hitchhiker's Guide to the Galaxy*, the answer to the ultimate question of life, the universe, and everything is **42**.\n\nHowever, Deep Thought was unable to determine what the actual question was! That's the joke.\n", 
              "role"=>"assistant"}}], 
  "created"=>1747313683, 
  "model"=>"gemini-2.0-flash", 
  "object"=>"chat.completion", 
  "usage"=>{"completion_tokens"=>64, "prompt_tokens"=>14, "total_tokens"=>78}}

  {"choices"=>[{"finish_reason"=>"stop", "index"=>0, 
   "message"=>{"content"=>"That's the million-dollar question! Since the question is unknown, any interpretation of \"42\" is essentially just speculation and creative interpretation. Here are some of the more popular and amusing theories:\n\n*   **It's a placeholder:** Some believe it's simply a random number, a humorous commentary on the futility of seeking ultimate answers. Adams himself claimed he chose it because it \"seemed like a small, unassuming number.\"\n*   **Base 13:** Some have suggested that 42 in base 13 is equivalent to 6 x 9 which translates to \"sex times\".\n*   **Binary Code:** Others see significance in its binary representation, linking it to various computer science concepts.\n*   **Towel connection:** Many fans point to the importance of towels in the books, a number of calculations can lead you to 42 from the word \"towel\".\n*   **A Joke About Meaninglessness:** The most likely (and somewhat meta) interpretation is that it's a joke about the human desire for meaning and the inherent meaninglessness of existence. We want a profound answer, but we get something utterly banal.\n*   **The Author's Intention (or Lack Thereof):** Douglas Adams repeatedly stated that he chose the number at random. He said it was just a number that popped into his head and he didn't consciously attach any deep meaning to it.\n\nUltimately, the meaning of \"42\" is whatever you want it to be. That's part of the joke and the charm of *The Hitchhiker's Guide to the Galaxy*. The lack of a definitive answer encourages us to think critically, creatively, and perhaps not take ourselves too seriously in the search for meaning.\n", 
               "role"=>"assistant"}}], 
   "created"=>1747316586, 
   "model"=>"gemini-2.0-flash", 
   "object"=>"chat.completion", 
   "usage"=>{"completion_tokens"=>361, "prompt_tokens"=>86, "total_tokens"=>447}}


