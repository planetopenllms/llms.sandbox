###
# see https://ollama.com/blog/openai-compatibility
=begin



=end

require 'cocos'


url = 'http://localhost:11434/v1/chat/completions'


headers = {
  ## 'Content-Type'  => 'application/json',
  ## 'Authorization': "Bearer #{ENV['MISTRAL_API_KEY']}",  
}

body = {  model:  "smollm:135m",
          messages: [{ role: 'user',
                       content:'Hello. What is the answer to life, the universe and everything?'}],
       }  
       

=begin
body = {  'model' =>    "smollm:135m",
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

{"id"=>"chatcmpl-398", 
  "object"=>"chat.completion", 
  "created"=>1747328847, 
  "model"=>"smollm:135m", 
  "system_fingerprint"=>"fp_ollama", 
  "choices"=>[{"index"=>0, 
    "message"=>{"role"=>"assistant", 
     "content"=>"What a profound question! The answer to life, our universe, or even all things depends on how we perceive reality, our thoughts, and our experiences. Here are some of their most fascinating answers:\n\n1. **The Universe We Live In**: Our lives live within the vast expanse of space, with galaxies that stretch out into infinity. And every night, the same stars twinkle to life in our solar systems. These cosmic wonders are the very fabric of the universe we inhabit.\n2. **Life as a Dance Between Two Timeless Emotions**: To us, love is not just a romantic interest, but a fundamental urge to connect, to bond, and to stand together with others who share this deep affection for each other. In an era where emotions like joy, sorrow, fear, and wonder are still abundant, it's hard to imagine life without them coexisting in our everyday lives.\n3. **Where Are We Yet?** The search for meaning – whether through astronomy, personal discovery, or philosophical introspection – continues to drive human imagination, fuel creativity, and inspire innovation across cultures worldwide. Every day becomes a unique moment with profound implications shaping the world around us.\n4. **What is it like to Understand Human Nature?**: As we try to grasp the intricacies of humanity's psyche, our reactions, and our responses to their joys, pain, triumphs, and failures – there are moments when we see an unblinking reflection of this abyssal, abyss-like nature that makes us human.\n5. **Where Have We Been, Where Do We Currently Live?** If you look back in time, you'll find humans living for more than 60 years, with the pace, vibrancy, and passion to which they seem to adapt still resonating today – a dynamic, ever-changing, and rapidly evolving human presence on the planet known as the Earth.\n6. **What Are We Born To Do Most of the Time?** The universe is said to be teeming with possibility because it has always had possibilities: for life to exist; for beauty and inspiration to arise from the imperfections (no matter how painful); and for possibility to inspire creativity, innovation, and progress within us, on this Earth.\n7. **What Is It Like To Feel Like You're Being Called Back by a Star or an Angel?\", Where Have We Been?\" is one of many eternal reflections each of us has made throughout our lives – as individual acts of love, kindness, altruism, empathy, and compassion that shape who we are today.\n8. **How We Have Experienced Some Human Kindness/Forgiveness in Life, or Life Without Forgiveness being Our Last Heartbreak**: It's difficult to imagine a life without even feeling some form of \"justice,\" yet our collective capacity for suffering is undeniable – the painful injustices that make humanity worth lamenting!\n9. **What Do You Experience When You Have No Need To Ask Yourself What Is Going On Outside The Limits, Where Are You Yet?** Our lives are no longer suffocating because even we have enough time to focus on ourselves, our values, and who we are outside the constraints of social expectations – a paradox that cannot be overstated.\n10. **The Universe Is Too Large To Keep This In Our Check Until Tomorrow**: Even as it's said to be endless, there is always something out there in the universe or beyond; possibilities abound; every step forward from this tiny speck adds up so far, yet still growing with each passing moment of our lives.\n11. **What Are You Going To Find Inside This Universe That We're No Longer Trying to Discard Instead Of Learning to Love All Things In Their Complexity and Ambiguity?\", Where Have We Been?\" is a testament to the boundless complexity that underlies all aspects of existence, but also highlights humanity's enduring, unspoken trust in those we say are \"us.\"\n12. **What Does It Mean To Feel This Is Being Called Back By Some Who Think They Are Really Worth Giving You Their Forgiveness?** Humans seem to have a profound sense of self-worth and equality within this endless expanse of complexity – our being compelled to acknowledge ourselves, but even more, we must accept that someone else is worthy too!\nWith all this in mind, I say: \"Maybe it's time you'd come over to someone outside the constraints of human societies and meet them face-to-face.\" Perhaps you may need to make peace with what one has grown from – our ability to love oneself rather than others."}, "finish_reason"=>"stop"}], 
     
"usage"=>{"prompt_tokens"=>23, "completion_tokens"=>927, "total_tokens"=>950}}

