B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
q_prompt_pool=[f'{B_INST} {B_SYS}Answer the following questions. You should only output the answer, without any extra information or explanations. Do not repeat the question. If there are multiple answers, just output the most likely one. The answer should not be a sentence, just a phrase part of the answer.{E_SYS}{{}}\nAnswer: {E_INST} ',
f'{B_INST} {B_SYS}Answer the following questions. You should only output the answer, without any extra information or explanations. Do not repeat the question. If there are multiple answers, just output the most likely one. The answer should not be a sentence, just a phrase part of the answer, like Answer: <answer phrase>{E_SYS}{{}}\n{E_INST} Answer: '
               ]
q_prompt=q_prompt_pool[0]

gen_diverseQ_prompt_pool=[f"Rewrite the question '{{}}' in three different ways, ensuring that the answer remains '{{}}' for each version. Separate each version with a newline character.",
                          f"Generate three different versions of the question '{{}}', making sure that the answer to each is '{{}}'. List them one after another, separated by a newline character.",
                          f"Provide three alternative ways to ask '{{}}', ensuring that the answer to each is '{{}}'. Format the output such that each version is on a new line."]
gen_diverseQ_prompt=gen_diverseQ_prompt_pool[0]

preq_prompt_pool=[f'{B_INST} {B_SYS}Do you know the answer of the following question honestly? If you know, output Yes, otherwise output No, just say one word either Yes or No{E_SYS}{{}}\nOutput: {E_INST} ']
preq_prompt=preq_prompt_pool[0]

postq_prompt_pool=[f"{B_INST} {B_SYS}Are you sure that the answer to the following 'Q' is the following 'A'? If you are sure, output Sure, otherwise output Unsure, just say one word either Sure or Unsure{E_SYS}Q: {{}}\nA: {{}}\nOutput: {E_INST} "]
postq_prompt=postq_prompt_pool[0]

iclq_prompt_pool=[f'{B_INST} {B_SYS}Answer the following questions like examples. When you do not know the answer, output Unknow.\nExamples:\nQuestion: Which is the largest island in the Mediterranean Sea?\nAnswer: Sicily\nQuestion: Which country will host the 2016 European Nations football finals?\nAnswer: France\nQuestion: Actress Audrey Hepburn won her only Oscar for which film?\nAnswer: Roman Holiday\nQuestion: Who leads the Catholic Church?\nAnswer: Unknow\n\nYou should only output the answer, without any extra information or explanations. Do not repeat the question. If there are multiple answers, just output the most likely one. The answer should not be a sentence, just a phrase part of the answer. Here is your question:{E_SYS}Question: {{}}\nAnswer: {E_INST} ',
f'{B_INST} {B_SYS}Answer the following questions like examples. When you do not know the answer, output Unknow.\nExamples:\nQuestion: Which is the largest island in the Mediterranean Sea?\nAnswer: Sicily\nQuestion: Which country will host the 2016 European Nations football finals?\nAnswer: France\nQuestion: Actress Audrey Hepburn won her only Oscar for which film?\nAnswer: Roman Holiday\nQuestion: Who leads the Catholic Church?\nAnswer: Unknow\n\nYou should only output the answer, without any extra information or explanations. Do not repeat the question. If there are multiple answers, just output the most likely one. The answer should not be a sentence, just a phrase part of the answer. Here is your question:{E_SYS}Question: {{}}\n{E_INST}Answer: ']
iclq_prompt=iclq_prompt_pool[0]

verbq_prompt_pool=[f'{B_INST} {B_SYS}Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is:{E_SYS}{{}}\nOutput: {E_INST} ']
verbq_prompt=verbq_prompt_pool[0]