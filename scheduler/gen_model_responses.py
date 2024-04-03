from datasets import load_dataset
import pandas as pd
import math
import numpy as np
import time
from vllm import LLM, SamplingParams


LOAD_FROM_CSV = True
SAVE_RESPONSE_TO_CSV = True

model_to_serve = 'facebook/opt-350m'
model_to_serve_short = 'opt_350m'

if LOAD_FROM_CSV:
    # load the llm query dataset from csv
    df_prompts = pd.read_csv('sampled_prompts.csv')
    print(df_prompts.head())
else:
    # load the llm query dataset from lmsys dataset
    dataset_name = 'lmsys/lmsys-chat-1m'
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.select(range(10000))

    df_prompts = pd.DataFrame(columns=['model', 'content', 'conversation_id', 'round_id'])
    data = []
    conversation_id = 0
    for example in dataset:
        conversation = example['conversation']
        for j, sentence in enumerate(conversation):
            # print(sentence['role'], sentence['content'])
            if sentence['role'] == 'user':  # assistant
                data.append({'model': example['model'], 'content': sentence['content'], 'conversation_id': conversation_id, 'round_id': 0})
            break  # only considering 1st-round
        conversation_id += 1

    df_prompts = pd.concat([df_prompts, pd.DataFrame(data)], ignore_index=True)
    print(df_prompts.head())

    df_prompts.to_csv('sampled_prompts.csv', index=False)

print('# of prompts for vicuna:', len(df_prompts[df_prompts['model']=='vicuna-13b']))
# print(df_prompts.groupby('model')['conversation_id'].describe()['count'])
# print('Output distribution:', df_prompts[df_prompts['model']=='vicuna-13b']['response_length'].describe())

data_src_model = 'vicuna-13b'
prompts = df_prompts[df_prompts['model']==data_src_model]['content'].to_list()

sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
llm = LLM(model=model_to_serve)

start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()
execution_time_ms = int((end_time - start_time))
print('Completion time:', execution_time_ms, 's')

if SAVE_RESPONSE_TO_CSV:
    df_responses = pd.DataFrame(columns=['model', 'prompt', 'response', 'response_length', 'conversation_id', 'round_id'])
    data = []
    conversation_id = 0
    for output in outputs:
        prompt = output.prompt
        generated_text = ''.join([output.outputs[i].text for i in range(len(output.outputs))])
        if len(output.outputs[0].token_ids) <= 1 or len(output.outputs[0].token_ids) >= 1024:
            continue
        data.append({'model': model_to_serve, 'prompt': prompt, 'response': generated_text, 'response_length': len(output.outputs[0].token_ids), 'conversation_id': conversation_id, 'round_id': 0})
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        conversation_id += 1

    df_responses = pd.concat([df_responses, pd.DataFrame(data)], ignore_index=True)
    print(df_responses.head())

    df_responses.to_csv('sampled_prompts_responses_'+model_to_serve_short+'.csv', index=False)
