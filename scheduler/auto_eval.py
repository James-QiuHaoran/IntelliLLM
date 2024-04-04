import pandas as pd
import math
import numpy as np
import time
from vllm import LLM, SamplingParams


def eval(df_prompts, llm, max_batch_size, methods=['fcfs', 'sjf']):
    num_exp = 20
    num_jobs_per_exp = 50
    avg_jct_exp_list = {}
    throughput_exp_list = {}

    for method in methods:
        avg_jct_exp_list[method] = []
        throughput_exp_list[method] = []

    for exp_id in range(num_exp):
        print('>>>> Exp #' + str(exp_id))
        # randomly sample jobs from the dataset
        sampled_rows = df_prompts.sample(n=num_jobs_per_exp)
        prompts = sampled_rows['prompt'].tolist()
        output_lengths = sampled_rows['response_length'].tolist()

        for method in methods:
            print('Starting experiments for', method)
            if method == 'sjf':
                # sort the prompts
                # dict_lst_order = {prompts[i]: output_lengths[i] for i in range(len(prompts))}
                # sorted_prompts = sorted(dict_lst_order, key=dict_lst_order.get)
                # sorted_lengths = [dict_lst_order[key] for key in sorted_prompts]
                # prompts = sorted_prompts
                # output_lengths = sorted_lengths

                # create a dictionary to store prompts and their corresponding output lengths
                prompt_dict = {}
                for i in range(len(prompts)):
                    prompt = prompts[i]
                    length = output_lengths[i]
                    if prompt in prompt_dict:
                        prompt_dict[prompt].append(length)
                    else:
                        prompt_dict[prompt] = [length]

                # sort the dictionary based on the values (output lengths)
                sorted_prompts_lengths = sorted(prompt_dict.items(), key=lambda x: sorted(x[1]))

                # recreate the original lists
                prompts = []
                output_lengths = []
                for prompt, lengths in sorted_prompts_lengths:
                    prompts.extend([prompt] * len(lengths))
                    output_lengths.extend(lengths)

            waiting_time = 0
            jct_list = []
            num_completed_jobs = []
            for i in range(math.ceil(len(prompts) / max_batch_size)):
                start_time = time.perf_counter()
                outputs = llm.generate(prompts[i * max_batch_size : (i+1) * max_batch_size], sampling_params)
                num_completed_jobs.append(len(outputs))
                end_time = time.perf_counter()
                execution_time_ms = int((end_time - start_time) * 1000) + waiting_time
                waiting_time = execution_time_ms
                jct_list.append(execution_time_ms)
            avg_jct = round(np.average(jct_list, weights=num_completed_jobs), 3)
            print('Avg JCT:', avg_jct, 'ms for executing', round(np.mean(num_completed_jobs), 2), 'requests')
            avg_throughput = round(np.mean([num_jobs * 1000 / elapsed_ms for elapsed_ms, num_jobs in zip(jct_list, num_completed_jobs)]), 3)
            print('Mean throughput:', avg_throughput, 'req/s')
            avg_jct_exp_list[method].append(avg_jct)
            throughput_exp_list[method].append(avg_throughput)
            print()

    # end of the experiments
    print('>>>> Statistics over', num_exp, 'experiments:')
    ret_avg_jct = {}
    ret_avg_throughput = {}
    for method in methods:
        print(method)
        print('Avg JCT:', round(np.mean(avg_jct_exp_list[method]), 3), 'ms', '(+/-', str(round(np.std(avg_jct_exp_list[method]), 2)) + ')')
        print('Avg throughput:', round(np.mean(throughput_exp_list[method]), 3), 'req/s', '(+/-', str(round(np.std(throughput_exp_list[method]), 2)) + ')')
        ret_avg_jct[method] = round(np.mean(avg_jct_exp_list[method]), 3)
        ret_avg_throughput[method] = round(np.mean(throughput_exp_list[method]), 3)
    print()
    return ret_avg_jct, ret_avg_throughput


if __name__ == '__main__':
    df = pd.DataFrame(columns=['model', 'max_batch_size', 'method', 'jct', 'throughput'])
    data = []
    methods = ['fcfs', 'sjf']

    model_to_serve = 'facebook/opt-350m'
    model_to_serve_short = 'opt_350m'
    dataset_path = 'sampled_prompts_responses_' + model_to_serve_short + '.csv'

    # load the llm query dataset from csv
    df_prompts = pd.read_csv(dataset_path)
    # print(df_prompts.head())
    print('# of total prompts:', len(df_prompts[df_prompts['model']==model_to_serve]))

    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
    llm = LLM(model=model_to_serve, gpu_memory_utilization=0.9)

    for max_batch_size in [5, 10, 15, 20, 25]:
        print('Max batch size:', max_batch_size)
        jct, throughput = eval(df_prompts, llm, max_batch_size, methods=methods)
        for method in methods:
            data.append({'model': 'opt-350m', 'max_batch_size': max_batch_size, 'method': method, 'jct': jct[method], 'throughput': throughput[method]})

    # clear the GPU memory used by the model
    del llm.llm_engine.driver_worker

    # model_to_serve = 'openai-community/gpt2-medium'
    # model_to_serve_short = 'gpt2_medium'
    # dataset_path = 'sampled_prompts_responses_' + model_to_serve_short + '.csv'

    # # load the llm query dataset from csv
    # df_prompts = pd.read_csv(dataset_path)
    # # print(df_prompts.head())
    # print('# of total prompts:', len(df_prompts[df_prompts['model']==model_to_serve]))

    # sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
    # llm = LLM(model=model_to_serve, gpu_memory_utilization=0.9)

    # for max_batch_size in [5, 10, 15, 20, 25]:
    #     print('Max batch size:', max_batch_size)
    #     jct, throughput = eval(df_prompts, llm, max_batch_size, methods=methods)
    #     for method in methods:
    #         data.append({'model': 'gpt2-medium', 'max_batch_size': max_batch_size, 'method': method, 'jct': jct[method], 'throughput': throughput[method]})

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    print(df.head())
    df.to_csv('results.csv', index=False)
