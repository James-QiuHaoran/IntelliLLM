from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
import argparse


def tokenize_function(example):
    example = bert_tokenizer(example["prompt"], truncation=False)
    if len(example['input_ids']) >= 512:
        example['input_ids'] = example['input_ids'][-512: ]
        example['token_type_ids'] = example['token_type_ids'][-512: ]
        example['attention_mask'] = example['attention_mask'][-512: ]
    return example


def calc_percentile(dataset):
    output_token_lengths = []
    for sample in dataset:
        output_token_lengths.append(sample['response_length'])
    s = pd.Series(output_token_lengths)
    print(s.describe(percentiles=[.25, .5, .75, .99]))
    # s = s[s < 2048]
    # sns.histplot(s,
    #          kde=False, 
    #          bins=100, color = 'blue')
    # plt.xlabel('Output Token Length')
    # plt.ylabel('User Requests')
    # plt.savefig('dist.png')
    return dataset


def preprocess_dataset(dataset):
    dataset = Dataset.from_pandas(dataset)

    dataset = dataset.remove_columns(['model', 'round_id'])  # 'conversation_id'
    # Columns left: prompt, response_length, conversation_id

    # Tokenize the user prompt
    # dataset = dataset.map(tokenize_function, batched=True, remove_columns=['prompt'])
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=['prompt'])
    # Columns left: input_ids, token_type_ids, attention_mask, response_length, conversation_id

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=int, help='0 for regression, 1 for binary cls, 2 for multi-cls', default=0)
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification;
    task_type = args.task_type
    if task_type == 1:
        multi_cls_thresholds = [24, 977, 1000000]  # p50, p99, max
    else:
        multi_cls_thresholds = [16, 24, 35, 977, 1000000]  # p25, p50, p75, p99, max

    # proxy-model
    proxy_model_name = 'bert-base-uncased'
    bert_tokenizer = AutoTokenizer.from_pretrained(proxy_model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    dataset_path = 'sampled_prompts_responses_opt_350m_14K.csv'
    model_name = 'opt-350m'
    dataset = pd.read_csv(dataset_path)
    selected_data_size = len(dataset)
    print(f'Selected data size: {selected_data_size}')

    dataset_path = model_name + '_'
    dataset_path = dataset_path + 'reg_' if task_type == 0 else dataset_path + 'cls_' if task_type == 1 else dataset_path + 'multi_cls_'
    dataset_path = 'data/' + dataset_path + f'{max(1, int(selected_data_size / 1000))}K'

    dataset = preprocess_dataset(dataset)

    if task_type != 0:
        dataset = calc_percentile(dataset)

    dataset.set_format("torch")

    # print(len(dataset))
    # print(dataset.column_names)
    # print(dataset[0])

    dataset.save_to_disk(dataset_path)
    print('Saved dataset to ' + dataset_path)
