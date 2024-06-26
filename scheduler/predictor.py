import argparse
import datasets
from datasets import load_dataset
import transformers
from transformers import AutoConfig, AutoTokenizer, BertModel, DataCollatorWithPadding
import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
import time


class BertClassificationModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim, num_classes):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        # Fix the weights of the pretrained model
        if not FLAG_BERT_TUNING:
            for param in self.bert.parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.logsoftmax(self.fc2(output))
        return output


class BertRegressionModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        # Fix the weights of the pretrained model
        if not FLAG_BERT_TUNING:
            for param in self.bert.parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output


def generate_dataloaders(dataset, train_batch_size, tokenizer):
    train_validationtest = dataset.train_test_split(test_size=0.4, shuffle=False)
    validation_test = train_validationtest['test'].train_test_split(test_size=0.5, shuffle=False)
    train_dataset = train_validationtest['train']
    validation_dataset = validation_test['train']
    test_dataset = validation_test['test']

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size, collate_fn=data_collator)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=data_collator)
    weights = []
    if TASK_TYPE == 1 or TASK_TYPE == 2:
        for i in range(num_classes):
            n_samples_for_label_i = len(dataset.filter(lambda example: example["labels"] == i)['labels'])
            print('Number of samples for class ' + str(i) + ': ' + str(n_samples_for_label_i))
            if n_samples_for_label_i == 0:
                weights.append(0.0)
            else:
                weights.append(1.0 / n_samples_for_label_i)
    return train_dataloader, validation_dataloader, test_dataset, weights


def write_loss_to_file(training_loss_list, validation_loss_list):
    cur_dir = os.path.dirname(__file__)
    train_dir = os.path.join(cur_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join(train_dir, date_time + f'-size_{int(selected_data_size / 1000)}K.txt'), 'w') as f:
        f.write('Training loss:\n')
        for loss in training_loss_list:
            f.write(str(loss) + '\t')
        f.write('\nValidation loss:\n')
        for loss in validation_loss_list:
            f.write(str(loss) + '\t')
        f.write('\n')


def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device):
    num_training_steps = num_epochs * len(train_dataloader)
    # Using a learning rate with a linear decay
    lr_scheduler = transformers.get_scheduler(
        'linear',
        # 'constant',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    training_loss_list = []
    validation_loss_list = []
    if FLAG_WRITE_RESULTS:
        writer = SummaryWriter()

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0
        model.train()
        # Fix the BERT weights after 3 training epochs
        if FLAG_BERT_TUNING and epoch == 3:
            for param in model.bert.parameters():
                param.requires_grad = False
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            if TASK_TYPE == 0:
                labels = batch['response_length'].to(device)
            else:
                labels = batch['labels'].to(device)
            if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
                loss = criterion(output, labels.float())
            else:
                loss = criterion(output, labels)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            training_loss += loss.item()

        if FLAG_WRITE_RESULTS:
            writer.add_scalar("Loss/train", training_loss / len(train_dataloader), epoch)
        print(f"Training loss for epoch {epoch}: {training_loss / len(train_dataloader)}")
        training_loss_list.append(training_loss / len(train_dataloader))
        if epoch % 1 == 0:
            if TASK_TYPE == 0:
                validation_metrics = eval_regression(model, validation_dataloader, device)
            elif TASK_TYPE == 3 or TASK_TYPE == 4:
                validation_metrics = eval_regression(model, validation_dataloader, device)
                validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device)
            else:
                validation_metrics = eval_classification(model, validation_dataloader, device)
            print(f'Validation loss after epoch {epoch}: ')
            for k, v in validation_metrics.items():
                print(f'{k}: {v:.4f}', end='\t')
            print(' ')
    if FLAG_WRITE_RESULTS:
        writer.flush()
        writer.close()
        write_loss_to_file(training_loss_list, validation_loss_list)


def eval_classification(model, dataloader, device):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")
    model.eval()
    labels = []
    predictions = []
    for batch in dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            label = batch['labels'].to(device)

            if TASK_TYPE != 3 and TASK_TYPE != 4:
                prediction = torch.argmax(output, dim=-1)
            else:
                prediction = torch.round(output).type(torch.LongTensor)
                for i in range(len(prediction)):
                    if prediction[i] >= num_classes:
                        prediction[i] = num_classes - 1
                    elif prediction[i] < 0:
                        prediction[i] = 0
            labels.extend(label)
            predictions.extend(prediction)
    metric = accuracy_metric.compute(references=labels, predictions=predictions) | \
        f1_metric.compute(references=labels, predictions=predictions, average='macro') | \
        precision_metric.compute(references=labels, predictions=predictions, average='macro') | \
        recall_metric.compute(references=labels, predictions=predictions, average='macro')
    return metric


def eval_regression(model, dataloader, device):
    l1loss = nn.L1Loss()
    mseloss = nn.MSELoss()
    model.eval()

    l1err = 0
    mse = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prediction = model(input_ids=input_ids, attention_mask=attention_mask)
            if TASK_TYPE == 0:
                labels = batch['response_length'].to(device)
            else:
                labels = batch['labels'].to(device)
            l1err += l1loss(prediction, labels.type_as(prediction))
            mse += mseloss(prediction, labels.type_as(prediction))

    metric = {'L1 error': l1err.item() / len(dataloader), 'MSE': mse.item() / len(dataloader)}
    return metric


def predict(model, dataloader, device):
    model.eval()
    predicted_labels = []
    conversation_ids = []
    # round_ids = []
    actual_lengths = []
    latencies = []
    print_model_names = []
    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
                lengths = batch['response_length']
                predictions = predictions
            else:
                predictions = torch.argmax(predictions, dim=-1)
                lengths = batch['num_tokens']
            end_time = time.time()

            # associate the conversation_id and round_id with the predicted label
            conversation_ids_batch = batch['conversation_id']
            # round_ids_batch = batch['round_id']

            predicted_labels.extend(predictions.cpu().numpy())
            conversation_ids.extend(conversation_ids_batch.numpy())
            # round_ids.extend(round_ids_batch.numpy())
            actual_lengths.extend(lengths.numpy())
            latencies.append(end_time - start_time)
            for _ in range(len(input_ids)):
                print_model_names.append(model_name)

    df = pd.DataFrame({'actual_length': actual_lengths,
                       'predicted_label': predicted_labels,
                       'conversation_id': conversation_ids,
                    #    'round_id': round_ids,
                       'latency': latencies, 'model_name': print_model_names})
    return df


def get_output_file_name():
    output_filename = 'predictions_'
    if FLAG_BERT_TUNING:
        output_filename += 'warmup_'
    if FLAG_TINY_BERT:
        output_filename += 'berttiny_'
    if TASK_TYPE == 0:
        output_filename += 'reg_'
        output_filename += 'l1_' if FLAG_L1_LOSS else 'mse_'
    elif TASK_TYPE == 1:
        output_filename += 'cls_'
    elif TASK_TYPE == 2:
        output_filename += 'multi_cls_'
    elif TASK_TYPE == 3:
        output_filename += 'ordinal_multi_cls_'
        output_filename += 'l1_' if FLAG_L1_LOSS else 'mse_'
    elif TASK_TYPE == 4:
        output_filename += 'ordinal_cls_'
        output_filename += 'l1_' if FLAG_L1_LOSS else 'mse_'
    output_filename += f'{max(1, int(selected_data_size / 1000))}K.csv'
    return output_filename


def get_dataset_path():
    dataset_path = 'data/' + model_name + '_'

    if TASK_TYPE == 0:
        dataset_path += f'reg_{max(1, int(selected_data_size / 1000))}K'
    elif TASK_TYPE == 1 or TASK_TYPE == 4:
        dataset_path = f'cls_{max(1, int(selected_data_size / 1000))}K'
    elif TASK_TYPE == 2 or TASK_TYPE == 3:
        # multi_cls or ordinal_cls:
        dataset_path = f'multi_cls_{max(1, int(selected_data_size / 1000))}K'
    return dataset_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_tiny', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, help='0: regression, 1: binary cls, 2: multi-cls, 3: multi-cls ordinal, 4: bi-cls ordinal', default=0)
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification; 
    # 3: multi-class ordinal classification; 4: bi-class ordinal classification; 

    TASK_TYPE = args.task_type
    FLAG_LOAD_MODEL_WEIGHTS = False
    FLAG_SAVE_MODEL_WEIGHTS = False if FLAG_LOAD_MODEL_WEIGHTS else True
    FLAG_BERT_TUNING = True
    FLAG_TINY_BERT = args.bert_tiny
    FLAG_L1_LOSS = args.l1_loss
    FLAG_WRITE_RESULTS = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prompt-response dataset
    model_name = 'opt-350m'
    selected_data_size = 14000  # len(dataset)
    dataset_path = get_dataset_path()
    dataset = datasets.load_from_disk(dataset_path)
    print(f'Loaded dataset from ' + dataset_path)
    print(len(dataset))
    # print(dataset.column_names)
    # print(dataset[0])
    # print(len(dataset))

    # proxy-model for prediction
    proxy_model_name = 'prajjwal1/bert-tiny' if FLAG_TINY_BERT else 'bert-base-uncased'

    num_classes = 3 if (TASK_TYPE == 1 or TASK_TYPE == 4) else 5
    bert_tokenizer = AutoTokenizer.from_pretrained(proxy_model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    output_filename = get_output_file_name()
    print('Output file name:', output_filename)

    num_epochs = 6
    train_batch_size = 16
    test_batch_size = 1
    lr = 1e-5 if FLAG_BERT_TUNING else 1e-4

    train_dataloader, validation_dataloader, test_dataset, weights = generate_dataloaders(dataset, train_batch_size, bert_tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size, collate_fn=data_collator)
    config = AutoConfig.from_pretrained(proxy_model_name)
    if TASK_TYPE == 1 or TASK_TYPE == 2:
        print('Cross entropy weights: ')
        print(weights)

    # regression or ordinal classification
    if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
        model = BertRegressionModel(config, proxy_model_name, hidden_dim=128).to(device)
        if FLAG_L1_LOSS:
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
    # classification
    elif TASK_TYPE == 1 or TASK_TYPE == 2:
        model = BertClassificationModel(config, proxy_model_name, hidden_dim=128, num_classes=num_classes).to(device)
        # criterion = nn.NLLLoss()
        criterion = nn.NLLLoss(weight=torch.tensor(weights).to(device))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    if FLAG_LOAD_MODEL_WEIGHTS:
        model.load_state_dict(torch.load('./models/' + output_filename.split('.')[0] + '.pth'))
        model.to(device)
        print("Loaded model weights from disk.")
    else:
        # Training
        print("Start training...")
        train(model,
              criterion,
              optimizer,
              train_dataloader,
              validation_dataloader,
              num_epochs,
              device)

    if TASK_TYPE == 0:
        validation_metrics = eval_regression(model, validation_dataloader, device)
    elif TASK_TYPE == 3 or TASK_TYPE == 4:
        validation_metrics = eval_regression(model, validation_dataloader, device)
        validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device)
    else:
        validation_metrics = eval_classification(model, validation_dataloader, device)
    print(f'Validation metrics after training:')
    for k, v in validation_metrics.items():
        print(f'{k}: {v:.4f}')

    if FLAG_SAVE_MODEL_WEIGHTS:
        # Create ./models directory if it doesn't exist
        os.makedirs('./models', exist_ok=True)
        torch.save(model.state_dict(), './models/' + output_filename.split('.')[0] + '.pth')


    # Inference
    print("Start inference...")
    df = predict(model, test_dataloader, device)
    # Create ./results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/' + output_filename)
    print('Saved results to ./results/' + output_filename)

    if TASK_TYPE == 0:
        validation_metrics = eval_regression(model, test_dataloader, device)
    elif TASK_TYPE == 3 or TASK_TYPE == 4:
        validation_metrics = eval_regression(model, test_dataloader, device)
        validation_metrics = validation_metrics | eval_classification(model, test_dataloader, device)
    else:
        validation_metrics = eval_classification(model, test_dataloader, device)
    print(f'Metrics on test set:')
    os.makedirs('./metrics', exist_ok=True)
    with open('./metrics/' + output_filename.split('.')[0] + '.txt', 'a') as f:
        for k, v in validation_metrics.items():
            f.write(f'{k}: {v:.4f}\n')
            print(f'{k}: {v:.4f}')
