import argparse
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from train_utils import train_model,eval_model
from Fed import FedAvg
import numpy as np
def run(args):
    #加载不同数据集数据
    dataset_loader1=CQADatasetLoader()
    dataset_loader2=ANLI1DatasetLoader()
    dataset_loader3=ESNLIDatasetLoader()
    dataset_loader4=SVAMPDatasetLoader()
    dataset_loaders=[dataset_loader1,dataset_loader2,dataset_loader3,dataset_loader4]

    dataset1=dataset_loader1.load_from_json()
    dataset2=dataset_loader2.load_from_json()
    dataset3=dataset_loader3.load_from_json()
    dataset4=dataset_loader4.load_from_json()
    datasets=[dataset1,dataset2,dataset3,dataset4]

    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    tokenized_datasets=[]
    for i in range(len(datasets)):
        dataset=datasets[i]
        dataset_loader=dataset_loaders[i]
        train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')

        dataset['train'] = dataset['train'].add_column('llm_label', train_llm_labels)
        dataset['test'] = dataset['test'].add_column('llm_label', test_llm_labels)
        dataset['train'] = dataset['train'].add_column('llm_rationale', train_llm_rationales)
        dataset['test'] = dataset['test'].add_column('llm_rationale', test_llm_rationales)

        if args.subsample < 1.0:
            dataset['train'] = dataset['train'].train_test_split(test_size=1.0 - args.subsample, seed=0)['train']
        if dataset_loader.has_valid:
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
            dataset['valid'] = dataset['valid'].add_column('llm_label', valid_llm_labels)
            dataset['valid'] = dataset['valid'].add_column('llm_rationale', valid_llm_rationales)
        else:
            train_valid_datasets = dataset['train'].train_test_split(test_size=0.1, seed=0)

            dataset = DatasetDict({
                'train': train_valid_datasets['train'],
                'valid': train_valid_datasets['test'],
                'test': dataset['test'],
            })

        if 'rationale' in dataset['train'].column_names:
            dataset = dataset.remove_columns('rationale')
        dataset = dataset.rename_column('llm_rationale', 'rationale')

        if i==1 or i==2:
            dataset = dataset.map(
                lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
                remove_columns=['premise', 'hypothesis'],
            )

        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']],
                                     max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']],
                                          max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )
        tokenized_datasets.append(tokenized_dataset)
        del dataset

    del datasets
    del dataset_loaders
    model = T5ForConditionalGeneration.from_pretrained("model").to("cuda")
    w_glob = model.state_dict()

    for epoch in range(args.epochs):
        print("epoch{}".format(epoch))
        w_locals = []
        for client in range(len(tokenized_datasets)):
            change_state = train_model(args, args.run, tokenizer, tokenized_datasets[client]['train'], w_glob)
            # w_glob=copy.deepcopy(local_model)
            w_locals.append(change_state)
            del change_state
            torch.cuda.empty_cache()
        change_glob = FedAvg(w_locals)
        for k, v in change_glob.items():
            w_glob[k] += v
        accs=[]
        for i in range(len(tokenized_datasets)):
            if i==3:
                args.dataset ='svamp'
            else:
                args.dataset=" "
            acc=eval_model(args,args.run,tokenizer,tokenized_datasets[i]['test'],w_glob)
            accs.append(acc)
        print(f'epoch{epoch}\'s acc={np.mean(accs)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subsample', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--from_pretrained', type=str, default='./model')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--run', type=int, default=0)

    args = parser.parse_args()

    run(args)