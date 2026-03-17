import json
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import click
from tqdm import tqdm


# Create a custom Dataset class to handle batching
# Custom dataset using batch tokenization
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def compute(predictions: list[str], model_id: str, batch_size: int = 1, stride: int = 128, max_length: int = 2048):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    device = model.device

    def collate_fn(batch):
        encodings = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return encodings

    dataset = TextDataset(predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            prev_end_loc = 0
            encoding = batch['input_ids']
            seq_len = encoding.size(1)
            for begin_loc in tqdm(range(0, seq_len, stride), desc="Windowing"):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encoding[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100


                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                # Accumulate the total negative log-likelihood and the total number of tokens
                num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
                batch_size = target_ids.size(0)
                num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)

    return {"perplexity": ppl.item()}


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--dataset_path', help='Path to the dataset in jsonl format')
@click.option('--output_path', default='perplexity.json', help='Path to save the perplexity value')
@click.option('--batch_size', default=1, help='Batch size for tokenization')
@click.option('--stride', default=128, help='Stride for window ppl')
@click.option('--max_length', default=2048, help='Maximum length for window ppl')
def main(model_path, dataset_path, output_path='perplexity.json', batch_size=1, stride=1024, max_length=2048):
    print(f'Evaluating perplexity on model {model_path} and dataset {dataset_path}...')

    # Assert it is a jsonl file
    assert dataset_path.endswith('.jsonl'), "Dataset must be in jsonl format"

    # If the dataset is a jsonl file, read it into a pandas dataframe
    if dataset_path.endswith('.jsonl'):
        df = pd.read_json(dataset_path, lines=True)
    else:
        # Load from HF dataset
        df = load_dataset(dataset_path, split='dev')
        # Convert to pandas dataframe
        df = df.to_pandas()

    # Assert there is a text column
    assert 'text' in df.columns, "Dataset must have a 'text' column"

    texts = df['text'].tolist()

    # Compute perplexity
    metric = compute(texts, model_path, batch_size=batch_size, stride=stride, max_length=max_length)

    dataset_name = dataset_path.split("/")[-1].replace(".jsonl","")
    if output_path == 'perplexity.json':
        output_path = f'perplexity_{model_path.split("/")[-1].replace(".","_")}_{dataset_name}.json'

    # Save the perplexity value
    with open(output_path, 'w') as f:
        json.dump(metric, f, indent=4)


if __name__ == '__main__':
    main()
