import json
from datasets import load_dataset
import pandas as pd
import click
from model import VllmModel
import re


def get_answers(answer):
    answers_list = []
    matches = re.findall(r"(?:The answer is: |The answers are: )([A-Z](?:,\s*[A-Z])*)", answer)
    if matches:
        match = matches[0]
        # Split on spaces and strip
        answers_list = [ans.strip() for ans in match.split(",")]
    return answers_list


def subset_accuracy(references, predictions):
    correct_count = 0
    for correct, pred in zip(references, predictions):
        if set(correct) == set(pred):
            correct_count += 1
    return correct_count / len(references)


def jaccard_index(references, predictions):
    jaccard_scores = []
    for correct, pred in zip(references, predictions):
        intersection = len(set(correct) & set(pred))
        union = len(set(correct) | set(pred))
        jaccard_scores.append(intersection / union)
    return sum(jaccard_scores) / len(jaccard_scores)


def compute(model_id: str, questions: list[str], references: list[list[str]], generation_args: dict):
    model = VllmModel(model_id, model_id)
    outputs = model.generate(questions, generation_args)

    predictions = [get_answers(output) for output in outputs]

    accuracy = subset_accuracy(references, predictions)
    jaccard = jaccard_index(references, predictions)

    return {"accuracy": accuracy, "jaccard": jaccard}


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--dataset_path', help='Path to the dataset in jsonl format')
@click.option('--output_path', default=None, help='Path to save the perplexity value')
def main(model_path, dataset_path, output_path='q&a.json'):
    print(f'Evaluating perplexity on model {model_path} and dataset {dataset_path}...')

    # Assert it is a jsonl file
    #assert dataset_path.endswith('.jsonl'), "Dataset must be in jsonl format"

    # If the dataset is a jsonl file, read it into a pandas dataframe
    if dataset_path.endswith('.jsonl'):
        df = pd.read_json(dataset_path, lines=True)
    else:
        # Load from HF dataset
        df = load_dataset(dataset_path, split='train')
        # Convert to pandas dataframe
        df = df.to_pandas()

    # Assert there is a text column
    assert 'instruction' in df.columns, "Dataset must have a 'instruction' column"
    assert 'input' in df.columns, "Dataset must have a 'input' column"
    assert 'output' in df.columns, "Dataset must have a 'output' column"

    df['references'] = df['output'].apply(lambda x: get_answers(x))
    print(df['references'])

    # Prompt
    instruction = '''You are a knowledge expert, you are tasked to answer the following multiple-choice question (there may be more than one correct option). Give your final answer in the format of 'The answer is (chosen multiple-choice options)'.'''

    # Combine instruction and question together:
    df['question'] = instruction + "\n\n" + df['input']

    

    generation_args = {'temperature': 0.1, 'top_p': 0.75}
    # Compute metrics
    metric = compute(model_id=model_path, questions=df['question'].tolist(), references=df['references'].tolist(),
                    generation_args=generation_args)

    print(len(df['question'].tolist()))
    dataset_name = dataset_path.split("/")[-1].replace(".jsonl", "")
    if output_path is None:
        output_path = f'perplexity_{model_path.split("/")[-1].replace(".", "_")}_{dataset_name}.json'

    # Save the perplexity value
    with open(output_path, 'w') as f:
        json.dump(metric, f, indent=4)


if __name__ == '__main__':
    main()
