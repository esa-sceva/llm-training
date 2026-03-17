from perplexity import compute
from datasets import load_dataset
import click
import json


@click.command()
@click.option("--model_path", help="Path to the model checkpoint")
@click.option("--output_path", default="perplexity.json", help="Path to save the perplexity value")
@click.option("--split", default="validation", help="Dataset split to evaluate on")
@click.option("--stride", default=4096, help="Stride for window ppl")
@click.option("--max_length", default=8192, help="Maximum length for window ppl")
@click.option("--batch_size", default=1, help="Batch size for tokenization")
def main(
    model_path,
    output_path="perplexity.json",
    split="validation",
    stride=4096,
    max_length=8192,
    batch_size=1,
):
    # Load Mini Pile
    dataset = load_dataset("JeanKaddour/minipile", split=split)

    # Compute perplexity
    perplexity = compute(
        dataset["text"],
        batch_size=batch_size,
        model_id=model_path,
        stride=stride,
        max_length=max_length,
    )

    if output_path == "perplexity.json":
        output_path = f'perplexity_{model_path.split("/")[-1].replace(".","_")}_mini_pile.json'

    # Save the perplexity value
    with open(output_path, "w") as f:
        json.dump(perplexity, f, indent=4)


if __name__ == "__main__":
    main()
