from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bertscore_score

# from llm_judge.correctness import LLMCorrectnessEvaluator
# Load your custom model and tokenizer
model_path = "nasa-impact/nasa-smd-ibm-st-v2"
indus = SentenceTransformer("nasa-impact/nasa-smd-ibm-st-v2")
custom_tokenizer = AutoTokenizer.from_pretrained(model_path)


def cosine_sim(references, predictions):
    query = indus.encode(predictions)
    targets = indus.encode(references)
    return util.cos_sim(query, targets).item()


def bertscore_indus(
    references: list[str], predictions: list[str], threshold=0.50
) -> dict[str, float]:
    # Compute BERTScore with custom model and tokenizer
    P, R, F1 = bertscore_score(
        cands=predictions,
        refs=references,
        model_type=model_path,  # Use model path here
        num_layers=12,  # Adjust if needed for your model
        verbose=False,
        idf=False,
        lang="en",
    )

    f1 = F1[0].item()
    precision = P[0].item()
    recall = R[0].item()
    accuracy = 1 if f1 > threshold else 0

    result = {
        "bertscore_f1": f1,
        "bertscore_precision": precision,
        "bertscore_recall": recall,
        "bertscore_accuracy": accuracy,
    }
    return result


def rouge(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(references[0], predictions[0])
    return {"rouge1": scores["rouge1"].fmeasure, "rougeL": scores["rougeL"].fmeasure}


def test_metric(references, predictions):
    print(references, predictions)
    return cosine_sim(references, predictions)


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

    if len(jaccard_scores) == 0:
        return 0
    return sum(jaccard_scores) / len(jaccard_scores)
