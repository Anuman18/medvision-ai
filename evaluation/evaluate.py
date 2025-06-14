from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

def evaluate(generated, reference):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated, reference)[0]
    bleu_score = sentence_bleu([reference.split()], generated.split())

    return {
        "BLEU": round(bleu_score, 4),
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
    }

# Example
if __name__ == "__main__":
    gen = "The chest X-ray shows cardiomegaly and effusion."
    ref = "Cardiomegaly and pleural effusion are seen in the chest X-ray."

    scores = evaluate(gen, ref)
    print("📊 Evaluation Scores:", scores)
