import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM

def load_model(model_dir="BiCEbert-mlm-final"):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", add_prefix_space=True)
    model = ModernBertForMaskedLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device

def predict_masks(texts, model_dir="BiCEbert-mlm-final", top_k=5, max_length=512, batch_size=8):
    tokenizer, model, device = load_model(model_dir)
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError("mask_token is not set in tokenizer")
    mask_tok = tokenizer.mask_token
    norm_texts = [t.replace("[MASK]", mask_tok) for t in texts]
    results = []
    with torch.no_grad():
        for i in range(0, len(norm_texts), batch_size):
            batch_texts = norm_texts[i : i + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc)[0]
            for b, txt in enumerate(batch_texts):
                ids = enc["input_ids"][b]
                pos = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                preds_for_sample = []
                for p in pos.tolist():
                    probs = torch.softmax(logits[b, p], dim=-1)
                    values, indices = torch.topk(probs, top_k)
                    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
                    preds_for_sample.append(list(zip(tokens, values.tolist())))
                results.append({"text": txt, "predictions": preds_for_sample})
    return results

if __name__ == "__main__":
    examples = [
    "[MASK] a the capital of France.",
]
    
    examples = [
    "Paris [MASK] the capital of France.",
    "Paris a the [MASK] of France.",
    "The sun [MASK] yellow.",
    "Dogs [MASK] four legs.", 
    "Water [MASK] at 100 degrees.",
    "The sky [MASK] blue.",
    "Two plus two [MASK] four.",
    "Birds can [MASK].",
    "Fish live in [MASK].",
    "Snow is [MASK].",
    "Elephants are [MASK] animals.",
    "Monday comes [MASK] Sunday.",
    "People [MASK] with their eyes.",
    "The [MASK] shines during the day.",
    "Cats say [MASK].",
    "We [MASK] food when hungry.",
    "Winter is [MASK] than summer.",
    "Roses are [MASK].",
    "People have two [MASK].",
    "Fire is [MASK].",
    "Babies [MASK] milk.",
]
    out = predict_masks(examples, model_dir="modern_bert/checkpoint-4000", top_k=5)
    for item in out:
        print(item["text"])
        for mi, cand in enumerate(item["predictions"], 1):
            print(f"  <mask> #{mi}: " + ", ".join([f"{t}:{p:.3f}" for t, p in cand]))
