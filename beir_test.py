import torch
import numpy as np
import json
from datetime import datetime
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoTokenizer, ModernBertForMaskedLM

class BiCEModel:
    def __init__(self, model_path, model_type="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", add_prefix_space=True)
        
        if model_type == "auto":
            try:
                from src import BiCEBertForMaskedLM
                model = BiCEBertForMaskedLM.from_pretrained(model_path)
                self.model_type = "bice"
                print(f"Loaded BiCEBertForMaskedLM from {model_path}")
            except:
                model = ModernBertForMaskedLM.from_pretrained(model_path)
                self.model_type = "modernbert"
                print(f"Loaded ModernBertForMaskedLM from {model_path}")
        elif model_type == "bice":
            from src import BiCEBertForMaskedLM
            model = BiCEBertForMaskedLM.from_pretrained(model_path)
            self.model_type = "bice"
        elif model_type == "modernbert":
            model = ModernBertForMaskedLM.from_pretrained(model_path)
            self.model_type = "modernbert"
        
        self.encoder = model.base_model
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device).eval()
        
    def encode(self, sentences, batch_size=32, **kwargs):
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                      max_length=512, return_tensors="pt").to(self.device)
                
                outputs = self.encoder(**inputs)
                hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_queries(self, queries, batch_size=32, **kwargs):
        return self.encode(queries, batch_size, **kwargs)
    
    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        sentences = []
        for doc in corpus:
            text = doc.get("text", "")
            title = doc.get("title", "")
            if title:
                text = title + " " + text
            sentences.append(text)
        return self.encode(sentences, batch_size, **kwargs)

def test_beir(model_path, dataset="scifact", model_type="auto"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, f"./datasets/{dataset}")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    print(f"Dataset: {dataset}")
    print(f"Corpus: {len(corpus)}, Queries: {len(queries)}")
    
    model = BiCEModel(model_path, model_type)
    retriever = DRES(model, batch_size=64)
    
    results = retriever.search(corpus, queries, top_k=100, score_function="cos_sim")
    
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [1, 3, 5, 10, 100])
    
    metrics = {
        "NDCG@1": float(ndcg['NDCG@1']),
        "NDCG@3": float(ndcg['NDCG@3']),
        "NDCG@5": float(ndcg['NDCG@5']),
        "NDCG@10": float(ndcg['NDCG@10']),
        "NDCG@100": float(ndcg['NDCG@100']),
        "MAP@1": float(_map['MAP@1']),
        "MAP@3": float(_map['MAP@3']),
        "MAP@5": float(_map['MAP@5']),
        "MAP@10": float(_map['MAP@10']),
        "MAP@100": float(_map['MAP@100']),
        "Recall@1": float(recall['Recall@1']),
        "Recall@3": float(recall['Recall@3']),
        "Recall@5": float(recall['Recall@5']),
        "Recall@10": float(recall['Recall@10']),
        "Recall@100": float(recall['Recall@100']),
        "P@1": float(precision['P@1']),
        "P@3": float(precision['P@3']),
        "P@5": float(precision['P@5']),
        "P@10": float(precision['P@10']),
        "P@100": float(precision['P@100'])
    }
    
    print(f"\nNDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"Recall@100: {metrics['Recall@100']:.4f}")
    print(f"MAP@100: {metrics['MAP@100']:.4f}")
    
    return metrics

if __name__ == "__main__":
    datasets = ["scifact", "nfcorpus", "arguana"]
    
    models_to_test = [
        ("BiCEbert-mlm/checkpoint-8000", "bice"),
        ("modern_bert/checkpoint-4000", "modernbert")
    ]
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    for model_path, model_type in models_to_test:
        print(f"\nTesting {model_type} model: {model_path}")
        all_results["models"][model_path] = {
            "model_type": model_type,
            "datasets": {}
        }
        
        for dataset in datasets:
            try:
                print(f"\n{'='*40}")
                print(f"Testing {dataset}")
                print(f"{'='*40}")
                metrics = test_beir(model_path, dataset, model_type)
                all_results["models"][model_path]["datasets"][dataset] = metrics
                print(f"Final NDCG@10: {metrics['NDCG@10']:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                all_results["models"][model_path]["datasets"][dataset] = {"error": str(e)}
                import traceback
                traceback.print_exc()
    
    # Save results to JSON
    output_file = f"beir_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_path, model_data in all_results["models"].items():
        print(f"\nModel: {model_path} ({model_data['model_type']})")
        for dataset, metrics in model_data["datasets"].items():
            if "error" not in metrics:
                ndcg10 = metrics['NDCG@10']
                recall100 = metrics['Recall@100']
                print(f"  {dataset:12s} | NDCG@10: {ndcg10:.4f} | Recall@100: {recall100:.4f}")
            else:
                print(f"  {dataset:12s} | ERROR: {metrics['error']}")