from typing import List, Dict
import torch
import evaluate

class MetricsTracker:
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        
        # BLEU scores
        bleu_score = self.bleu.compute(
            predictions=predictions,
            references=references
        )
        metrics.update(bleu_score)
        
        # ROUGE scores
        rouge_score = self.rouge.compute(
            predictions=predictions,
            references=references
        )
        metrics.update(rouge_score)
        
        return metrics
