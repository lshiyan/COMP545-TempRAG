from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
import os

load_dotenv()

class Reranker:
    def __init__(self, device: str = None):
        """
        Initialize the reranker model and tokenizer.

        Args:
            device (str): "cuda" if GPU is available, else "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = os.getenv("RERANKER_MODEL")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

    def _format_prompt(self, query: str, doc: str) -> str:
        """
        Format the reranking prompt. Adjust this based on your model's training format.
        
        Common format for rerankers:
        "Query: {query}\nDocument: {doc}\nRelevant:"
        """
        return f"Query: {query}\nDocument: {doc}\nRelevant:"

    @torch.no_grad()
    def score_batch(self, query: str, docs: List[str]) -> List[float]:
        """
        Computes pointwise relevance scores for a batch of (query, doc) pairs.

        Args:
            query (str): The decomposed subquery from your agent
            docs (List[str]): A list of candidate documents/edge texts

        Returns:
            List[float]: Relevance scores (0-1) aligned with input order
        """
        prompts = [self._format_prompt(query, doc) for doc in docs]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64 #NOTE: Might have to change this depending on if we have longer documents or not.
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        yes_logits = logits[:, self.yes_token_id]
        no_logits = logits[:, self.no_token_id]

        scores_tensor = torch.stack([no_logits, yes_logits], dim=1)
        probs = F.softmax(scores_tensor, dim=1)[:, 1]
        
        return probs.cpu().tolist()

    @torch.no_grad()
    def rerank(self, query: str, candidates: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """
        Reranks a list of candidates using batch scoring.

        Args:
            query (str): The agent-resolved query string
            candidates (List[str]): List of candidate documents/edges
            top_k (int): Optional, return only the top-k ranked results

        Returns:
            List[(candidate, score)]: Sorted in decreasing relevance
        """
        if not candidates:
            return []
        
        scores = self.score_batch(query, candidates)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = [cand for cand, _ in scored] #Extract candidate sentences.
        
        return scored[:top_k] if top_k else scored