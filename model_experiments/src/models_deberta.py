import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DebertaJudge:
    """
    NLI-based hallucination judge using a DeBERTa-v3 cross-encoder.
    Interpretation:
    - entailment -> supported -> Not Hallucination
    - neutral / contradiction -> unsupported -> Hallucination
    Soft score:
    - p(Hallucination) = 1 - p(entailment)
    """
    def __init__(
        self,
        model_name: str,
        max_input_length: int = 512, # Use a fixed max input length to balance context coverage and efficiency; 512 tokens is typically sufficient for SHROOM-style prompts
    ) -> None:
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Example: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, context: str, hyp: str) -> tuple[str, float, str]:
        """
        Predict SHROOM label and p(Hallucination) from NLI outputs.
        Returns:
        - label: "Hallucination" or "Not Hallucination"
        - p_hall: 1 - p(entailment)
        - raw_text: predicted NLI label for debugging
        """
        inputs = self.tokenizer(
            context,
            hyp,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        entailment_idx = None
        predicted_idx = int(torch.argmax(probs).item())

        for idx, label in self.id2label.items():
            if str(label).lower() == "entailment":
                entailment_idx = int(idx)
                break

        if entailment_idx is None:
            raise ValueError(
                f"Could not find 'entailment' in id2label mapping: {self.id2label}"
            )

        p_entail = float(probs[entailment_idx].item())
        p_hall = 1.0 - p_entail

        label = "Hallucination" if p_hall >= 0.5 else "Not Hallucination"
        raw_text = str(self.id2label[predicted_idx])

        return label, p_hall, raw_text

    def get_num_parameters(self) -> int:
        """
        Return the total number of model parameters.
        """
        return sum(parameter.numel() for parameter in self.model.parameters())