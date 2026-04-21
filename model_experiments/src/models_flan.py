import math
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class FlanJudge:
    """
    Prompt-based hallucination judge built on a FLAN-T5-style seq2seq model; the model is asked whether a hypothesis is supported by a given context;
    internally, the judge scores the relative probability of "yes" versus "no" and converts that into:
    - a binary SHROOM label
    - a soft hallucination probability p(Hallucination)
    Interpretation:
    - "yes"  -> supported -> Not Hallucination
    - "no"   -> unsupported -> Hallucination
    """
    def __init__(
        self,
        model_name: str,
        max_input_length: int = 512, # Use a fixed max input length to balance context coverage and efficiency; 512 tokens is typically sufficient for SHROOM-style prompts
    ) -> None:
        """
        Load tokenizer and model, move model to device, and prepare answer variants.
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.device = "cpu" # Manually set device to CPU to avoid GPU usage, even if a GPU is available

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Multiple surface forms are included so the score is not sensitive to capitalization differences
        self.yes_variants = ["yes", "Yes"]
        self.no_variants = ["no", "No"]

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        """
        Tokenize the input prompt for the encoder side of the seq2seq model.
        """
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

    @torch.no_grad()
    def _encode_target(self, text: str) -> torch.Tensor:
        """
        Tokenize a candidate decoder target sequence such as 'yes' or 'no'.
        """
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def _sequence_logprob(self, prompt: str, answer: str) -> float:
        """
        Compute log P(answer | prompt) using teacher forcing.
        The seq2seq model receives:
        - the prompt as encoder input
        - the candidate answer as decoder target
        Hugging Face returns the mean cross-entropy loss over the target tokens; since cross-entropy is negative log-likelihood, we approximate:
        - total log-probability ~= -(mean loss * number of target tokens)
        This is used to compare how strongly the model prefers short answers such as "yes" and "no".
        """
        encoder_inputs = self._encode_prompt(prompt)
        target_ids = self._encode_target(answer)

        outputs = self.model(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs["attention_mask"],
            labels=target_ids,
        )

        seq_len = target_ids.shape[1]
        mean_loss = outputs.loss.item()
        logprob = -mean_loss * seq_len
        return float(logprob)

    def _aggregate_variant_logprob(self, prompt: str, variants: list[str]) -> float:
        """
        Combine multiple answer variants using log-sum-exp.
        Example:
        - yes variants: ["yes", "Yes"]
        - no variants:  ["no", "No"]
        This gives one combined log-probability for the semantic answer.
        """
        logps = [self._sequence_logprob(prompt, variant) for variant in variants]

        max_logp = max(logps)
        return float(
            max_logp + math.log(sum(math.exp(lp - max_logp) for lp in logps))
        )

    @torch.no_grad()
    def generate_raw(self, prompt: str) -> str:
        """
        Generate the model's raw answer with greedy decoding; this is mainly useful for debugging and inspection; it is not the main scoring mechanism.
        """
        inputs = self._encode_prompt(prompt)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _normalize_yes_no(self, logp_yes: float, logp_no: float) -> tuple[float, float]:
        """
        Convert two log-scores into normalized probabilities over {yes, no}.
        """
        max_logp = max(logp_yes, logp_no)
        p_yes = math.exp(logp_yes - max_logp)
        p_no = math.exp(logp_no - max_logp)
        z = p_yes + p_no

        return p_yes / z, p_no / z

    def predict(self, prompt: str) -> tuple[str, float, str]:
        """
        Predict a SHROOM label and soft hallucination probability.
        Returns:
        - label: "Hallucination" or "Not Hallucination"
        - p_hall: soft hallucination probability in [0, 1]
        - raw_text: raw greedy model output for debugging
        """
        logp_yes = self._aggregate_variant_logprob(prompt, self.yes_variants)
        logp_no = self._aggregate_variant_logprob(prompt, self.no_variants)

        p_yes, p_no = self._normalize_yes_no(logp_yes, logp_no)

        # "no" means the hypothesis is not supported by the context, so its probability is treated as hallucination probability
        p_hall = float(p_no)

        label = "Hallucination" if p_hall >= 0.5 else "Not Hallucination"
        raw_text = self.generate_raw(prompt)

        return label, p_hall, raw_text

    def get_num_parameters(self) -> int:
        """
        Return the total number of model parameters.
        """
        return sum(parameter.numel() for parameter in self.model.parameters())