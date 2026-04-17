import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.prompts import support_prompt_qwen

class QwenJudge:
    """
    Prompt-based hallucination judge using a decoder-only Qwen Instruct model.
    Interpretation:
    - "supported"   -> Not Hallucination
    - "unsupported" -> Hallucination
    """
    def __init__(
        self,
        model_name: str,
        max_input_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.supported_variants = [
            "supported",
            "Supported",
            "supported.",
            "Supported.",
        ]
        self.unsupported_variants = [
            "unsupported",
            "Unsupported",
            "unsupported.",
            "Unsupported.",
        ]

    def _build_chat_prompt(self, context: str, hyp: str) -> str:
        """
        Build a chat-formatted prompt for Qwen using the tokenizer's chat template.
        """
        user_prompt = support_prompt_qwen(context, hyp)

        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt_text

    @torch.no_grad()
    def _prompt_logprob(self, prompt_text: str, answer: str) -> float:
        """
        Compute log P(answer | prompt_text) by scoring only the answer tokens conditioned on the chat-formatted prompt.
        """
        full_text = prompt_text + answer

        full_inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        prompt_inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        answer_start = max(prompt_len - 1, 0)

        answer_logprob = token_log_probs[:, answer_start:].sum().item()
        return float(answer_logprob)

    def _aggregate_candidate_logprob(self, prompt_text: str, variants: list[str]) -> float:
        """
        Aggregate multiple verbalizer variants using log-sum-exp.
        """
        logps = [self._prompt_logprob(prompt_text, variant) for variant in variants]
        max_logp = max(logps)
        return float(max_logp + math.log(sum(math.exp(lp - max_logp) for lp in logps)))

    @torch.no_grad()
    def generate_raw(self, prompt_text: str) -> str:
        """
        Deterministic greedy generation for debugging.
        """
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def predict(self, context: str, hyp: str) -> tuple[str, float, str]:
        """
        Returns:
        - label: SHROOM label
        - p_hall: soft hallucination probability in [0,1]
        - raw_text: deterministic greedy generation for debugging
        """
        prompt_text = self._build_chat_prompt(context, hyp)

        logp_supported = self._aggregate_candidate_logprob(
            prompt_text, self.supported_variants
        )
        logp_unsupported = self._aggregate_candidate_logprob(
            prompt_text, self.unsupported_variants
        )

        max_logp = max(logp_supported, logp_unsupported)
        p_supported = math.exp(logp_supported - max_logp)
        p_unsupported = math.exp(logp_unsupported - max_logp)
        z = p_supported + p_unsupported

        p_supported /= z
        p_unsupported /= z

        p_hall = float(p_unsupported)
        label = "Hallucination" if p_hall >= 0.5 else "Not Hallucination"
        raw_text = self.generate_raw(prompt_text)

        return label, p_hall, raw_text

    def get_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())