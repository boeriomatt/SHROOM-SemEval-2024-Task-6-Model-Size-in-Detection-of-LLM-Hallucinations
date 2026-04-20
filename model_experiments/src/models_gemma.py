import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.prompts import support_prompt
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

class GemmaJudge:
    """
    Prompt-based hallucination judge using an instruction-tuned Gemma 3 decoder model.
    Interpretation:
    - "yes" -> supported -> Not Hallucination
    - "no"  -> unsupported -> Hallucination
    """
    def __init__(
        self,
        model_name: str,
        max_input_length: int = 512,
        enable_raw_generation: bool = False,
        use_single_token_verbalizers: bool = True,
        use_4bit: bool = False,
        reserve_answer_tokens: int = 8,
        attn_implementation: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.enable_raw_generation = enable_raw_generation
        self.use_single_token_verbalizers = use_single_token_verbalizers
        self.use_4bit = use_4bit
        self.reserve_answer_tokens = reserve_answer_tokens
        self.attn_implementation = attn_implementation

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}

        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        if self.use_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "BitsAndBytesConfig is unavailable. Install/update transformers and bitsandbytes first."
                )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.model.to(self.device)

        self.model.eval()

        if self.use_single_token_verbalizers:
            self.supported_variants = ["yes"]
            self.unsupported_variants = ["no"]
        else:
            self.supported_variants = ["yes", "Yes", "yes.", "Yes."]
            self.unsupported_variants = ["no", "No", "no.", "No."]

    def _build_chat_prompt(self, context: str, hyp: str) -> str:
        user_prompt = support_prompt(context, hyp)
        messages = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _prompt_logprob(self, prompt_text: str, answer: str) -> float:
        prompt_budget = max(1, self.max_input_length - self.reserve_answer_tokens)
        full_text = prompt_text + answer

        prompt_inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_budget,
        )

        full_inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        device = self._model_device()
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

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
        logps = [self._prompt_logprob(prompt_text, variant) for variant in variants]
        max_logp = max(logps)
        return float(max_logp + math.log(sum(math.exp(lp - max_logp) for lp in logps)))

    @torch.no_grad()
    def generate_raw(self, prompt_text: str) -> str:
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        device = self._model_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def predict(self, context: str, hyp: str) -> tuple[str, float, str]:
        prompt_text = self._build_chat_prompt(context, hyp)

        logp_supported = self._aggregate_candidate_logprob(prompt_text, self.supported_variants)
        logp_unsupported = self._aggregate_candidate_logprob(prompt_text, self.unsupported_variants)

        max_logp = max(logp_supported, logp_unsupported)
        p_supported = math.exp(logp_supported - max_logp)
        p_unsupported = math.exp(logp_unsupported - max_logp)
        z = p_supported + p_unsupported

        p_supported /= z
        p_unsupported /= z

        p_hall = float(p_unsupported)
        label = "Hallucination" if p_hall >= 0.5 else "Not Hallucination"

        raw_text = ""
        if self.enable_raw_generation:
            raw_text = self.generate_raw(prompt_text)

        return label, p_hall, raw_text

    def get_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())