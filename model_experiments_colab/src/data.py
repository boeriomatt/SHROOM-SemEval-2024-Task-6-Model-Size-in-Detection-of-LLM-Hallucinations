import json
from pathlib import Path
from typing import Any

def load_json(path: str | Path) -> list[dict[str, Any]]:
    """
    Load data from the json file; example of single example record output: 
    - {'hyp': 'Resembling or characteristic of a weasel.', 'ref': 'tgt', 'src': 'The writer had just entered into his eighteenth year , 
      when he met at the table of a certain Anglo - Germanist an individual , apparently somewhat under thirty , of middle stature , 
      a thin and <define> weaselly </define> figure , a sallow complexion , a certain obliquity of vision , and a large pair of spectacles .', 
      'tgt': 'Resembling a weasel (in appearance).', 'model': '', 'task': 'DM', 
      'labels': ['Hallucination', 'Not Hallucination', 'Not Hallucination', 'Not Hallucination', 'Not Hallucination'], 'label': 'Not Hallucination', 'p(Hallucination)': 0.2}
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def select_context(item: dict[str, Any]) -> str:
    """
    Defines context selection logic based on task type:
    - PG -> use src
    - DM / MT -> use tgt
    """
    task = str(item["task"])
    src = str(item["src"])
    tgt = str(item["tgt"])

    if task == "PG":
        return src
    return tgt

def build_examples(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert raw SHROOM records into a unified internal format; example of single example record output:
    - {'id': None, 'task': 'DM', 'context': 'Resembling a weasel (in appearance).', 'hyp': 'Resembling or characteristic of a weasel.', 'label': 'Not Hallucination', 
      'p_hallucination_gold': 0.2}
    """
    examples = []

    for item in data:
        ex = {
            "id": item.get("id"),
            "task": str(item["task"]),
            "context": select_context(item),
            "hyp": str(item["hyp"]),
            "label": item.get("label"),
            "p_hallucination_gold": item.get("p(Hallucination)"),
        }
        examples.append(ex)

    return examples

def preview_examples(examples: list[dict[str, Any]], n: int = 2) -> None:
    """
    Print a preview of the first n examples in the dataset.
    """
    for i, ex in enumerate(examples[:n]):
        print(f"\n--- Example {i + 1} ---")
        print(f"id: {ex['id']}")
        print(f"task: {ex['task']}")
        print(f"context: {ex['context'][:300]}")
        print(f"hyp: {ex['hyp'][:300]}")
        print(f"label: {ex['label']}")
        print(f"p_hallucination_gold: {ex['p_hallucination_gold']}")