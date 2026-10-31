"""The three questions, defined once in request.json, and how to read an answer back."""

import hashlib
import json
from pathlib import Path

REQUEST = Path(__file__).resolve().parent.parent / "request.json"

# Both systems get exactly these questions, in this option order: Laya's answers change with it
QUESTIONS: dict = json.loads(REQUEST.read_text())["questions"]


# Option labels of `food_group`, keyed by the Open Food Facts tag they stand for
FOOD_GROUP_OPTION = {
    "en:alcoholic-beverages": "alcoholic beverages",
    "en:beverages": "beverages",
    "en:cereals-and-potatoes": "cereals and potatoes",
    "en:composite-foods": "composite foods",
    "en:fats-and-sauces": "fats and sauces",
    "en:fish-meat-eggs": "fish, meat and eggs",
    "en:fruits-and-vegetables": "fruits and vegetables",
    "en:milk-and-dairy-products": "milk and dairy products",
    "en:salty-snacks": "salty snacks",
    "en:sugary-snacks": "sugary snacks",
}
assert list(FOOD_GROUP_OPTION.values()) == list(QUESTIONS["food_group"]["criteria"])

GRADES = "ABCDE"  # the Nutri-Score levels, index 0 first
OPTIONS = {
    "food_group": list(QUESTIONS["food_group"]["criteria"]),
    "ultra_processed": ["yes", "no"],
    "nutriscore": list(GRADES),
}
assert list(OPTIONS) == list(QUESTIONS)


def questions_sha256(questions: dict) -> str:
    """Pins the wording and option order of a run, so a resumed run cannot mix two versions."""
    return hashlib.sha256(json.dumps(questions).encode()).hexdigest()


def probabilities(answer: dict, question: str) -> dict[str, float]:
    """The probability of every option of `question`, in the order of OPTIONS."""
    if answer["type"] == "noul":
        # Rounded to drop float noise: 1 - 0.7 is 0.30000000000000004
        return {"yes": answer["noul"], "no": round(1 - answer["noul"], 12)}
    if answer["type"] == "score":
        return {grade: answer["probabilities"][str(level)] for level, grade in enumerate(GRADES)}
    return {option: answer["probabilities"][option] for option in OPTIONS[question]}


def read_answer(probs: dict[str, float]) -> str:
    """The most likely option; a tie goes to the option listed first."""
    return max(probs, key=probs.__getitem__)


def truth_of(food_group: str, ultra_processed: bool, nutriscore: str) -> dict[str, str]:
    """The ground truth of a sample row, in the same labels as the answers."""
    return {
        "food_group": FOOD_GROUP_OPTION[food_group],
        "ultra_processed": "yes" if ultra_processed else "no",
        "nutriscore": nutriscore,
    }
