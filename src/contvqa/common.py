import argparse
import json
import os

from collections import defaultdict, Counter
from functools import lru_cache
from typing import Literal
from tabulate import tabulate
from itertools import chain
from contvqa.answer_preprocessing import prep_ans


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


@lru_cache(maxsize=1)
def get_vqa_v2_mc_answers(vqa_dir: str) -> dict[str, str]:
    """Get the multiple choice answer for each question id."""
    vqa_answers = []
    for split in ["train", "val"]:
        vqa_question_file = os.path.join(
            vqa_dir, f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        )
        vqa_answers_file = os.path.join(
            vqa_dir, f"v2_mscoco_{split}2014_annotations.json"
        )
        vqa_answers.extend(read_json(vqa_answers_file)["annotations"])

    mc_answers = {
        str(x["question_id"]): prep_ans(x["multiple_choice_answer"])
        for x in vqa_answers
    }
    return mc_answers


def get_answer_set_for_question(answers):
    """Count an answer if it appears in the question only once."""
    return set(
        [
            prep_ans(a["answer"])
            for a in answers
            if a["answer"].lower() not in {"none", "maybe"}
        ]
    )


@lru_cache(maxsize=1)
def get_vqa_v2_answer_sets(vqa_dir: str) -> dict[str, str]:
    """Get the multiple choice answer for each question id."""
    vqa_answers = []
    for split in ["train", "val"]:
        vqa_question_file = os.path.join(
            vqa_dir, f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        )
        vqa_answers_file = os.path.join(
            vqa_dir, f"v2_mscoco_{split}2014_annotations.json"
        )
        vqa_answers.extend(read_json(vqa_answers_file)["annotations"])

    answers = {
        str(x["question_id"]): get_answer_set_for_question(x["answers"])
        for x in vqa_answers
    }
    return answers
