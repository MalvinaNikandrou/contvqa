import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import spacy

from collections import Counter, defaultdict
from itertools import chain
from spacy.lang.en import English
from typing import Dict, Set

from common import get_answer_set_for_question


cls = spacy.util.get_lang_class("en")
cls.Defaults.stop_words.remove("many")
cls.Defaults.stop_words.remove("see")
cls.Defaults.stop_words.remove("made")
nlp = English()


def get_question_ids_to_answers(vqa_dir: str) -> Dict[str, Set[str]]:
    with open(os.path.join(vqa_dir, "v2_mscoco_train2014_annotations.json"), "r") as fp:
        annotations = json.load(fp)["annotations"]
    answers = {
        str(ann["question_id"]): get_answer_set_for_question(ann["answers"])
        for ann in annotations
    }
    return answers


def get_question_ids_to_questions(vqa_dir: str) -> Dict[str, str]:
    with open(
        os.path.join(vqa_dir, "v2_OpenEnded_mscoco_train2014_questions.json"), "r"
    ) as fp:
        questions = json.load(fp)["questions"]

    questions = {str(q["question_id"]): q["question"] for q in questions}
    return questions


def plot_hist(
    counter: Counter,
    name: str,
    task: str,
    output_dir: str,
    color: str,
    width: float = 0.8,
    topk: int = 20,
):
    """Creates a histogram of the topk most common words in the counter."""
    indexes = np.arange(topk)

    labels = []
    values = []
    for label, value in counter.most_common(topk):
        labels.append(label)
        values.append(value)

    plt.figure(figsize=(12, 4))
    plt.rc("axes", axisbelow=True)
    plt.bar(indexes, values, width, color=color)
    plt.xticks(indexes, labels, rotation=35)
    plt.title(f"{topk} most common words in {task.replace('_', ' ').capitalize()}")
    plt.grid(axis="y")
    plt.savefig(
        os.path.join(output_dir, f"top{topk}_{name}_{task}.png"),
        bbox_inches="tight",
    )
    plt.close()


def tokenize(text):
    """Tokenizes the questions."""
    doc = nlp(text)
    return [
        token.text.lower()
        for token in doc
        if (not token.is_stop and not token.is_punct)
    ]


def plot_histograms_for_setting(setting, data_dir, vqa_dir, output_dir):
    answers = get_question_ids_to_answers(vqa_dir)
    questions = get_question_ids_to_questions(vqa_dir)
    task_question_ids = defaultdict(list)
    for split in ["train", "valid"]:
        with open(
            os.path.join(data_dir, setting, f"{split}_question_ids.json"), "r"
        ) as fp:
            contvqa_data = json.load(fp)
        for task, question_ids in contvqa_data.items():
            task_question_ids[task] = task_question_ids.get(task, []) + question_ids

    for task, question_ids in task_question_ids.items():
        vocab = list(
            chain.from_iterable([tokenize(questions[qid]) for qid in question_ids])
        )
        plot_hist(
            counter=Counter(vocab),
            name="vocab",
            task=task,
            output_dir=output_dir,
            color="lightsteelblue",
        )

        task_answers = list(
            chain.from_iterable([answers[str(qid)] for qid in question_ids])
        )
        plot_hist(
            counter=Counter(task_answers),
            name="ans",
            task=task,
            output_dir=output_dir,
            color="gray",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contvqa_path", default="data/")
    parser.add_argument("--vqa_path", default="data/vqa_v2_data")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="plots", help="Path to save resulting plots"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for setting in ["diverse_domains", "taxonomy_domains", "question_types"]:
        setting_output_dir = os.path.join(output_dir, setting)
        os.makedirs(setting_output_dir, exist_ok=True)
        plot_histograms_for_setting(
            setting=setting,
            data_dir=args.contvqa_path,
            vqa_dir=args.vqa_path,
            output_dir=setting_output_dir,
        )
