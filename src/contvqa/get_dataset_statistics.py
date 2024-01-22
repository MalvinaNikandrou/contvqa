import argparse
import json
import os

from collections import defaultdict, Counter
from functools import lru_cache
from typing import Literal, Dict
from tabulate import tabulate

from common import read_json, get_vqa_v2_mc_answers, get_vqa_v2_answer_sets
from itertools import chain


def get_num_classes_per_task(
    contvqa_dir: str = "data",
    vqa_dir: str = "data/vqa_v2_data",
    threshold: int = 8,
    apply_global_threshold: bool = False,
):
    """Get the number of classes per task for a given setting.

    Note that the selection of classes for training is a design choice,
    but the model performance should be computed against all ground truth answers.
    """
    # Get the VQA-v2 answers for each question id
    mc_answers = get_vqa_v2_mc_answers(vqa_dir)
    counter = Counter(mc_answers.values())

    # Get the question ids for each task
    task_question_ids = defaultdict(list)
    for split in ["train", "valid", "test"]:
        contvqa_data = read_json(
            os.path.join(contvqa_dir, f"{split}_question_ids.json")
        )
        for task, question_ids in contvqa_data.items():
            task_question_ids[task] = task_question_ids.get(task, []) + question_ids

    # Get the number of classes per task
    task_num_classes = {}
    for task, task_ids in task_question_ids.items():
        # Answers that appear in the task
        answers = [mc_answers[str(question_id)] for question_id in task_ids]
        # If we apply a global threshold, we only keep the answers that appear more than the threshold in the whole dataset
        # Otherwise, we apply the threshold per task
        if not apply_global_threshold:
            counter = Counter(answers)
        classes = list(set([ans for ans in answers if counter[ans] > threshold]))
        task_num_classes[task] = len(classes)
    return task_num_classes


def get_num_of_samples_per_task(contvqa_dir: str) -> Dict[str, Dict[str, int]]:
    """Get the number of samples per task for a given setting."""
    with open(os.path.join(contvqa_dir, "train_question_ids.json"), "r") as f:
        train_ids = json.load(f)
    with open(os.path.join(contvqa_dir, "valid_question_ids.json"), "r") as f:
        valid_ids = json.load(f)
    with open(os.path.join(contvqa_dir, "test_question_ids.json"), "r") as f:
        test_ids = json.load(f)
    samples_per_task = {}
    for task in train_ids.keys():
        samples_per_task[task] = {
            "train": len(train_ids[task]),
            "valid": len(valid_ids[task]),
            "test": len(test_ids[task]),
        }
    return samples_per_task


def print_statistics(
    setting: Literal["diverse_domains", "taxonomy_domains", "question_types"],
    args: argparse.Namespace,
    apply_global_threshold: bool = False,
) -> None:
    """Print statistics for the each setting settings."""
    contvqa_data_dir = os.path.join(args.contvqa_path, setting)
    num_classes = get_num_classes_per_task(
        contvqa_dir=contvqa_data_dir,
        vqa_dir=args.vqa_path,
        threshold=args.answers_threshold,
        apply_global_threshold=apply_global_threshold,
    )
    samples_per_task = get_num_of_samples_per_task(contvqa_data_dir)

    tasks = list(num_classes.keys())
    tasks.sort()
    results = [["Task", "Train", "Validation", "Test", "Num Classes"]]
    for task in tasks:
        results.append(
            [
                task.capitalize(),
                samples_per_task[task]["train"],
                samples_per_task[task]["valid"],
                samples_per_task[task]["test"],
                num_classes[task],
            ]
        )
    # Print the table
    print(tabulate(results, headers="firstrow", tablefmt="pretty"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contvqa_path", default="data/")
    parser.add_argument("--vqa_path", default="data/vqa_v2_data/")
    parser.add_argument(
        "--answers_threshold",
        type=int,
        default=8,
        help="Threshold for the number of samples per class",
    )

    args = parser.parse_args()
    print("\033[1mDiverse Domains Setting\033[0m")
    print_statistics("diverse_domains", args, True)
    print("\033[1mTaxonomy Domains Setting\033[0m")
    print_statistics("taxonomy_domains", args, True)
    print("\033[1mQuestion Types Setting\033[0m")
    print_statistics("question_types", args, False)
