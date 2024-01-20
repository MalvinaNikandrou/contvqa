"""
Get VQA splits based on https://github.com/delchiaro/RATT or randomly.

By running this script you can get the question ids for the taxonomy and diverse settings.
Note that for the training data you get are split further into train and val, and the validation data are used as test.
"""

import argparse
import itertools
import json

from collections import defaultdict
from os.path import join
from pycocotools.coco import COCO
from random import shuffle
from tabulate import tabulate
from typing import Literal


# Categories for each domain
categories_transport = [
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
]
categories_animals = [
    "bird",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]
categories_sports = [
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
]
categories_food = [
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
]
categories_interior = [
    "chair",
    "couch",
    "potted plant",
    "bed",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
]


categories_group_1 = [
    "bird",
    "car",
    "keyboard",
    "motorcycle",
    "orange",
    "pizza",
    "sink",
    "sports ball",
    "toilet",
    "zebra",
]
categories_group_2 = [
    "airplane",
    "baseball glove",
    "bed",
    "bus",
    "cow",
    "donut",
    "giraffe",
    "horse",
    "mouse",
    "sheep",
]
categories_group_3 = [
    "boat",
    "broccoli",
    "hot dog",
    "kite",
    "oven",
    "sandwich",
    "snowboard",
    "surfboard",
    "tennis racket",
    "tv",
]
categories_group_4 = [
    "apple",
    "baseball bat",
    "bear",
    "bicycle",
    "cake",
    "laptop",
    "microwave",
    "potted plant",
    "remote",
    "train",
]
categories_group_5 = [
    "banana",
    "carrot",
    "cell phone",
    "chair",
    "couch",
    "elephant",
    "refrigerator",
    "skateboard",
    "toaster",
    "truck",
]

SplitType = Literal["train", "val"]
SeetingQIdsType = dict[str, dict[SplitType, list[str]]]


def get_categories_per_task(
    setting: Literal["taxonomy", "diverse"], new_random_groups: bool = False
) -> dict[str, list[str]]:
    """Get categories per task depending on the setting."""
    domain_to_categories = {
        "transport": categories_transport,
        "animals": categories_animals,
        "sports": categories_sports,
        "food": categories_food,
        "interior": categories_interior,
    }
    if setting == "taxonomy":
        tasks = domain_to_categories
    elif setting == "diverse" and not new_random_groups:
        tasks = {
            "group_1": categories_group_1,
            "group_2": categories_group_2,
            "group_3": categories_group_3,
            "group_4": categories_group_4,
            "group_5": categories_group_5,
        }
    else:
        # Reassign categories to tasks randomly for the diverse setting
        tasks = {}
        all_categories = list(
            itertools.chain.from_iterable(domain_to_categories.values())
        )
        num_tasks = len(domain_to_categories)
        num_categories_per_task = len(all_categories) // num_tasks

        shuffle(all_categories)
        for idx in range(num_tasks):
            tasks[f"group_{idx+1}"] = all_categories[
                idx * num_categories_per_task : (idx + 1) * num_categories_per_task
            ]

    return tasks


def coco_tasks_img_ids(
    coco_instances_annotation_path: str, tasks: dict[str, list[str]]
) -> dict[str, set[str]]:
    """Get the image ids for each task."""
    # Loading COCO dataset with object detection labels:
    coco = COCO(coco_instances_annotation_path)
    img_ids = {}
    for task_name, task_cat_names in tasks.items():
        task_cat_ids = coco.getCatIds(task_cat_names)

        # For each category id in current task, get all images having at least an annotation with that category
        task_img_ids = [coco.getImgIds(catIds=[category]) for category in task_cat_ids]

        # flatten and remove duplicate:
        task_img_ids = set([t for tl in task_img_ids for t in tl])

        img_ids[task_name] = task_img_ids

    # Remove all the images that are appear in more than one tasks
    commons = []
    for task_name, task_img_ids in img_ids.items():
        others = [k for k in img_ids.keys() if k != task_name]
        for other_task in others:
            commons += img_ids[other_task].intersection(task_img_ids)
    for task_name in img_ids.keys():
        img_ids[task_name] -= set(commons)
    return img_ids


def coco_split_tasks_img_ids(
    coco_path: str, tasks: dict[str, list[str]]
) -> dict[str, list[str]]:
    """You can download the COCO dataset from here: http://cocodataset.org/#download

    Specifically, download and extract "2014 Train/Val annotations"
    """
    print("\nComputing image ids for the training set")
    train_tasks_img_ids = coco_tasks_img_ids(
        join(coco_path, "instances_train2014.json"), tasks
    )
    print("\nComputing image ids for the validation set")
    val_tasks_img_ids = coco_tasks_img_ids(
        join(coco_path, "instances_val2014.json"), tasks
    )

    # Create a dictionary with the image ids for each task and split
    tasks_splits_img_ids = {}
    for task in train_tasks_img_ids.keys():
        tasks_splits_img_ids[task] = {}
        tasks_splits_img_ids[task]["train"] = train_tasks_img_ids[task]
        tasks_splits_img_ids[task]["val"] = val_tasks_img_ids[task]
    return tasks_splits_img_ids


def get_img2qids(vqa_path: str) -> SeetingQIdsType:
    """Get image to questions mapping."""
    print("Get image to questions mapping")
    question_to_img_id = {}
    for split in ["train", "val"]:
        questions_file = join(
            vqa_path, f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        )
        with open(questions_file, "r") as fp:
            quest = json.load(fp)["questions"]
        question_to_img_id.update({a["question_id"]: a["image_id"] for a in quest})

    image_to_question_ids = defaultdict(list)
    for qid, img_id in question_to_img_id.items():
        image_to_question_ids[img_id].append(str(qid))

    return image_to_question_ids


def get_splits_for_setting(
    setting: Literal["taxonomy", "diverse"], args: argparse.Namespace
) -> SeetingQIdsType:
    print(f"\nGet questions ids for {setting} domains...")
    tasks = get_categories_per_task(
        setting=setting, new_random_groups=args.new_random_groups
    )
    # assing each COCO image to a task based on the object categories
    timg_ids = coco_split_tasks_img_ids(args.coco_path, tasks)

    # aggregate all Qs per image
    img2qids = get_img2qids(args.vqa_path)

    # from img-to-task to task-to-Qs
    tasks = list(timg_ids.keys())
    task_qids = {t: {"train": [], "val": []} for t in tasks}
    for task in tasks:
        for split in ["train", "val"]:
            for img_id in timg_ids[task][split]:
                task_qids[task][split].extend(img2qids[img_id])
    return task_qids


def get_common_questions_from_settings(
    taxonomy_task_qids: SeetingQIdsType, diverse_task_qids: SeetingQIdsType
) -> tuple[SeetingQIdsType, SeetingQIdsType]:
    """Keep only the question ids that appear in both settings."""
    print("\nGet common questions from both settings")
    final_taxonomy_task_qids = {
        task: {"train": [], "val": []} for task in taxonomy_task_qids
    }
    final_diverse_task_qids = {
        task: {"train": [], "val": []} for task in diverse_task_qids
    }
    for split in ["train", "val"]:
        # Get all the question ids for each setting
        taxonomy_qids = list(
            itertools.chain.from_iterable(
                [ids[split] for ids in taxonomy_task_qids.values()]
            )
        )
        diverse_qids = list(
            itertools.chain.from_iterable(
                [ids[split] for ids in diverse_task_qids.values()]
            )
        )

        # Keep the question ids that appear in both settings
        for task in taxonomy_task_qids.keys():
            split_qids = set(taxonomy_task_qids[task][split])
            final_qids = list(split_qids.intersection(diverse_qids))
            final_taxonomy_task_qids[task][split] = final_qids

        for task in diverse_task_qids.keys():
            split_qids = set(diverse_task_qids[task][split])
            final_qids = list(split_qids.intersection(taxonomy_qids))
            final_diverse_task_qids[task][split] = final_qids
    return final_taxonomy_task_qids, final_diverse_task_qids


def print_statistics(setting_task_qids: SeetingQIdsType) -> None:
    """Print statistics for the taxonomy and diverse settings."""
    tasks = list(setting_task_qids.keys())
    tasks.sort()
    results = [["Task", "Train+Validation", "Test"]]
    for task in tasks:
        train_val = len(setting_task_qids[task]["train"])
        test = len(setting_task_qids[task]["val"])
        results.append([task, train_val, test])
    # Print the table
    print(tabulate(results, headers="firstrow", tablefmt='pretty'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", default="vqa_v2_data/")
    parser.add_argument("--vqa_path", default="vqa_v2_data/")
    parser.add_argument(
        "--new_random_groups",
        action="store_true",
        help="Split objects into 5 tasks randomly",
    )

    args = parser.parse_args()
    taxonomy_task_qids = get_splits_for_setting(setting="taxonomy", args=args)
    diverse_task_qids = get_splits_for_setting(setting="diverse", args=args)
    taxonomy_task_qids, diverse_task_qids = get_common_questions_from_settings(
        taxonomy_task_qids, diverse_task_qids
    )
    print("\033[1mDiverse Setting\033[0m")
    print_statistics(diverse_task_qids) 
    print("\033[1mTaxonomy Setting\033[0m")
    print_statistics(taxonomy_task_qids)
