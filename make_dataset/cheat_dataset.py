import json
import os.path as p
from datasets import load_from_disk, concatenate_datasets

from tools import get_args


def check_is_real_cheating(mrc_test_dataset, all_mrc_dummy_dataset, cheat_id):
    q_idx = mrc_test_dataset["validation"]["id"].index(cheat_id)
    c_idx = all_mrc_dummy_dataset["id"].index(cheat_id)
    print("질문: ", mrc_test_dataset["validation"]["question"][q_idx], end="\n\n")
    print(all_mrc_dummy_dataset["context"][c_idx], end="\n\n")
    print(all_mrc_dummy_dataset["answers"][c_idx], end="\n\n")

    return all_mrc_dummy_dataset["answers"][c_idx]["text"][0]


def main(args):
    mrc_test_dataset = load_from_disk(p.join(args.path.train_data_dir, "test_dataset"))
    mrc_dummy_dataset = load_from_disk(p.join(args.path.train_data_dir, "dummy_dataset"))

    all_mrc_dummy_dataset = concatenate_datasets(
        [mrc_dummy_dataset["train"].flatten_indices(), mrc_dummy_dataset["validation"].flatten_indices()]
    )

    cheat_ids = list(set(mrc_test_dataset["validation"]["id"]).intersection(set(all_mrc_dummy_dataset["id"])))

    cheats = {}

    for cheat_id in cheat_ids:  # ex) cheat_id: 'mrc-1-000711'
        temp = check_is_real_cheating(mrc_test_dataset, all_mrc_dummy_dataset, cheat_id)
        cheats[cheat_id] = temp

    cheat_path = p.join(args.path.train_data_dir, "cheat.json")

    print(cheats)

    with open(cheat_path, "w") as f:
        f.write(json.dumps(cheats, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
