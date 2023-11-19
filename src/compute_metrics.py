import json
import argparse

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Model-agnostic tool for explaining link predictions"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["necessary", "sufficient"],
        help="The mode for which to plot the explanation lengths: necessary or sufficient",
    )

    parser.add_argument(
        "--explanations_path",
        type=str,
        help="Path of the explanations to evaluate.",
    )

    return parser.parse_args()


def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return round(count / float(len(ranks)), 3)


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0 / float(rank)
    return round(reciprocal_rank_sum / float(len(ranks)), 3)


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return round(rank_sum / float(len(ranks)), 3)


def main(args):
    mode = args.mode
    explanations_path = Path(args.explanations_path)

    explanations_filepath = explanations_path / "output_end_to_end.json"

    with open(explanations_filepath, "r") as input_file:
        triple_to_details = json.load(input_file)
    if mode == "necessary":
        ranks = [float(details["rank"]) for details in triple_to_details]
        new_ranks = [float(details["new_rank"]) for details in triple_to_details]
    else:
        ranks = [
            float(conversion["rank"])
            for details in triple_to_details
            for conversion in details["conversions"]
        ]
        new_ranks = [
            float(conversion["new_rank"])
            for details in triple_to_details
            for conversion in details["conversions"]
        ]

    original_mrr, original_h1 = mrr(ranks), hits_at_k(ranks, 1)
    new_mrr, new_h1 = mrr(new_ranks), hits_at_k(new_ranks, 1)
    mrr_delta = round(new_mrr - original_mrr, 3)
    h1_delta = round(new_h1 - original_h1, 3)

    explanations_filepath = explanations_path / "output.json"
    with open(explanations_filepath, "r") as input_file:
        explanations = json.load(input_file)
    rels = [x["#relevances"] for x in explanations]
    rels = sum(rels)

    print(f"rels: {rels}")
    print(f"H@1 delta: {h1_delta}")
    print(f"MRR delta: {mrr_delta}")

if __name__ == "__main__":
    main(parse_args())
