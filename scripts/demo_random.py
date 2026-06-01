import torch

from samix import build_samix_lite


def main() -> None:
    model = build_samix_lite(topk=2)
    batch = {
        "support_images": torch.randn(2, 3, 3, 256, 256),
        "support_masks": torch.randint(0, 2, (2, 3, 1, 256, 256)).float(),
        "query_images": torch.randn(2, 3, 256, 256),
    }

    with torch.no_grad():
        output = model(**batch)

    print("logits:", tuple(output["logits"].shape))
    print("retrieval_indices:", tuple(output["retrieval_indices"].shape))
    print("retrieval_scores:", tuple(output["retrieval_scores"].shape))


if __name__ == "__main__":
    main()
