import unittest

import torch

from samix import build_samix_lite


class SAMIXLiteSmokeTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = build_samix_lite(topk=2)
        batch = {
            "support_images": torch.randn(2, 4, 3, 128, 128),
            "support_masks": torch.randint(0, 2, (2, 4, 1, 128, 128)).float(),
            "query_images": torch.randn(2, 3, 128, 128),
        }

        out = model(**batch)
        self.assertEqual(out["logits"].shape, (2, 1, 128, 128))
        self.assertEqual(out["retrieval_indices"].shape, (2, 2))
        self.assertEqual(out["retrieval_scores"].shape, (2, 2))
        self.assertEqual(out["retrieved_prototypes"].shape[:3], (2, 2, 256))

    def test_backward(self) -> None:
        model = build_samix_lite(topk=1)
        batch = {
            "support_images": torch.randn(1, 2, 3, 96, 96),
            "support_masks": torch.randint(0, 2, (1, 2, 1, 96, 96)).float(),
            "query_images": torch.randn(1, 3, 96, 96),
        }
        logits = model(**batch)["logits"]
        loss = logits.mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))


if __name__ == "__main__":
    unittest.main()
