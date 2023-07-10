import logging

import pytest
import torch

from no_sharing.model import Mapformer


def test_model():
    torch.manual_seed(42)

    model = Mapformer()
    logging.info("Model:\n%s", model)
    logging.info(
        "Parameters:  total: %.0fM, attn: %.0fM, mlp: %.0fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for name, p in model.named_parameters() if "attn" in name) / 1e6,
        sum(p.numel() for name, p in model.named_parameters() if "mlp" in name) / 1e6,
    )
    logging.info(
        "Buffers:  total: %.0fM",
        sum(p.numel() for p in model.buffers()) / 1e6,
    )

    img_size = model.patch_embed.img_size[0]
    input = torch.randn(8, 3, img_size, img_size)

    with torch.no_grad():
        output = model(input)
    loss = model.loss()

    logging.info("Output shape: %s", output.shape)
    logging.info("Loss: %.6g", loss.item())

    assert output.shape == (8, 576, 256)


if __name__ == "__main__":
    pytest.main([__file__])
