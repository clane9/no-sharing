import logging
import math

import pytest
import torch

from no_sharing.model import Mapformer


def test_model():
    torch.manual_seed(42)

    model = Mapformer()
    logging.info("Model:\n%s", model)
    logging.info(
        "Parameters:  attn: %.0fM, mlp: %.0fM, total: %.0fM",
        sum(p.numel() for name, p in model.named_parameters() if "attn" in name) / 1e6,
        sum(p.numel() for name, p in model.named_parameters() if "mlp" in name) / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    logging.info(
        "Buffers:  total: %.0fM",
        sum(p.numel() for p in model.buffers()) / 1e6,
    )

    img_size = model.patch_embed.img_size[0]
    input = torch.randn(8, 3, img_size, img_size)

    with torch.no_grad():
        output, activations, attention_maps = model(input, return_intermediates=True)
    loss = model.loss()

    logging.info("Activations: %.0fM", sum(x[0].numel() for x in activations) / 1e6)
    logging.info(
        "Interconnections: %.0fM", sum(x[0].numel() for x in attention_maps) / 1e6
    )
    logging.info("Output shape: %s", output.shape)
    logging.info("Output[0, :5]: %s", output[0, :5])
    logging.info("Loss: %.6g", loss.item())

    assert output.shape == (8, 1000)
    assert math.isclose(loss.item(), 16.9966, rel_tol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
