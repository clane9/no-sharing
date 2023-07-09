# No sharing

> Work in progress

A brain-inspired topographic vision model with learned contrastive weight sharing.

Popular vision models like convnets and transformers use weight sharing to conserve parameters and add inductive bias. However, strict weight sharing is biologically implausible. Here, we instead aim to *learn* to share weights by promoting smooth representation maps over a grid of feature columns.

As a complement, we also penalize the wiring cost for long-range connections within the grid.

Finally, we evaluate the importance of long-range backpropagation by comparing the performance of the learned representations under end-to-end vs layerwise learning (using stop-grad operations between network blocks).

## Inspiration

Geoff Hinton. [The Robot Brains Season 2 Episode 22.](https://www.therobotbrains.ai/geoff-hinton-transcript-part-one) (2022).

Doshi, Fenil R., and Talia Konkle. [Cortical topographic motifs emerge in a self-organized map of object space.](https://doi.org/10.1126/sciadv.ade8187) Science Advances (2023).

Margalit, Eshed, et al. [A Unifying Principle for the Functional Organization of Visual Cortex.](https://www.biorxiv.org/content/10.1101/2023.05.18.541361v1) bioRxiv (2023).

Xiong, Yuwen, Mengye Ren, and Raquel Urtasun. [Loco: Local contrastive representation learning.](https://proceedings.neurips.cc/paper/2020/hash/7fa215c9efebb3811a7ef58409907899-Abstract.html) NeurIPS (2020).
