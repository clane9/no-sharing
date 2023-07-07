# No sharing

> Work in progress

A brain-inspired topographic vision model with learned contrastive weight sharing.

Popular vision models like convnets and transformers use weight sharing to conserve parameters and add inductive bias. However, strict weight sharing is biologically implausible. Here, we instead aim to *learn* to share weights by promoting smooth representation maps over a grid of feature columns.

As a complement, we also penalize the wiring cost for long-range connections within the grid.

Finally, we evaluate the importance of long-range backpropagation by comparing the performance of the learned representations under end-to-end vs layerwise learning (using stop-grad operations between network blocks).
