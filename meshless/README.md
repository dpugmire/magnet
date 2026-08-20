# Meshless rendering

This directory contains autoencoder-conditioned implicit-field experiments for
meshless rendering and contour extraction. The learned field can be evaluated
at arbitrary coordinates, while a quadtree focuses evaluations near requested
isovalues.

`meshless_render.py` is the primary prototype. `meshless_render_topo.py` adds
topology-oriented evaluation and regularization,
`meshless_render_contours_topology.py` combines quadtree contours with
persistence comparisons, and `pd_example.py` is a small persistence-diagram
example.
