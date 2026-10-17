"""Import shim for facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f ``main_moco.py`` (Spec A, B1).

That commit's ``main_moco.py`` imports ``deeplearning.cross_image_ssl.moco.builder`` / ``.loader``, which exist only
in Meta's internal tree. With this directory and the clone root on ``sys.path`` (``scripts/parity`` does both), the
names resolve to the clone's unmodified ``moco/builder.py`` and ``moco/loader.py``. No reference source lives here.
"""
