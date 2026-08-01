# Experimental protocol text ready for Word review

All incremental variants are trained under one fixed protocol at an input size
of 352 by 352 pixels. The encoder is frozen for the first 10 epochs and is then
optimized jointly with the decoder. AdamW uses learning rates of 1e-5 for the
encoder and 1e-4 for the decoder and added modules, with weight decay 1e-4.
Checkpoint selection uses the equal mean of validation Dice on Kvasir-SEG and
CVC-ClinicDB. External test datasets are not used for checkpoint, epoch,
threshold, or configuration selection. The independent seeds are 42, 3407, and
2026, and test-time augmentation is disabled for the ablation comparison.

This text contains protocol facts only. No performance claim is ready while the
official experiment matrix remains incomplete.
