"""Stage 2: context-distillation fine-tuning data. gpt-oss-20b answers constitution questions, MATH and
KodCode problems with the constitution in its developer message; the kept answers are trained on without it.
"""

STAGE = "distillation"
"""The results folder of this stage: `results/distillation/scripts/<script>/`."""
