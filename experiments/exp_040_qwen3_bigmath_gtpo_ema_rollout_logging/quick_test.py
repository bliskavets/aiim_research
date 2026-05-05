"""quick_test.py — run 2 steps of exp_040 to verify rollout logging works."""
import train
train.TRAINING_CONFIG["max_steps"] = 2
train.TRAINING_CONFIG["save_steps"] = 9999
train.main()
