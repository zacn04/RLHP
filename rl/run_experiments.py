import subprocess

learning_rates = [1e-3, 5e-4, 1e-4]
reward_modes = [False, True]

for lr in learning_rates:
    for weighted in reward_modes:
        cmd = ["python", "train.py", "--lr", str(lr)]
        if weighted:
            cmd.append("--freq_weighted")
        print("Running", " ".join(cmd))
        subprocess.run(cmd)
