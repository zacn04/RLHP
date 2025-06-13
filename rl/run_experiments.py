import subprocess
import pandas as pd

learning_rates = [1e-3, 5e-4, 1e-4]
reward_modes = [False, True]

for lr in learning_rates:
    for weighted in reward_modes:
        cmd = ["python", "train.py", "--lr", str(lr), "--timesteps 500_000", "--eval_freq 10_000"]
        if weighted:
            cmd.append("--freq_weighted")
        print("Running", " ".join(cmd))
        subprocess.run(cmd)
        results = []
for lr in learning_rates:
    for weighted in reward_modes:
        log_file = f"logs/training_lr{lr}_weighted{weighted}.csv"
        df = pd.read_csv(log_file)
        final_reward = df['reward'].iloc[-1]
        results.append({
            'learning_rate': lr,
            'weighted': weighted,
            'final_reward': final_reward
        })

        comparison_df = pd.DataFrame(results)
        print("\nResults Comparison:")
        print(comparison_df)