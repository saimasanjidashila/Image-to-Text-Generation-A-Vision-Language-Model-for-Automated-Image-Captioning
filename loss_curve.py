import re
import matplotlib.pyplot as plt

# Read the training log file
with open("training_log.txt", "r") as f:
    log_lines = f.readlines()

# Extract epoch and loss from log
epoch_loss_pairs = []
for line in log_lines:
    match = re.search(r"'loss': ([\d.]+), .*'epoch': ([\d.]+)", line)
    if match:
        loss = float(match.group(1))
        epoch = float(match.group(2))
        if epoch >= 1:  # Filter for epoch >= 1
            epoch_loss_pairs.append((epoch, loss))

# Sort by epoch
epoch_loss_pairs.sort()

# Unpack
epochs, losses = zip(*epoch_loss_pairs)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Training Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch (Epochs 1 to 5)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
