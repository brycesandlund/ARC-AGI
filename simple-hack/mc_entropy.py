import pdb

# Simulation: Model logits becoming deterministic, entropy and log-prob plot
import numpy as np
import matplotlib.pyplot as plt


def softmax(logits):
    e_logits = np.exp(logits - np.max(logits))
    return e_logits / e_logits.sum(axis=-1, keepdims=True)


def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-12))


def simulate(num_classes=1000, steps=200):
    entropies = []
    sum_log_probs = []
    for step in range(steps):
        # Make logits more deterministic over time
        scale = 1 + step * 0.1
        logits = np.random.randn(num_classes)
        logits[0] += scale  # Make class 0 more likely
        logits[1] += scale * 0.5
        probs = softmax(logits)
        ent = entropy(probs)
        # pdb.set_trace()
        # random_probs = np.random.choice(probs, size=5, replace=5)
        # estimated_entropy = entropy(random_probs)
        # log_prob = -np.log(np.random.choice(probs, size=5, replace=False)).mean()
        
        # Corrected unbiased Monte Carlo estimator for entropy
        indices = np.random.choice(len(probs), size=500, replace=True, p=probs)
        log_prob = -np.log(probs[indices] + 1e-12).mean()

        sum_log_probs.append(log_prob)

        entropies.append(ent)
    return entropies, sum_log_probs


if __name__ == "__main__":
    entropies, sum_log_probs = simulate()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(entropies, label="Entropy")
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.title("Entropy over steps")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(sum_log_probs, label="Log Prob (class 0)")
    plt.xlabel("Step")
    plt.ylabel("Log Probability")
    plt.title("Log Probability over steps")
    plt.legend()
    plt.tight_layout()
    plt.show()