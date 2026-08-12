import numpy as np
from hmmlearn import hmm

print("hmmlearn Example — Gaussian Hidden Markov Model\n")

# Reproducible results
np.random.seed(42)

# --- 1. Define a 2-state Gaussian HMM ---
print("1. Creating a Gaussian HMM with 2 hidden states")

model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)

# Initial state probabilities
model.startprob_ = np.array([0.7, 0.3])

# Transition matrix: state 0 is "low", state 1 is "high"
model.transmat_ = np.array([
    [0.8, 0.2],
    [0.3, 0.7],
])

# Emission means (1-D observations)
model.means_ = np.array([[1.0], [5.0]])

# Emission covariances
model.covars_ = np.array([[0.5], [0.5]])

print("   Start probabilities :", model.startprob_)
print("   Transition matrix   :\n", model.transmat_)
print("   Emission means      :", model.means_.flatten())

# --- 2. Sample a sequence from the model ---
print("\n2. Sampling 20 observations from the model")
observations, true_states = model.sample(20)
print(f"   Observations (first 10): {observations[:10].flatten().round(2)}")
print(f"   True states  (first 10): {true_states[:10]}")

# --- 3. Decode hidden states using Viterbi ---
print("\n3. Decoding most likely hidden-state sequence (Viterbi)")
log_prob, decoded_states = model.decode(observations, algorithm="viterbi")
print(f"   Log-probability       : {log_prob:.4f}")
print(f"   Decoded states (first 10): {decoded_states[:10]}")

accuracy = np.mean(decoded_states == true_states)
print(f"   State recovery accuracy: {accuracy:.0%}")

# --- 4. Re-fit a new model on the sampled data ---
print("\n4. Fitting a new GaussianHMM on the sampled sequence")
fitted = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
fitted.fit(observations)
score = fitted.score(observations)
print(f"   Log-likelihood of fitted model: {score:.4f}")

print("\nhmmlearn example completed successfully!")
