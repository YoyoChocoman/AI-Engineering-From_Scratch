import numpy as np

def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

def dpo_loss(
    policy_logprob_preferred,
    policy_logprob_rejected,
    ref_logprob_preferred,
    ref_logprob_rejected,
    beta=0.1
):
    # policy = the current training ai
    # ref = the frozen past SFT ai
    # beta used of restricting the change
    preferred_ratio = policy_logprob_preferred - ref_logprob_preferred
    rejected_ratio = policy_logprob_rejected - ref_logprob_rejected

    logit = beta * (preferred_ratio - rejected_ratio)
    loss = -np.log(sigmoid(logit) + 1e-8)

    return loss, logit


# Test
# Original log-prob
ref_pref = -2.0
ref_rej = -2.0

# Situation A: better
policy_pref_A = -1.0
policy_rej_A = -3.0

loss_A, logit_A = dpo_loss(policy_pref_A, policy_rej_A, ref_pref, ref_rej, beta=0.1)
print(f"Situation A -> Loss: {loss_A:.4f} (Logit: {logit_A:.2f})")

# Situation B：worse
policy_pref_B = -3.0
policy_rej_B = -1.0

loss_B, logit_B = dpo_loss(policy_pref_B, policy_rej_B, ref_pref, ref_rej, beta=0.1)
print(f"Situation B -> Loss: {loss_B:.4f} (Logit: {logit_B:.2f})")