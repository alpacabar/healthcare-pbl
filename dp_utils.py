import tensorflow_privacy as tfp

def dp_optimizer(learning_rate=0.01, noise_multiplier=0.5, l2_norm_clip=1.0):
    return tfp.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        learning_rate=learning_rate
    )
