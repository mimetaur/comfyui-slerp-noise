"""
Slerp noise and Brownian bridge noise generators for ComfyUI.

Outputs NOISE objects compatible with SamplerCustomAdvanced.
Used for smooth latent space traversal between seed endpoints.
"""

import torch
import math


def _generate_noise_from_seed(seed, shape):
    """Generate a reproducible noise tensor from a seed.

    Uses torch.Generator for isolation — does not touch the global RNG state.
    Returns float32 on CPU, matching ComfyUI's prepare_noise convention.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator, device="cpu")


def _slerp(t, a, b):
    """Spherical linear interpolation between tensors a and b.

    Operates on flattened vectors, returns result in original shape.
    Falls back to lerp when vectors are nearly parallel (theta < 1e-6).
    """
    original_shape = a.shape
    flat_a = a.reshape(-1).double()
    flat_b = b.reshape(-1).double()

    norm_a = flat_a.norm()
    norm_b = flat_b.norm()

    # Guard against zero-norm tensors
    if norm_a < 1e-10 or norm_b < 1e-10:
        result = (1.0 - t) * a + t * b
        return result

    dot = torch.dot(flat_a, flat_b) / (norm_a * norm_b)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)

    if theta.abs() < 1e-6:
        # Nearly parallel — lerp is equivalent and numerically stable
        result = (1.0 - t) * a + t * b
    else:
        sin_theta = torch.sin(theta)
        w_a = (torch.sin((1.0 - t) * theta) / sin_theta).float()
        w_b = (torch.sin(t * theta) / sin_theta).float()
        result = w_a * a + w_b * b

    return result.reshape(original_shape)


# ---------------------------------------------------------------------------
# NOISE object wrappers
#
# ComfyUI's SamplerCustomAdvanced expects a NOISE object with:
#   - .seed (int) — used for logging/display, not generation
#   - .generate_noise(input_latent) — returns tensor matching input shape
# ---------------------------------------------------------------------------

class SlerpNoiseObject:
    """NOISE object that returns a pre-computed slerp-interpolated tensor."""

    def __init__(self, seed_a, seed_b, ratio, width, height):
        self.seed = seed_a  # Reported seed — used by ComfyUI for display
        self.seed_a = seed_a
        self.seed_b = seed_b
        self.ratio = ratio
        self.width = width
        self.height = height

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        shape = latent_image.shape  # (batch, channels, h, w)

        noise_a = _generate_noise_from_seed(self.seed_a, shape)
        noise_b = _generate_noise_from_seed(self.seed_b, shape)
        result = _slerp(self.ratio, noise_a, noise_b)

        return result.to(dtype=latent_image.dtype)


class BrownianBridgeNoiseObject:
    """NOISE object that returns noise from a Brownian bridge path between two seeds.

    The path interpolates from seed_a to seed_b via slerp, with stochastic
    perturbation that follows a Brownian bridge variance envelope:
    sigma = wander_scale * sqrt(t * (1 - t))

    This gives zero deviation at endpoints and maximum deviation at the midpoint,
    producing organic drift through latent space.
    """

    def __init__(self, seed_a, seed_b, frame_index, total_frames,
                 wander_scale, num_keyframes, perturbation_seed, width, height):
        self.seed = seed_a
        self.seed_a = seed_a
        self.seed_b = seed_b
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.wander_scale = wander_scale
        self.num_keyframes = num_keyframes
        self.perturbation_seed = perturbation_seed
        self.width = width
        self.height = height

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        shape = latent_image.shape

        # Clamp frame_index to valid range
        frame = max(0, min(self.frame_index, self.total_frames - 1))
        t = frame / max(self.total_frames - 1, 1)

        # --- Base path: slerp between endpoints ---
        noise_a = _generate_noise_from_seed(self.seed_a, shape)
        noise_b = _generate_noise_from_seed(self.seed_b, shape)
        base = _slerp(t, noise_a, noise_b)

        # --- Perturbation path ---
        # Generate deterministic sub-seeds for perturbation keyframes
        key_gen = torch.Generator(device="cpu")
        key_gen.manual_seed(self.perturbation_seed)
        keyframe_seeds = [
            torch.randint(0, 2**31, (1,), generator=key_gen).item()
            for _ in range(self.num_keyframes)
        ]

        # Find which two keyframes this frame falls between
        # Keyframes are evenly spaced: keyframe i is at position i / (num_keyframes - 1)
        keyframe_positions = [
            i / max(self.num_keyframes - 1, 1) for i in range(self.num_keyframes)
        ]

        # Find the enclosing keyframe pair
        kf_left = 0
        for i in range(self.num_keyframes - 1):
            if keyframe_positions[i + 1] >= t:
                kf_left = i
                break
        else:
            # t is at or past the last keyframe
            kf_left = self.num_keyframes - 2

        kf_right = kf_left + 1

        # Local interpolation parameter within this keyframe segment
        seg_start = keyframe_positions[kf_left]
        seg_end = keyframe_positions[kf_right]
        seg_len = seg_end - seg_start
        local_t = (t - seg_start) / seg_len if seg_len > 1e-10 else 0.0

        # Generate the two bounding keyframe noise tensors
        perturbation_left = _generate_noise_from_seed(keyframe_seeds[kf_left], shape)
        perturbation_right = _generate_noise_from_seed(keyframe_seeds[kf_right], shape)

        # Slerp between perturbation keyframes for smooth interpolation
        perturbation = _slerp(local_t, perturbation_left, perturbation_right)

        # --- Brownian bridge envelope ---
        # sigma = wander_scale * sqrt(t * (1 - t))
        # Zero at both endpoints, maximal at midpoint
        sigma = self.wander_scale * math.sqrt(t * (1.0 - t))

        # Combine: base path + scaled perturbation
        result = base + sigma * perturbation

        return result.to(dtype=latent_image.dtype)


# ---------------------------------------------------------------------------
# ComfyUI node definitions (legacy API for maximum compatibility)
# ---------------------------------------------------------------------------

class SlerpNoise:
    """Spherical linear interpolation between two noise seeds.

    Produces a NOISE object that smoothly traverses the hypersphere
    between two random noise tensors. At ratio 0.0, pure seed_a;
    at ratio 1.0, pure seed_b. The path is geodesic — constant
    perceptual velocity along the great circle.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_a": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_b": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "width": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "generate"
    CATEGORY = "sampling/custom_sampling/noise"

    def generate(self, seed_a, seed_b, ratio, width, height):
        noise_obj = SlerpNoiseObject(seed_a, seed_b, ratio, width, height)
        return (noise_obj,)


class BrownianBridgeNoise:
    """Brownian bridge noise traversal between two seed endpoints.

    Follows the geodesic slerp path from seed_a to seed_b, with
    stochastic perturbation controlled by a Brownian bridge variance
    envelope. The deviation is zero at both endpoints and maximal
    at the midpoint, producing an organic "drunk walk" through
    latent space that always arrives at the destination.

    wander_scale controls the amplitude of deviation.
    num_keyframes controls the temporal frequency of the wander —
    more keyframes means more nervous, higher-frequency drift.
    perturbation_seed makes the wander path fully deterministic.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_a": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_b": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "total_frames": ("INT", {"default": 240, "min": 2, "max": 999999}),
                "wander_scale": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_keyframes": ("INT", {"default": 10, "min": 2, "max": 50}),
                "perturbation_seed": ("INT", {"default": 12345, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "width": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "generate"
    CATEGORY = "sampling/custom_sampling/noise"

    def generate(self, seed_a, seed_b, frame_index, total_frames,
                 wander_scale, num_keyframes, perturbation_seed, width, height):
        noise_obj = BrownianBridgeNoiseObject(
            seed_a, seed_b, frame_index, total_frames,
            wander_scale, num_keyframes, perturbation_seed, width, height
        )
        return (noise_obj,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SlerpNoise": SlerpNoise,
    "BrownianBridgeNoise": BrownianBridgeNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SlerpNoise": "Slerp Noise",
    "BrownianBridgeNoise": "Brownian Bridge Noise",
}
