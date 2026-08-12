"""
Sanity check script for BrawlSniperEnv.
Validates Gymnasium API compliance and runs a quick benchmark.
"""
import time
import numpy as np
from gymnasium.utils.env_checker import check_env
from env.brawl_sniper_env import BrawlSniperEnv


def main():
    env = BrawlSniperEnv()

    # --- 1. Gymnasium API Validation ---
    print("Running Gymnasium API validation...")
    check_env(env)
    print("✅ Environment passes Gymnasium API validation!\n")

    # --- 2. Quick Functional Test ---
    print("Running functional test (1,000 random steps)...")
    obs, info = env.reset(seed=42)
    assert obs.shape == (30,), f"Obs shape mismatch: {obs.shape}"
    assert obs.dtype == np.float32, f"Obs dtype mismatch: {obs.dtype}"

    total_reward = 0.0
    episodes = 0
    kills = 0

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        assert obs.shape == (30,), f"Step {step}: obs shape {obs.shape}"
        assert not np.any(np.isnan(obs)), f"Step {step}: NaN in observation!"
        assert not np.any(np.isinf(obs)), f"Step {step}: Inf in observation!"

        if terminated or truncated:
            if terminated:
                kills += 1
            episodes += 1
            obs, info = env.reset()

    print(f"   Episodes completed: {episodes}")
    print(f"   Kills (terminated): {kills}")
    print(f"   Total reward: {total_reward:.2f}")
    print("✅ Functional test passed!\n")

    # --- 3. Performance Benchmark ---
    print("Running performance benchmark (100,000 steps)...")
    obs, _ = env.reset(seed=0)
    start = time.perf_counter()

    for _ in range(100_000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    elapsed = time.perf_counter() - start
    steps_per_sec = 100_000 / elapsed

    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {steps_per_sec:,.0f} steps/sec")
    print(f"   Per-step latency: {elapsed / 100_000 * 1e6:.1f} µs")

    if steps_per_sec > 5000:
        print("✅ Performance is EXCELLENT for single-env training.")
    elif steps_per_sec > 1000:
        print("⚠️  Performance is acceptable but could be optimized.")
    else:
        print("❌ Performance is too slow — investigate bottlenecks.")

    # --- 4. Observation Range Check ---
    print("\nRunning observation range check (10,000 steps)...")
    obs, _ = env.reset(seed=123)
    obs_min = np.full(30, np.inf)
    obs_max = np.full(30, -np.inf)

    for _ in range(10_000):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        obs_min = np.minimum(obs_min, obs)
        obs_max = np.maximum(obs_max, obs)
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"   Obs min range: [{obs_min.min():.3f}, {obs_min.max():.3f}]")
    print(f"   Obs max range: [{obs_max.min():.3f}, {obs_max.max():.3f}]")

    out_of_bounds = np.any(obs_min < -2.0) or np.any(obs_max > 2.0)
    if out_of_bounds:
        print("❌ Some observations exceed the declared [-2.0, 2.0] bounds!")
        for i in range(30):
            if obs_min[i] < -2.0 or obs_max[i] > 2.0:
                print(f"   obs[{i}]: min={obs_min[i]:.3f}, max={obs_max[i]:.3f}")
    else:
        print("✅ All observations within declared bounds.\n")

    print("=" * 50)
    print("ALL CHECKS PASSED — Environment is ready for training.")
    print("=" * 50)


if __name__ == "__main__":
    main()
