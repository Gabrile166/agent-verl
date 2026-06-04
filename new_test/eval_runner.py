"""Eval-only runner for ALFWorld/SciWorld with an OpenAI-compatible API model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.environments import make_envs  # noqa: E402
from agent_system.multi_turn_rollout import TrajectoryCollector  # noqa: E402
from agent_system.reward_manager.episode import EpisodeRewardManager  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.trainer.main_ppo import create_rl_dataset  # noqa: E402
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402

from api_actor import APIActorWrapper  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run API-backed agent evaluation.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("eval_config.yaml")),
        help="Path to eval_config.yaml.",
    )
    return parser.parse_known_args()


def load_config(config_path: str, overrides: list[str]):
    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(config, cli_config)


def build_dummy_gen_batch(batch_size: int, tokenizer, config) -> DataProto:
    max_prompt_length = int(config.data.max_prompt_length)
    pad_token_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((batch_size, max_prompt_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_prompt_length), dtype=torch.long)
    position_ids = torch.zeros((batch_size, max_prompt_length), dtype=torch.long)
    dummy_chat = [{"role": "user", "content": "Placeholder"}]

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors={
            "raw_prompt": np.array([dummy_chat for _ in range(batch_size)], dtype=object),
            "data_source": np.array([str(config.env.env_name)] * batch_size, dtype=object),
            "raw_prompt_ids": np.array([[pad_token_id] for _ in range(batch_size)], dtype=object),
        },
        meta_info=build_meta_info(tokenizer),
    )


def build_meta_info(tokenizer) -> dict[str, Any]:
    return {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": False,
        "validate": True,
    }


def build_gen_batch(config, tokenizer) -> DataProto:
    val_files = config.data.get("val_files")
    if val_files in (None, "null", ""):
        return build_dummy_gen_batch(int(config.data.val_batch_size), tokenizer, config)

    dataset = create_rl_dataset(val_files, config.data, tokenizer, processor=None)
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.data.val_batch_size),
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    try:
        test_data = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("Validation dataloader is empty.") from exc

    test_batch = DataProto.from_single_dict(test_data)
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
    for optional_key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs"):
        if optional_key in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append(optional_key)
    gen_batch = test_batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )
    gen_batch.meta_info = build_meta_info(tokenizer)
    return gen_batch


def init_ray(config) -> None:
    import ray

    if ray.is_initialized():
        return
    group_n = int(config.env.rollout.n) if int(config.env.rollout.n) > 0 else 1
    worker_cpus = float(config.env.resources_per_worker.get("num_cpus", 1))
    ray_num_cpus = int((int(config.data.train_batch_size) * group_n + int(config.data.val_batch_size)) * worker_cpus) + 2
    ray.init(num_cpus=max(ray_num_cpus, 2), include_dashboard=False, ignore_reinit_error=True)


def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.path,
        trust_remote_code=bool(config.model.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for eval response masking.")
    return tokenizer


def run_evaluation(config, tokenizer, api_actor, traj_collector, val_envs, reward_fn) -> dict[str, float]:
    gen_batch = build_gen_batch(config, tokenizer)
    if len(gen_batch) != int(config.data.val_batch_size):
        raise ValueError(f"gen_batch size {len(gen_batch)} must equal data.val_batch_size {config.data.val_batch_size}")

    test_batch = traj_collector.multi_turn_loop(
        gen_batch=gen_batch,
        actor_rollout_wg=api_actor,
        envs=val_envs,
        is_train=False,
    )
    result = reward_fn(test_batch, return_dict=True)
    return collect_metrics(test_batch, result["reward_tensor"])


def collect_metrics(test_batch: DataProto, reward_tensor: torch.Tensor) -> dict[str, float]:
    reward_tensor_1d = reward_tensor.sum(-1).cpu()
    data_sources = np.asarray(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor_1d.shape[0]))
    tool_callings = np.asarray(test_batch.non_tensor_batch["tool_callings"])
    traj_uids = np.asarray(test_batch.non_tensor_batch["traj_uid"])

    success_rate_dict: dict[str, list[float]] = {}
    for key, value in test_batch.non_tensor_batch.items():
        if "success_rate" in key:
            values = np.asarray(value)
            if len(values) > 0:
                success_rate_dict[key] = [float(values[0])]

    data_source_reward: dict[str, list[float]] = {}
    for i in range(reward_tensor_1d.shape[0]):
        data_source = str(data_sources[i])
        data_source_reward.setdefault(data_source, []).append(float(reward_tensor_1d[i].item()))

    data_source_tool_calling: dict[str, list[float]] = {}
    _, unique_idx = np.unique(traj_uids, return_index=True)
    unique_data_sources = data_sources[unique_idx]
    unique_tool_callings = tool_callings[unique_idx]
    for i in range(unique_tool_callings.shape[0]):
        data_source = str(unique_data_sources[i])
        data_source_tool_calling.setdefault(data_source, []).append(float(unique_tool_callings[i]))

    metric_dict: dict[str, float] = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"val/{data_source}/test_score"] = float(np.mean(rewards))
    for data_source, tool_calls in data_source_tool_calling.items():
        metric_dict[f"val/{data_source}/tool_call_count/mean"] = float(np.mean(tool_calls))
    for key, values in success_rate_dict.items():
        metric_dict[f"val/{key}"] = float(np.mean(values))
    return metric_dict


def save_results(config, metrics: dict[str, float]) -> Path:
    output_dir = Path(config.eval.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics.json"
    output_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_file


def main() -> None:
    args, overrides = parse_args()
    config = load_config(args.config, overrides)
    print(OmegaConf.to_yaml(config, resolve=True))

    tokenizer = load_tokenizer(config)
    init_ray(config)

    envs = None
    val_envs = None
    try:
        envs, val_envs = make_envs(config)
        api_actor = APIActorWrapper(
            tokenizer=tokenizer,
            api_base_urls=list(config.api.base_urls),
            model_name=str(config.api.model_name),
            api_key=str(config.api.get("api_key", "EMPTY")),
            max_prompt_length=int(config.data.max_prompt_length),
            max_response_length=int(config.data.max_response_length),
            temperature=float(config.api.get("temperature", 0.4)),
            top_p=float(config.api.get("top_p", 1.0)),
            max_tokens=int(config.api.get("max_tokens", config.data.max_response_length)),
            timeout=int(config.api.get("timeout", 120)),
        )
        traj_collector = TrajectoryCollector(config, tokenizer, processor=None, trajectory_saver=None)
        reward_fn = EpisodeRewardManager(
            tokenizer=tokenizer,
            num_examine=int(config.eval.get("num_examine", 0)),
            normalize_by_length=False,
        )
        metrics = run_evaluation(config, tokenizer, api_actor, traj_collector, val_envs, reward_fn)
        output_file = save_results(config, metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"[eval_runner] Metrics saved to {output_file}")
    finally:
        for manager in (val_envs, envs):
            if manager is not None:
                try:
                    manager.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"[eval_runner] Failed to close env manager: {exc}")
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
