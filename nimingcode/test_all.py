from __future__ import annotations
import os
import warnings
import argparse
import json
from datetime import datetime
import numpy as np
import cv2
import torch
from gym import Wrapper
from gym.wrappers import TimeLimit as _TimeLimit
from tokenizers import Tokenizer, AddedToken
from einops import rearrange

from vima.utils import *
from vima_bench import *
from vima_policy import VIMAPolicy  # 假设你使用的是这个Policy文件，如果不是请改回 vima_policy

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =============================================================================
# Tokenizer & Placeholders Setup
# =============================================================================
_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

PLACEHOLDER_TOKENS = [
    AddedToken("{base_obj}", **_kwargs),
    AddedToken("{base_obj_1}", **_kwargs),
    AddedToken("{base_obj_2}", **_kwargs),
    AddedToken("{dragged_obj}", **_kwargs),
    AddedToken("{dragged_obj_1}", **_kwargs),
    AddedToken("{dragged_obj_2}", **_kwargs),
    AddedToken("{dragged_obj_3}", **_kwargs),
    AddedToken("{dragged_obj_4}", **_kwargs),
    AddedToken("{dragged_obj_5}", **_kwargs),
    AddedToken("{swept_obj}", **_kwargs),
    AddedToken("{bounds}", **_kwargs),
    AddedToken("{constraint}", **_kwargs),
    AddedToken("{scene}", **_kwargs),
    AddedToken("{demo_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_blicker_obj_3}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_3}", **_kwargs),
    AddedToken("{start_scene}", **_kwargs),
    AddedToken("{end_scene}", **_kwargs),
    AddedToken("{before_twist_1}", **_kwargs),
    AddedToken("{after_twist_1}", **_kwargs),
    AddedToken("{before_twist_2}", **_kwargs),
    AddedToken("{after_twist_2}", **_kwargs),
    AddedToken("{before_twist_3}", **_kwargs),
    AddedToken("{after_twist_3}", **_kwargs),
    AddedToken("{frame_0}", **_kwargs),
    AddedToken("{frame_1}", **_kwargs),
    AddedToken("{frame_2}", **_kwargs),
    AddedToken("{frame_3}", **_kwargs),
    AddedToken("{frame_4}", **_kwargs),
    AddedToken("{frame_5}", **_kwargs),
    AddedToken("{frame_6}", **_kwargs),
    AddedToken("{ring}", **_kwargs),
    AddedToken("{hanoi_stand}", **_kwargs),
    AddedToken("{start_scene_1}", **_kwargs),
    AddedToken("{end_scene_1}", **_kwargs),
    AddedToken("{start_scene_2}", **_kwargs),
    AddedToken("{end_scene_2}", **_kwargs),
    AddedToken("{start_scene_3}", **_kwargs),
    AddedToken("{end_scene_3}", **_kwargs),
]
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
tokenizer = Tokenizer.from_file("/path/to/workspacevima_code/configs/tokenizer.json")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)

# =============================================================================
# Utils
# =============================================================================
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def prepare_prompt(*, prompt: str, prompt_assets: dict, views: list[str]):
    views = sorted(views)
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in PLACEHOLDERS:
            filled_prompt.append(id)
        else:
            asset_name = token[1:-1]
            asset = prompt_assets[asset_name]
            obj_info = asset["segm"]["obj_info"]
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "object":
                objects = [obj_info["obj_id"]]
            elif placeholder_type == "scene":
                objects = [each_info["obj_id"] for each_info in obj_info]
            
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }
            for view in views:
                rgb_this_view = asset["rgb"][view]
                segm_this_view = asset["segm"][view]
                bboxes = []
                cropped_imgs = []
                for obj_id in objects:
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img, pad_width, mode="constant", constant_values=0
                        )
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img, (32, 32), interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
            filled_prompt.append(obj_repr)
            
    raw_prompt = [filled_prompt]
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["cropped_img"][view])
                    )
    
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {view: len(token["cropped_img"][view]) for view in views}
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=bool) for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view] for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros((n_objs_to_pad[view], 3, 32, 32), dtype=np.uint8)
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=bool) for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
        
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))
    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch

def prepare_obs(*, obs: dict, rgb_dict: dict | None = None, meta: dict):
    rgb_dict = rgb_dict or obs.pop("rgb")
    segm_dict = obs.pop("segm")
    views = sorted(rgb_dict.keys())
    objects = list(meta["obj_id_to_info"].keys())
    L_obs = get_batch_size(obs)

    obs_list = {
        "ee": obs["ee"],
        "objects": {
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    for l in range(L_obs):
        rgb_dict_this_step = any_slice(rgb_dict, np.s_[l])
        segm_dict_this_step = any_slice(segm_dict, np.s_[l])
        for view in views:
            rgb_this_view = rgb_dict_this_step[view]
            segm_this_view = segm_dict_this_step[view]
            bboxes = []
            cropped_imgs = []
            n_pad = 0
            for obj_id in objects:
                ys, xs = np.nonzero(segm_this_view == obj_id)
                if len(xs) < 2 or len(ys) < 2:
                    n_pad += 1
                    continue
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                h, w = ymax - ymin, xmax - xmin
                bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                if cropped_img.shape[1] != cropped_img.shape[2]:
                    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                    if cropped_img.shape[1] > cropped_img.shape[2]:
                        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                    else:
                        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    cropped_img = np.pad(
                        cropped_img, pad_width, mode="constant", constant_values=0
                    )
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img, (32, 32), interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                cropped_imgs.append(cropped_img)
            bboxes = np.asarray(bboxes)
            cropped_imgs = np.asarray(cropped_imgs)
            mask = np.ones(len(bboxes), dtype=bool)
            if n_pad > 0:
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )
                cropped_imgs = np.concatenate(
                    [cropped_imgs, np.zeros((n_pad, 3, 32, 32), dtype=cropped_imgs.dtype)],
                    axis=0,
                )
                mask = np.concatenate([mask, np.zeros(n_pad, dtype=bool)], axis=0)
            obs_list["objects"]["bbox"][view].append(bboxes)
            obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_list["objects"]["mask"][view].append(mask)
            
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(obs_list["objects"]["bbox"][view], axis=0)
        obs_list["objects"]["cropped_img"][view] = np.stack(obs_list["objects"]["cropped_img"][view], axis=0)
        obs_list["objects"]["mask"][view] = np.stack(obs_list["objects"]["mask"][view], axis=0)

    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor()
    obs = any_transpose_first_two_axes(obs)
    return obs

class ResetFaultToleranceWrapper(Wrapper):
    max_retries = 10
    def __init__(self, env): super().__init__(env)
    def reset(self):
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except:
                try:
                    current_seed = self.env.unwrapped.task.seed
                    self.env.global_seed = current_seed + 1
                except:
                    pass
        raise RuntimeError("Failed to reset environment after {} retries".format(self.max_retries))

class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)

def create_policy_from_ckpt(ckpt_path, device, nlir_decomposer="NLIRDecomposer"):
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    cfg = checkpoint.get("cfg", {})
    if "xf_n_layers" not in cfg: 
        print("[Warning] cfg missing xattn params, using defaults")
        cfg.update({"xf_n_layers": 1, "sattn_n_heads": 8, "xattn_n_heads": 8})
    cfg["nlir_decomposer"] = nlir_decomposer
    
    policy_instance = VIMAPolicy(**cfg)
    
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = policy_instance.load_state_dict(new_state_dict, strict=False)
    policy_instance.to(device)
    policy_instance.eval()

    print("="*40)
    print(f"Checkpoint Loaded: {ckpt_path}")
    critical_modules = ["action_generator", "nlir_decomposer", "sent_decomposer"]
    has_critical_missing = False
    for module in critical_modules:
        module_missing = [k for k in missing_keys if k.startswith(module)]
        if len(module_missing) > 0:
            has_critical_missing = True
            print(f"⚠️ [CRITICAL WARNING] Module '{module}' has {len(module_missing)} missing keys!")
        else:
            print(f"✅ Module '{module}' loaded successfully.")
    print("="*40)
    return policy_instance

# =============================================================================
# Core Evaluation Loop
# =============================================================================
@torch.no_grad()
def run_episodes_for_task(policy, device, partition: str, task_name: str, episodes_per_seed: int, seeds: list[int]):
    try:
        specs = PARTITION_TO_SPECS["test"][partition][task_name]
    except KeyError:
        print(f"Skipping task {task_name}, not found in specs.")
        return 0.0

    total_success = 0
    total_episodes = 0

    for seed in seeds:
        env = TimeLimitWrapper(
            ResetFaultToleranceWrapper(
                make(
                    task_name,
                    modalities=["segm", "rgb"],
                    task_kwargs=specs,
                    seed=seed,
                    render_prompt=False,
                    display_debug_window=False,
                    hide_arm_rgb=True,
                )
            ),
            bonus_steps=2,
        )
        max_steps = getattr(env, "_max_episode_steps", 10)

        for ep_idx in range(1, episodes_per_seed + 1):
            env.global_seed = seed
            try:
                obs = env.reset()
            except:
                total_episodes += 1
                continue

            meta_info = env.meta_info
            prompt = env.prompt
            prompt_assets = env.prompt_assets

            prompt_token_type, word_batch, image_batch = prepare_prompt(
                prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
            )
            word_batch = word_batch.to(device)
            image_batch = image_batch.to_torch_tensor(device=device)
            prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
                (prompt_token_type, word_batch, image_batch)
            )

            inference_cache = {"obs_tokens": [], "obs_masks": [], "action_tokens": []}
            success = False
            done = False
            
            for step_i in range(max_steps):
                obs_copy = {
                    "rgb": {k: v.copy() for k, v in obs.get("rgb", {}).items()},
                    "segm": {k: v.copy() for k, v in obs.get("segm", {}).items()},
                    "ee": np.asarray(obs["ee"]),
                }
                obs_for_encode = {"rgb": obs_copy["rgb"], "segm": obs_copy["segm"], "ee": obs_copy["ee"]}
                obs_with_batch = add_batch_dim(obs_for_encode)
                obs_tensor = prepare_obs(obs=obs_with_batch, rgb_dict=None, meta=meta_info).to_torch_tensor(
                    device=device
                )
                
                obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs_tensor)
                obs_token_this_step = obs_token_this_step.squeeze(0)  
                obs_mask_this_step = obs_mask_this_step.squeeze(0)    
                inference_cache["obs_tokens"].append(obs_token_this_step[0])
                inference_cache["obs_masks"].append(obs_mask_this_step[0])

                # 对齐处理
                max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
                obs_tokens_this_env, obs_masks_this_env = [], []
                for t in range(len(inference_cache["obs_tokens"])):
                    obs_this = inference_cache["obs_tokens"][t]
                    mask_this = inference_cache["obs_masks"][t]
                    required_pad = max_objs - obs_this.shape[0]
                    obs_tokens_this_env.append(
                        any_concat(
                            [
                                obs_this,
                                torch.zeros(required_pad, obs_this.shape[1], device=device, dtype=obs_this.dtype),
                            ],
                            dim=0,
                        )
                    )
                    obs_masks_this_env.append(
                        any_concat(
                            [
                                mask_this,
                                torch.zeros(required_pad, device=device, dtype=mask_this.dtype),
                            ],
                            dim=0,
                        )
                    )
                
                obs_tokens_to_forward = any_stack([any_stack(obs_tokens_this_env, dim=0)], dim=0).transpose(0, 1)
                obs_masks_to_forward = any_stack([any_stack(obs_masks_this_env, dim=0)], dim=0).transpose(0, 1)

                action_tokens_to_forward = (
                    None
                    if step_i == 0
                    else any_stack([any_stack(inference_cache["action_tokens"], dim=0)], dim=0).transpose(0, 1)
                )

                # =========================================================
                # 这里使用的是 test.py 中指定的 forward_test_res_gate 逻辑
                # =========================================================
                predicted_action_tokens, _, _, _ = policy.forward_test_res_gate(
                    obs_token=obs_tokens_to_forward,
                    action_token=action_tokens_to_forward,
                    prompt_token=prompt_tokens,
                    prompt_token_mask=prompt_masks,
                    obs_mask=obs_masks_to_forward,
                    subprompt_multimodals=None,
                    subprompt_masks=None,
                )
                
                predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(0)
                dist_dict = policy.forward_action_decoder(predicted_action_tokens)
                actions = {k: v.mode() for k, v in dist_dict.items()}
                action_tokens = policy.forward_action_token(actions).squeeze(0)
                inference_cache["action_tokens"].append(action_tokens[0])

                actions = policy._de_discretize_actions(actions)
                action_bounds = [meta_info["action_bounds"]]
                low = torch.tensor(np.asarray([a["low"] for a in action_bounds]), dtype=torch.float32, device=device)
                high = torch.tensor(np.asarray([a["high"] for a in action_bounds]), dtype=torch.float32, device=device)
                actions["pose0_position"] = torch.clamp(actions["pose0_position"] * (high - low) + low, min=low, max=high)
                actions["pose1_position"] = torch.clamp(actions["pose1_position"] * (high - low) + low, min=low, max=high)
                actions["pose0_rotation"] = torch.clamp(actions["pose0_rotation"] * 2 - 1, min=-1, max=1)
                actions["pose1_rotation"] = torch.clamp(actions["pose1_rotation"] * 2 - 1, min=-1, max=1)

                actions = {k: v.cpu().numpy() for k, v in actions.items()}
                actions = any_slice(actions, np.s_[0, 0])
                try:
                    obs, _, done, info = env.step(actions)
                except Exception:
                    done = True
                    info = {"success": False}

                if done:
                    success = bool(info.get("success", False))
                    break

            total_episodes += 1
            total_success += int(success)

        try:
            env.close()
        except:
            pass

    task_success_rate = (total_success / total_episodes) if total_episodes > 0 else 0.0
    return task_success_rate

def main(cfg):
    policy = create_policy_from_ckpt(cfg.ckpt, cfg.device, nlir_decomposer=cfg.nlir_decomposer)
    
    seeds = [int(s) for s in cfg.seeds.split(",")] if isinstance(cfg.seeds, str) else cfg.seeds
    if isinstance(seeds, str): seeds = [int(s) for s in seeds.split(',')]
    episodes_per_seed = int(cfg.episodes)

    # 获取所有测试 Partition
    test_partitions = PARTITION_TO_SPECS["test"]
    partition_names = sorted(list(test_partitions.keys()))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_results = {
        "timestamp": timestamp,
        "ckpt": cfg.ckpt,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "partition_details": {}
    }

    _ensure_dir(cfg.save_dir)

    print(f"Starting Evaluation on {len(partition_names)} partitions...")

    for partition in partition_names:
        tasks_map = test_partitions[partition]
        task_names = sorted(list(tasks_map.keys()))
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing Partition: {partition}")
        
        partition_stats = {
            "tasks": {},
            "partition_average": 0.0
        }

        success_rates_in_partition = []

        for task_name in task_names:
            print(f"  > Evaluating Task: {task_name} ...", end=" ", flush=True)
            sr = run_episodes_for_task(
                policy=policy,
                device=cfg.device,
                partition=partition,
                task_name=task_name,
                episodes_per_seed=episodes_per_seed,
                seeds=seeds
            )
            partition_stats["tasks"][task_name] = sr
            success_rates_in_partition.append(sr)
            print(f"SR: {sr:.4f}")

        if len(success_rates_in_partition) > 0:
            avg_sr = sum(success_rates_in_partition) / len(success_rates_in_partition)
        else:
            avg_sr = 0.0
        
        partition_stats["partition_average"] = avg_sr
        final_results["partition_details"][partition] = partition_stats
        
        print(f"  [Result] Partition '{partition}' Average: {avg_sr:.4f}")

        # 实时保存，防止中断
        out_fp = os.path.join(cfg.save_dir, f"result_all_{timestamp}.json")
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] All results saved to {out_fp}")

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    # 下面的 partition 和 task 参数被忽略，脚本会运行所有任务
    arg.add_argument("--partition", type=str, default="start", help="Ignored")
    arg.add_argument("--task", type=str, default="start", help="Ignored")
    arg.add_argument("--ckpt", type=str, default="/path/to/workspacevima_code/result/20260118-235721/epoch_2.ckpt")
    arg.add_argument("--device", default="cuda:0")
    arg.add_argument("--save_dir", type=str, default="/path/to/workspacetestresult_vima")
    arg.add_argument("--seeds", type=str, default="42,43,44,45,46")  
    arg.add_argument("--episodes", type=int, default=40)   
    arg.add_argument(
        "--nlir_decomposer",
        type=str,
        choices=["NLIRDecomposer", "NLIRDecomposer_t5"],
        default="NLIRDecomposer_t5",
    )    
    cfg = arg.parse_args()
    main(cfg)