import torch
from torch.utils.data import Dataset
import numpy as np
from vima.utils import *
# from vima_bench import *
from tokenizers import Tokenizer
from tokenizers import AddedToken
import os
import random
from einops import rearrange
import cv2
import torch.nn.functional as F
import sys




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



def prepare_obs(
    *,
    obs: dict,
    rgb_dict,
    meta_list,
    device = "cuda:0"  # 添加 device 参数，默认为 "cuda:0"
):
    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    segm_dict = obs.pop("segm")
    views = sorted(rgb_dict.keys())
    #assert meta["n_objects"] == len(meta["obj_id_to_info"])
    objects = meta_list

    L_obs = get_batch_size(obs)

    obs_list = {
        "ee": obs["ee"], #机械臂末端执行器的信息
        "objects": {
            "cropped_img": {view: [] for view in views}, #裁剪过的图像
            "bbox": {view: [] for view in views}, #边框
            "mask": {view: [] for view in views}, #掩码
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
                    continue #这个地方是记录对象是否过小需要填充
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                h, w = ymax - ymin, xmax - xmin
                bboxes.append([int(x_center), int(y_center), int(h), int(w)])#计算边框
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
                    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                if isinstance(cropped_img, torch.Tensor):
                    cropped_img = cropped_img.cpu().numpy()  # 先移到 CPU，再转 NumPy
                else:
                    cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img,
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                cropped_imgs.append(cropped_img) #图像裁剪->变成32*32的正方形
            bboxes = np.asarray(bboxes)
            cropped_imgs = np.asarray(cropped_imgs)
            mask = np.ones(len(bboxes), dtype=bool)
            if n_pad > 0:
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )
                cropped_imgs = np.concatenate(
                    [
                        cropped_imgs,
                        np.zeros(
                            (n_pad, 3, 32, 32),
                            dtype=cropped_imgs.dtype,
                        ),
                    ],
                    axis=0,
                )
                mask = np.concatenate([mask, np.zeros(n_pad, dtype=bool)], axis=0)
            obs_list["objects"]["bbox"][view].append(bboxes)
            obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_list["objects"]["mask"][view].append(mask)
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(
            obs_list["objects"]["bbox"][view], axis=0
        )
        obs_list["objects"]["cropped_img"][view] = np.stack(
            obs_list["objects"]["cropped_img"][view], axis=0
        )
        obs_list["objects"]["mask"][view] = np.stack(
            obs_list["objects"]["mask"][view], axis=0
        )

    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor(device=device)
    obs = any_transpose_first_two_axes(obs)
    return obs#把图像变成tensor


def preprocess_actions(actions, cfg):
    """
    将环境实际动作（物理空间）归一化为policy输出的动作格式。
    输入:
        actions: dict，包含实际动作（如pose0_position等），numpy或torch均可
        meta_info: 包含action_bounds的字典
        cfg: 配置，需有device
    输出:
        normed_actions: dict，归一化后的动作
    """
    # 1. 获取动作边界
    # action_bounds = [{'low': np.array([ 0.25, -0.5 ], dtype=np.float32), 'high': np.array([0.75, 0.5 ], dtype=np.float32)}]
    # action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
    # action_bounds_high = [action_bound["high"] for action_bound in action_bounds]
    # action_bounds_low = np.asarray(action_bounds_low)
    # action_bounds_high = np.asarray(action_bounds_high)
    # action_bounds_low = torch.tensor(action_bounds_low, dtype=torch.float32, device=cfg.device)
    # action_bounds_high = torch.tensor(action_bounds_high, dtype=torch.float32, device=cfg.device)

    # 1. 获取动作边界
    action_bounds_low = torch.tensor([0.25, -0.5], dtype=torch.float32, device=cfg.device)
    action_bounds_high = torch.tensor([0.75, 0.5], dtype=torch.float32, device=cfg.device)
    normed_actions = {}

    # 2. 归一化位置
    for key in ["pose0_position", "pose1_position"]:
        v = actions[key]
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float32, device=cfg.device)

        v_xy = v[..., :2]
        normed = (v_xy - action_bounds_low) / (action_bounds_high - action_bounds_low)
        normed = torch.clamp(normed, 0, 1)
        normed_actions[key] = normed

    # 3. 归一化旋转
    for key in ["pose0_rotation", "pose1_rotation"]:
        v = actions[key]
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float32, device=cfg.device)
        normed = (v + 1) / 2
        normed = torch.clamp(normed, 0, 1)
        normed_actions[key] = normed

    # 4. 其它动作直接复制
    for key in actions:
        if key not in normed_actions:
            normed_actions[key] = torch.tensor(actions[key], dtype=torch.float32, device=cfg.device) \
                if not torch.is_tensor(actions[key]) else actions[key]

    return normed_actions


# def preprocess_actions(actions, cfg):
#     """
#     将环境实际动作（物理空间）归一化为policy输出的动作格式。
#     输入:
#         actions: dict，包含实际动作（如pose0_position等），numpy或torch均可
#         meta_info: 包含action_bounds的字典
#         cfg: 配置，需有device
#     输出:
#         normed_actions: dict，归一化后的动作
#     """
#     # 1. 获取动作边界
#     action_bounds = [{'low': np.array([ 0.25, -0.5 ], dtype=np.float32), 'high': np.array([0.75, 0.5 ], dtype=np.float32)}]
#     action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
#     action_bounds_high = [action_bound["high"] for action_bound in action_bounds]
#     action_bounds_low = np.asarray(action_bounds_low)
#     action_bounds_high = np.asarray(action_bounds_high)
#     action_bounds_low = torch.tensor(action_bounds_low, dtype=torch.float32, device=cfg.device)
#     action_bounds_high = torch.tensor(action_bounds_high, dtype=torch.float32, device=cfg.device)

#     normed_actions = {}

#     # 2. 归一化位置
#     for key in ["pose0_position", "pose1_position"]:
#         v = actions[key]
#         if not torch.is_tensor(v):
#             v = torch.tensor(v, dtype=torch.float32, device=cfg.device)
#         normed = (v - action_bounds_low) / (action_bounds_high - action_bounds_low)
#         normed = torch.clamp(normed, 0, 1)
#         normed_actions[key] = normed

#     # 3. 归一化旋转
#     for key in ["pose0_rotation", "pose1_rotation"]:
#         v = actions[key]
#         if not torch.is_tensor(v):
#             v = torch.tensor(v, dtype=torch.float32, device=cfg.device)
#         normed = (v + 1) / 2
#         normed = torch.clamp(normed, 0, 1)
#         normed_actions[key] = normed

#     # 4. 其它动作直接复制
#     for key in actions:
#         if key not in normed_actions:
#             normed_actions[key] = torch.tensor(actions[key], dtype=torch.float32, device=cfg.device) \
#                 if not torch.is_tensor(actions[key]) else actions[key]

#     return normed_actions
def prepare_subprompt(prompt, views, device, policy, max_seq_len = 15):
    views = sorted(views)
    #text_prompt = prompt.get('text_prompt')
    text_prompt = prompt.get("text_prompt") if isinstance(prompt, dict) else None
    if text_prompt is None:
        raise ValueError("text_prompt missing")  # 让上层捕获并跳过
    #这个地方因为数据存在问题，所以进行了修改
    step = int(text_prompt[5])
    text_prompt = text_prompt[8:]
    text_prompt = text_prompt.split('[ALTER]')
    # 随机取text_prompt的一个元素作为text_prompt
    #text_prompt = random.choice(text_prompt)

    #!!!这里进行了修改！！！！
    text_prompt = text_prompt[0]
    # dragged_obj = prompt.get('{dragged_obj}')
    # dragged_obj_1 = prompt.get('{dragged_obj_1}')
    # base_obj = prompt.get('{base_obj}')
    encoding = tokenizer.encode(text_prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in PLACEHOLDERS:
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{") and token.endswith("}")
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
                }
            obj = prompt.get(token)
            for view in views:
                obj_repr["bbox"][view] = torch.stack([obj[f'{view}_bbox']])
                obj_repr["cropped_img"][view] = torch.stack([obj[f'{view}']])
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
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                token["mask"] = {
                    view: torch.ones((n_objs_prompt[view],), dtype=torch.bool)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: torch.zeros((n_objs_to_pad[view], 4), dtype=torch.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: torch.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=torch.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: torch.zeros((n_objs_to_pad[view]), dtype=torch.bool)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()

    word_batch = word_batch.to(device)
    image_batch = image_batch.to_torch_tensor(device=device)
    prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
        (raw_prompt_token_type, word_batch, image_batch), is_encoding=False
    )

    seq_len = prompt_tokens.size(0)
    if seq_len < max_seq_len:
        prompt_tokens = F.pad(
            prompt_tokens, 
            (0, 0, 0, 0, 0, max_seq_len - seq_len), 
            value=0
        )
        prompt_masks = F.pad(
            prompt_masks, 
            (0, max_seq_len - seq_len), 
            value=0
        )
    elif seq_len > max_seq_len:
        print(f'{text_prompt}subseqlen{seq_len}')
        prompt_tokens = prompt_tokens[:max_seq_len]
        prompt_masks = prompt_masks[:, :max_seq_len]
    
    # 去除 prompt_tokens 和 prompt_masks 中为 1 的维度（实际上貌似没有用处，不过不影响代码）
    prompt_tokens = prompt_tokens.squeeze(1)  # 从 (15, 1, 768) 变为 (15, 768)
    prompt_masks = prompt_masks.squeeze(0)    # 从 (1, 15) 变为 (15,)

    return prompt_tokens, prompt_masks



    # return raw_prompt_token_type, word_batch, image_batch
        
        


    
    # encoding = tokenizer.encode(prompt, add_special_tokens=True)
    # prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    # assert set(prompt_assets.keys()) == set(
    #     [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
    # )
    # filled_prompt = []
    # for id, token in zip(prompt_ids, prompt_tokens):
    #     if token not in PLACEHOLDERS:
    #         assert "{" not in token and "}" not in token
    #         filled_prompt.append(id)
    #     else:
    #         assert token.startswith("{") and token.endswith("}")
    #         asset_name = token[1:-1]
    #         assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
    #         asset = prompt_assets[asset_name]
    #         obj_info = asset["segm"]["obj_info"]
    #         placeholder_type = asset["placeholder_type"]
    #         if placeholder_type == "object":
    #             objects = [obj_info["obj_id"]]
    #         elif placeholder_type == "scene":
    #             objects = [each_info["obj_id"] for each_info in obj_info]
    #         obj_repr = {
    #             "cropped_img": {view: [] for view in views},
    #             "bbox": {view: [] for view in views},
    #         }
    #         for view in views:
    #             rgb_this_view = asset["rgb"][view]
    #             segm_this_view = asset["segm"][view]
    #             bboxes = []
    #             cropped_imgs = []
    #             for obj_id in objects:
    #                 ys, xs = np.nonzero(segm_this_view == obj_id)
    #                 if len(xs) < 2 or len(ys) < 2:
    #                     continue
    #                 xmin, xmax = np.min(xs), np.max(xs)
    #                 ymin, ymax = np.min(ys), np.max(ys)
    #                 x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    #                 h, w = ymax - ymin, xmax - xmin
    #                 bboxes.append([int(x_center), int(y_center), int(h), int(w)])
    #                 cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
    #                 if cropped_img.shape[1] != cropped_img.shape[2]:
    #                     diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
    #                     pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
    #                     if cropped_img.shape[1] > cropped_img.shape[2]:
    #                         pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
    #                     else:
    #                         pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
    #                     cropped_img = np.pad(
    #                         cropped_img,
    #                         pad_width,
    #                         mode="constant",
    #                         constant_values=0,
    #                     )
    #                     assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
    #                 cropped_img = rearrange(cropped_img, "c h w -> h w c")
    #                 cropped_img = np.asarray(cropped_img)
    #                 cropped_img = cv2.resize(
    #                     cropped_img,
    #                     (32, 32),
    #                     interpolation=cv2.INTER_AREA,
    #                 )
    #                 cropped_img = rearrange(cropped_img, "h w c -> c h w")
    #                 cropped_imgs.append(cropped_img)
    #             bboxes = np.asarray(bboxes)
    #             cropped_imgs = np.asarray(cropped_imgs)
    #             obj_repr["bbox"][view] = bboxes
    #             obj_repr["cropped_img"][view] = cropped_imgs
    #         filled_prompt.append(obj_repr)
    # raw_prompt = [filled_prompt]
    # max_n_objs_prompt = {view: 0 for view in views}
    # for prompt in raw_prompt:
    #     for token in prompt:
    #         if isinstance(token, dict):
    #             for view in views:
    #                 max_n_objs_prompt[view] = max(
    #                     max_n_objs_prompt[view], len(token["cropped_img"][view])
    #                 )
    # raw_prompt_token_type, word_batch, image_batch = [], [], []
    # for prompt in raw_prompt:
    #     token_type = []
    #     for token in prompt:
    #         if isinstance(token, int):
    #             token_type.append(0)
    #             word_batch.append(token)
    #         elif isinstance(token, dict):
    #             token_type.append(1)
    #             n_objs_prompt = {
    #                 view: len(token["cropped_img"][view]) for view in views
    #             }
    #             token["mask"] = {
    #                 view: np.ones((n_objs_prompt[view],), dtype=np.bool_)
    #                 for view in views
    #             }
    #             n_objs_to_pad = {
    #                 view: max_n_objs_prompt[view] - n_objs_prompt[view]
    #                 for view in views
    #             }
    #             objs_pad = {
    #                 "bbox": {
    #                     view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
    #                     for view in views
    #                 },
    #                 "cropped_img": {
    #                     view: np.zeros(
    #                         (n_objs_to_pad[view], 3, 32, 32),
    #                         dtype=np.uint8,
    #                     )
    #                     for view in views
    #                 },
    #                 "mask": {
    #                     view: np.zeros((n_objs_to_pad[view]), dtype=np.bool_)
    #                     for view in views
    #                 },
    #             }
    #             token = any_concat([token, objs_pad], dim=0)
    #             image_batch.append(token)
    #     raw_prompt_token_type.append(token_type)
    # assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
    #     word_batch
    # ) + len(image_batch)
    # word_batch = any_stack(word_batch, dim=0)
    # image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    # word_batch = any_to_torch_tensor(word_batch)
    # image_batch = image_batch.to_torch_tensor()
    # return raw_prompt_token_type, word_batch, image_batch


    

def prepare_prompt(*, prompt: str, prompt_assets: dict, views: list[str]):
    views = sorted(views)
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    assert set(prompt_assets.keys()) == set(
        [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
    )
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in PLACEHOLDERS:
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{") and token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
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
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
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
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=np.bool_)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=np.bool_)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch

class VIMADataset(Dataset):

    def __init__(self, data_dir, subprompt_dir, num_start, num_data, policy,
                    time_step_max, num_obj_max, num_prompt_max, device='cuda:0'):
        """
        新的数据读取方式：
        - data_dir 下为若干样本子目录（如 000000、000001…）
        - 每个样本目录包含：action.pkl, obs.pkl, trajectory.pkl, rgb_front/*.jpg, rgb_top/*.jpg
        - segm_front/top 与 end_effector 来自 obs.pkl（obs['segm']['front'] / obs['segm']['top'] / obs['ee']）
        - prompt 与 prompt_assets 来自 trajectory.pkl
        - actions 来自 action.pkl（保持为 numpy array）
        - rgb_front/top 读取文件夹内 jpg，并转换为 numpy array，形状 [T+1, C, H, W]，dtype=uint8
        - 与 subprompt 的索引一一对应：subprompt_dir/样本名.pt
        """
        import pickle

        self.data_dir = data_dir
        self.subprompt_dir = subprompt_dir
        self.num_start = num_start
        self.num_data = num_data
        self.policy = getattr(policy, "module", policy)
        self.device = device
        
        self.time_step_max = time_step_max
        self.num_obj_max = num_obj_max
        self.max_seq_len = num_prompt_max
        self.cfg = type('Config', (), {'device': device})()
        

        # 让历史数据中的 "vimasim" 能在反序列化时解析为 vima_bench
        try:
            import vima_bench
            sys.modules["vimasim"] = vima_bench
        except Exception:
            pass  # 若不可用，则可能 trajectory.pkl 中部分复杂对象无法完全反序列化

        # 枚举样本子目录
        all_dirs = sorted(
            [d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))]
        )
        sample_dirs = all_dirs[num_start:num_start + num_data]

        self.features = []
        self.actions = []
        self.subprompts = []

        def read_images_chw(img_dir):
            # 读取该目录下所有 .jpg，返回 numpy array，形状 [N, C, H, W]，dtype=uint8
            frames = []
            if os.path.isdir(img_dir):
                # 文件名按数字排序
                jpgs = sorted(
                    [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")],
                    key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
                )
                for name in jpgs:
                    path = os.path.join(img_dir, name)
                    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, HWC
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
                    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    frames.append(img.astype(np.uint8))
            if len(frames) == 0:
                return np.zeros((0, 3, 0, 0), dtype=np.uint8)
            return np.stack(frames, axis=0)

        loaded = 0
        for d in sample_dirs:
            sample_dir = os.path.join(self.data_dir, d)

            action_pkl = os.path.join(sample_dir, "action.pkl")
            obs_pkl = os.path.join(sample_dir, "obs.pkl")
            traj_pkl = os.path.join(sample_dir, "trajectory.pkl")

            if not (os.path.isfile(action_pkl) and os.path.isfile(obs_pkl) and os.path.isfile(traj_pkl)):
                continue

            # 读取 pkl（分别一个变量）
            try:
                with open(action_pkl, "rb") as f:
                    action = pickle.load(f)
                with open(obs_pkl, "rb") as f:
                    obs = pickle.load(f)
                with open(traj_pkl, "rb") as f:
                    trajectory = pickle.load(f)
            except ModuleNotFoundError as e:
                # 轨迹里若包含 vimasim 且未映射成功，跳过该样本
                print(f"[warn] skip {sample_dir} due to import error: {e}")
                continue

            # 读取图片并转为 numpy array（[T+1, C, H, W]）
            rgb_front = read_images_chw(os.path.join(sample_dir, "rgb_front"))
            rgb_top = read_images_chw(os.path.join(sample_dir, "rgb_top"))

            # 从 obs 中取 segm 与 end_effector（保持为 numpy array）
            # 期望 obs['segm']['front'] / obs['segm']['top'] 与 obs['ee'] 为 numpy array
            try:
                segm_front = np.asarray(obs["segm"]["front"])
                segm_top = np.asarray(obs["segm"]["top"])
                end_effector = np.asarray(obs["ee"])
            except Exception as e:
                print(f"[warn] bad obs structure in {sample_dir}: {e}")
                continue

            # 从 trajectory 中取 prompt 与 prompt_assets（保持为 numpy 可用对象）
            try:
                prompt = trajectory.get("prompt")
                prompt_assets = trajectory.get("prompt_assets")
            except Exception as e:
                print(f"[warn] bad trajectory structure in {sample_dir}: {e}")
                continue

            # actions 保持为 numpy array（字典里的每个键是 numpy）
            # action.pkl 示例中已是 numpy array
            # 若有标量或列表，统一转为 numpy
            actions_np = {}
            try:
                for k, v in action.items():
                    actions_np[k] = np.asarray(v)
            except Exception as e:
                print(f"[warn] bad action in {sample_dir}: {e}")
                continue

            # 读取对应 subprompt（与样本名一一对应）
            subprompt_file = os.path.join(self.subprompt_dir, f"/path/to/weights.pt")
            if not os.path.exists(subprompt_file):
                # 若缺少 subprompt，则跳过以保证索引对齐
                continue
            try:
                subprompt = torch.load(subprompt_file)
                # 过滤空 subprompt
                is_empty_list = isinstance(subprompt, list) and len(subprompt) == 0
                is_single_empty_dict = (
                    isinstance(subprompt, list) and len(subprompt) == 1
                    and isinstance(subprompt[0], dict) and len(subprompt[0]) == 0
                )
                if is_empty_list or is_single_empty_dict:
                    continue
            except Exception as e:
                print(f"[warn] failed to load subprompt {subprompt_file}: {e}")
                continue

            features = {
                "rgb_front": rgb_front,              # numpy, [T+1, C, H, W], uint8
                "rgb_top": rgb_top,                  # numpy, [T+1, C, H, W], uint8
                "segm_front": segm_front,            # numpy
                "segm_top": segm_top,                # numpy
                "end_effector": end_effector,        # numpy
                "prompt": prompt,                    # 任意对象（通常字符串）
                "prompt_assets": prompt_assets,      # 任意对象（通常 dict）
            }

            self.features.append(features)
            self.actions.append(actions_np)
            self.subprompts.append(subprompt)
            loaded += 1

        assert len(self.features) == len(self.actions) == len(self.subprompts), \
            "Features, actions, subprompts length mismatch"
        print(f"Loaded {loaded} samples from {self.data_dir}")
    

    # def __init__(self, data_dir, subprompt_dir, num_start,num_data,policy,time_step_max,num_obj_max,num_prompt_max,device='cuda:0'):
    #     """
    #     初始化数据集
    #     参数说明:
    #         data_dir (str): 数据文件夹路径，包含多个 .npz 文件，每个文件为一个轨迹样本。
    #         num_start (int): 从第几个样本开始读取（用于分批或分布式训练）。
    #         num_data (int): 读取的样本数量。
    #         policy: VIMA 策略模型实例，用于生成 token。
    #         time_step_max (int): 单个样本的最大时间步数（轨迹长度）。
    #         num_obj_max (int): 单个样本的最大物体数量（用于观测和 mask 填充）。
    #         num_prompt_max (int): prompt 的最大长度（用于 prompt token 填充）。
    #         device (str): 数据加载和处理所用的设备（如 'cuda:0' 或 'cpu'）。

    #     数据处理流程:
    #         1. 遍历 data_dir 下所有 .npz 文件，读取观测（features）和动作（actions）。
    #         2. 对每个样本，处理视觉观测（rgb、segm）、末端执行器状态、prompt 及其 assets。
    #         3. 对观测、动作、prompt 进行归一化、填充和 token 化，保证输入 shape 一致。
    #         4. 支持按最大时间步、最大物体数、最大 prompt 长度自动补零或截断。
    #         5. __getitem__ 返回神经网络训练所需的所有张量，包括 obs_token、obs_mask、pre_action、action_label、prompt_tokens、prompt_masks、index 

    #     """
    #     self.data_dir = data_dir
    #     self.num_data = num_data
    #     self.policy = policy
    #     self.num_start = num_start
    #     self.device = device
    #     self.time_step_max = time_step_max
    #     self.num_obj_max = num_obj_max
    #     self.max_seq_len = num_prompt_max # 这个是VIMA的最大prompt的长度,也是将来model的最大输入长度
    #     self.subprompt_dir = subprompt_dir

    #     # 获取所有 .pt 文件路径
    #     self.file_paths = sorted(
    #         [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    #     )[num_start:num_start + num_data]

    #     # 初始化存储 features 和 actions 的列表
    #     self.features = []
    #     self.actions = []
    #     self.subprompts = []
    #     self.cfg = type('Config', (), {'device': device})() # 创建一个简单的配置对象，包含 device 属性

    #     # 读取文件数据
    #     for file_path in self.file_paths:
    #         subprompt_file = os.path.join(self.subprompt_dir, file_path.split('/')[-1].split('.')[0]+"/path/to/weights.pt")
    #         if os.path.exists(subprompt_file):
    #             subprompt = torch.load(subprompt_file)
    #             # 空 subprompt 过滤：[] 或 [{}] 则跳过整个样本
    #             is_empty_list = isinstance(subprompt, list) and len(subprompt) == 0
    #             is_single_empty_dict = (
    #                 isinstance(subprompt, list)
    #                 and len(subprompt) == 1
    #                 and isinstance(subprompt[0], dict)
    #                 and len(subprompt[0]) == 0
    #             )
    #             if is_empty_list or is_single_empty_dict:
    #                 continue
    #             self.subprompts.append(subprompt)
    #         else:
    #             continue
    #         data = np.load(file_path, allow_pickle=True)
    #         # features字典包含所有观测相关内容
    #         features = {
    #             "rgb_front": data["rgb_front"],
    #             "rgb_top": data["rgb_top"],
    #             "segm_front": data["segm_front"],
    #             "segm_top": data["segm_top"],
    #             "end_effector": data["end_effector"],
    #             "prompt": data["prompt"],
    #             "prompt_assets": data["prompt_assets"],
    #         }
    #         # actions字典只包含动作
    #         actions = data["action"].item() if isinstance(data["action"], np.ndarray) and data["action"].shape == () else data["action"]
    #         self.features.append(features)
    #         self.actions.append(actions)
            




    #     assert len(self.features) == len(self.actions), "Features and actions must have the same length"
    #     # print("Loaded {} files from {}".format(len(self.file_paths), data_dir))
    #     print("Loaded {} files from {}".format(len(self.features), data_dir))




    def __len__(self):
        # 返回数据集的大小（文件数量）
        # return len(self.file_paths)
        return len(self.features)

    #此处getitem修改，就是因为subprompt的数据有问题，因此如果出现数据有问题，那么就跳过这条数据
    def __getitem__(self, idx):

        features = self.features[idx]
        actions = self.actions[idx]
        subprompts = self.subprompts[idx]
        """
        处理obs_token和obs_mask以及action_tokens，还要返回对应labels
        """
        action_pose0_position = actions["pose0_position"]     
        time_step_max_now =  action_pose0_position.shape[0]#确定这个数据最大的timestep能取多少
        #assert time_step_max_now == features["rgb_front"].shape[0] - 1
         # 检查 `rgb_front` 的时间步数是否满足要求
        if time_step_max_now != features["rgb_front"].shape[0] - 1:
            print(f"[Warning] Skipping sample {idx}: time_step_max_now ({time_step_max_now}) "
                f"!= rgb_front.shape[0] - 1 ({features['rgb_front'].shape[0] - 1})")
            return None
        num_max_obj_now = np.unique(features["segm_front"]) - 2#确定这个数据最大的物体数量能取多少

        # 校验 subprompt 完整性：需要至少 time_step_max_now 个且每个都有非空 text_prompt
        def has_text_prompt(sp):
            return isinstance(sp, dict) and ("text_prompt" in sp) and bool(sp["text_prompt"])

        if (not isinstance(subprompts, list)) or (len(subprompts) < time_step_max_now) \
           or any(not has_text_prompt(subprompts[t]) for t in range(time_step_max_now)):
            # 丢弃该样本（由 collate_fn 过滤 None）
            return None

        index = random.randint(0, time_step_max_now-1)#比如当前最大的time_step = 2,那么其实index可以取值为0和1

        
        #index = 0
        
        #需要返回的东西，先进行初始化
        # obs_token
        # obs_mask
        # pre_action
        # action_label-> label
        
        obs_token = torch.zeros((self.time_step_max + 1, 1 , self.num_obj_max * 2,  256), dtype=torch.float32)
        obs_mask = torch.zeros((self.time_step_max + 1, 1 , self.num_obj_max * 2), dtype=torch.float32)
        pre_action = torch.zeros((self.time_step_max , 256), dtype=torch.float32)
        # pre_subprompt = torch.zeros((self.time_step_max , 10, 256), dtype=torch.float32)
        pre_action_mask = torch.zeros((self.time_step_max), dtype=torch.float32)

        action_label = {k: v[index] for k, v in actions.items()}#注意，真实的label我在考虑是否需要添加最后一步的obj，这个地方暂时没有添加进去

        # 处理obs_token和obs_mask
        for i in range(index+1):
            obs = {
                "rgb": {"front":features["rgb_front"][i:i+1],"top":features["rgb_top"][i:i+1]},
                "segm": {"front":features["segm_front"][i:i+1],"top":features["segm_top"][i:i+1]},
                "ee": features["end_effector"][i:i+1],#这个地方是否需要取i？存疑，因为貌似没有用到，代表是当前的末端执行器信息
            }

            meta_list = np.unique(features['segm_front'])#提取物体的id
            meta_list = np.setdiff1d(meta_list, [0, 1])#去掉背景和0

            obs_after_prepare = prepare_obs(obs=obs, rgb_dict=None, meta_list=meta_list,device = self.device)#相对于原来的prepare_obs函数，进行了简单的改变，将meta_list传入代替了meta_data，device
            
            obs_token_this_step, obs_mask_this_step = self.policy.forward_obs_token(obs_after_prepare)
            
            """
            这个地方填充过程存疑，因为没有进行验证
            假设最大物体数为4个
            然后这个数据是2个，那么我们需要把2填充成4
            但是因为有两个视角，所以2*2->4*2
            那么有两种填充方法
            1 1 1 1 0 0 0 0
            1 1 0 0 1 1 0 0（1代表原始物体，0代表填充）
            我这里采用的是第一种方法

            """
            # 修正填充逻辑
            # 在 VIMADataset.__getitem__ 中找到这部分代码并替换：

            # 修正填充逻辑
            if obs_token[i].shape != obs_token_this_step[0].shape:
                # 获取维度信息
                target_shape = obs_token[i].shape      # (1, 6, 256) - 预分配的目标形状
                current_shape = obs_token_this_step[0].shape  # (1, 4, 256) - 当前实际形状
                
                # 比较第二个维度（物体数量×2维度）
                Q_target = target_shape[1]    # 6 - 预分配的最大物体数×2
                Q_current = current_shape[1]  # 4 - 当前实际物体数×2
                
                if Q_current > Q_target:
                    # 情况1：当前物体数超过预分配大小，需要截断
                    obs_token_processed = obs_token_this_step[0][:, :Q_target, :]
                    obs_mask_processed = obs_mask_this_step[0][:, :Q_target]
                    
                elif Q_current < Q_target:
                    # 情况2：需要填充 - 按物体对填充
                    current_objects = Q_current // 2  # 当前物体数
                    target_objects = Q_target // 2    # 目标物体数
                    objects_to_pad = target_objects - current_objects  # 需要填充的物体数
                    
                    #print(f"Padding from {current_objects} objects to {target_objects} objects")
                    
                    # 重新排列：按物体对进行填充，而不是在末尾填充
                    obs_token_processed = obs_token_this_step[0].clone()  # [1, 4, 256]
                    obs_mask_processed = obs_mask_this_step[0].clone()    # [1, 4]
                    
                    # 为每个需要填充的物体添加一对位置（front + top）
                    for _ in range(objects_to_pad):
                        # 添加一对零填充（物体的front视角 + top视角）
                        zero_token_pair = torch.zeros(1, 2, obs_token_processed.shape[-1], 
                                                    dtype=obs_token_processed.dtype, 
                                                    device=obs_token_processed.device)  # [1, 2, 256]
                        zero_mask_pair = torch.zeros(1, 2, 
                                                dtype=obs_mask_processed.dtype, 
                                                device=obs_mask_processed.device)    # [1, 2]
                        
                        # 在第二个维度上连接
                        obs_token_processed = torch.cat([obs_token_processed, zero_token_pair], dim=1)
                        obs_mask_processed = torch.cat([obs_mask_processed, zero_mask_pair], dim=1)
                    
                else:
                    # 情况3：形状完全匹配，直接使用
                    obs_token_processed = obs_token_this_step[0]
                    obs_mask_processed = obs_mask_this_step[0]
                
                # 验证最终形状
                assert obs_token_processed.shape == target_shape, \
                    f"Shape mismatch after padding: got {obs_token_processed.shape}, expected {target_shape}"
                assert obs_mask_processed.shape == target_shape[:2], \
                    f"Mask shape mismatch: got {obs_mask_processed.shape}, expected {target_shape[:2]}"
                
                # 赋值
                obs_token[i] = obs_token_processed
                obs_mask[i] = obs_mask_processed
                
            else:
                # 形状匹配，直接赋值
                obs_token[i] = obs_token_this_step[0]
                obs_mask[i] = obs_mask_this_step[0]

        # 处理pre_action

        for i in range(index):
            selected_actions = {key: actions[key][i] for key in actions.keys() if key != 'action_mask'}
            
            processed_actions = preprocess_actions(selected_actions,self.cfg)#这个就是照抄example.py的处理方式

            # 修正位置动作维度
            for pos_key in ["pose0_position", "pose1_position"]:
                if pos_key in processed_actions:
                    tensor = processed_actions[pos_key]
                    
                    # 压缩多余维度
                    while tensor.dim() > 1 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    
                    # 如果还是多维，取第一行
                    if tensor.dim() > 1:
                        tensor = tensor[0]
                    
                    # 只取前2个元素（x, y）
                    if len(tensor) > 2:
                        tensor = tensor[:2]
                    
                    processed_actions[pos_key] = tensor
                    
                    # 验证最终结果
                    assert tensor.dim() == 1, f"{pos_key} must be 1D, got {tensor.shape}"
                    assert len(tensor) == 2, f"{pos_key} must have 2 elements, got {len(tensor)}"
                    assert torch.is_tensor(tensor), f"{pos_key} must be tensor, got {type(tensor)}"

            # 通过 policy.forward_action_token 获取 pre_action_prompt
            pre_action_prompt = self.policy.forward_action_token_train(processed_actions)
            assert pre_action[i].shape == pre_action_prompt.shape, f"pre_action shape mismatch: {pre_action[i].shape} vs {pre_action_prompt.shape}"
            pre_action[i] = pre_action_prompt
            pre_action_mask[i] = 1.0  # 标记为有效的预动作


        # 预分配定长张量并填充，避免额外的 concat/pad
        max_T = self.time_step_max+1
        steps_to_fill = min(index + 1, max_T)
        # print(str(features["prompt"]))
        # print(idx)
        # print(subprompts[0])
        
        first_mm, first_mask = prepare_subprompt(subprompts[0], ["front", "top"], self.device, self.policy)
        S, D = first_mm.shape
        subprompt_multimodals = torch.zeros((max_T, S, D), dtype=first_mm.dtype, device=self.device)
        subprompt_masks = torch.zeros((max_T, S), dtype=torch.bool, device=self.device)
        # 第 0 步已计算，先写入
        subprompt_multimodals[0] = first_mm
        subprompt_masks[0] = first_mask.bool()
        # 后续步骤逐步填入，超出 max_T 的自然截断
        for t in range(1, steps_to_fill):
            subprompt = subprompts[t]
            # print(subprompt)
            mm, msk = prepare_subprompt(subprompt, ["front", "top"], self.device, self.policy)
            # 保险起见，若形状不同则截断到一致形状
            if mm.shape != (S, D):
                mm = mm[:S, :D]
            if msk.shape != (S,):
                msk = msk[:S]
            subprompt_multimodals[t] = mm
            subprompt_masks[t] = msk.bool()

        



        """
        处理prompt_tokens和prompt_masks
        prompt_tokens  -> (self.max_seq_len, 256)
        prompt_masks   -> (self.max_seq_len)
        这里的self.max_seq_len是VIMA每个任务的最大prompt长度,也是将来model的最大输入长度
        """


        prompt_token_type, word_batch, image_batch = prepare_prompt(
            prompt=str(features["prompt"]), 
            prompt_assets=features["prompt_assets"], 
            views=["front", "top"]
        )

        word_batch = word_batch.to(self.device)
        image_batch = image_batch.to_torch_tensor(device=self.device)
        prompt_tokens, prompt_masks = self.policy.forward_prompt_assembly(
            (prompt_token_type, word_batch, image_batch)
        )

        seq_len = prompt_tokens.size(0)
        if seq_len < self.max_seq_len:
            prompt_tokens = F.pad(
                prompt_tokens, 
                (0, 0, 0, 0, 0, self.max_seq_len - seq_len), 
                value=0
            )
            prompt_masks = F.pad(
                prompt_masks, 
                (0, self.max_seq_len - seq_len), 
                value=0
            )
        elif seq_len > self.max_seq_len:
            prompt_tokens = prompt_tokens[:self.max_seq_len]
            prompt_masks = prompt_masks[:, :self.max_seq_len]
        
        # 去除 prompt_tokens 和 prompt_masks 中为 1 的维度（实际上貌似没有用处，不过不影响代码）
        prompt_tokens = prompt_tokens.squeeze(1)  # 从 (56, 1, 256) 变为 (56, 256)
        prompt_masks = prompt_masks.squeeze(0)    # 从 (1, 56) 变为 (56,)

        return prompt_tokens,prompt_masks,obs_token,obs_mask,pre_action,pre_action_mask,action_label,subprompt_multimodals,subprompt_masks,index
