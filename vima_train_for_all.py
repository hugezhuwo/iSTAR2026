import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import argparse
import sys
# import vima
# from vima import create_policy_from_ckpt
# from gym.wrappers import TimeLimit as _TimeLimit
from vima.utils import *
# sys.path.insert(0, "/path/to/workspaceVIMABench")
from vima_bench import *
# from vima.policy import *
from vima_policy import VIMAPolicy
# from gym import Wrapper
# from tokenizers import Tokenizer
# from tokenizers import AddedToken
import os
# import random
from einops import rearrange
# import cv2
# from torch.utils._pytree import tree_map
import json
from vima_dataset import VIMADataset, preprocess_actions
# from nlir_decomposer import PromptDecoder

def create_policy_from_ckpt(ckpt_path, device):
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    checkpoint = torch.load(ckpt_path, map_location=device)
    policy_instance = VIMAPolicy(**checkpoint["cfg"])
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items()}

    try:
        policy_instance.load_state_dict(new_state_dict, strict=False)
        # policy_instance.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded the status dictionary.")
    except RuntimeError as e:
        print("An error occurred while loading the state dictionary: {}".format(e))
        for key in state_dict.keys():
            try:
                policy_instance.load_state_dict({key.replace("policy.", ""): state_dict[key]}, strict=False)
                print("Successfully loaded key: {}".format(key))
            except:
                print("Unable to load key: {}".format(key))
    policy_instance.to(device)
    # policy_instance.eval()
    return policy_instance

#这是支持变长输入的VIMA_train
"""
可优化空间：对于不同的prompt，不一定要统一长度到56，而是可以根据任务进行判定
这个是最终的代码，可以直接在terminal运行
"""

def collate_by_index(batch):
    """
    按index分组的collate函数，解决不同序列长度的CUDA索引问题
    
    Args:
        batch: list of tuples, 每个tuple包含 (prompt_tokens, prompt_masks, obs_token, obs_mask, pre_action, pre_action_mask, action_label, index)
    
    Returns:
        list of grouped batches, 每个batch内部的序列长度一致
    """
    # 提取所有index值
    indices = [item[-1] for item in batch]  # index是最后一个元素
    indices = torch.tensor(indices)
    
    # 获取唯一的index值
    unique_indices = torch.unique(indices)
    
    # 按index分组
    grouped_batches = []
    
    for idx in unique_indices:
        # 找到所有index等于idx的sample
        mask = (indices == idx)
        group_indices = torch.where(mask)[0]
        
        # 提取该组的所有样本
        group_samples = [batch[i] for i in group_indices]
        
        # 对该组样本进行标准的tensor堆叠
        prompt_tokens_list = []
        prompt_masks_list = []
        obs_token_list = []
        obs_mask_list = []
        pre_action_list = []
        pre_action_mask_list = []
        action_label_list = {key: [] for key in group_samples[0][6].keys()}  # action_label是第7个元素
        subprompt_multimodals_list = []
        subprompt_masks_list = []
        index_list = []
        
        # 收集该组所有样本的数据
        for sample in group_samples:
            prompt_tokens, prompt_masks, obs_token, obs_mask, pre_action, pre_action_mask, action_label, subprompt_multimodals,subprompt_masks, index = sample
            
            prompt_tokens_list.append(prompt_tokens)
            prompt_masks_list.append(prompt_masks)
            obs_token_list.append(obs_token)
            obs_mask_list.append(obs_mask)
            pre_action_list.append(pre_action)
            pre_action_mask_list.append(pre_action_mask)
            subprompt_multimodals_list.append(subprompt_multimodals)
            subprompt_masks_list.append(subprompt_masks)
            index_list.append(index)
            
            # 收集action_label的每个key
            for key, value in action_label.items():
                action_label_list[key].append(value)
        
        # 堆叠成tensor
        grouped_prompt_tokens = torch.stack(prompt_tokens_list, dim=0)  # (group_size, seq_len, 256)
        grouped_prompt_masks = torch.stack(prompt_masks_list, dim=0)    # (group_size, seq_len)
        grouped_obs_token = torch.stack(obs_token_list, dim=0)          # (group_size, time_step_max+1, 1, num_obj*2, 256)
        grouped_obs_mask = torch.stack(obs_mask_list, dim=0)            # (group_size, time_step_max+1, 1, num_obj*2)
        grouped_pre_action = torch.stack(pre_action_list, dim=0)        # (group_size, time_step_max, 256)
        grouped_pre_action_mask = torch.stack(pre_action_mask_list, dim=0)  # (group_size, time_step_max)
        grouped_subprompt_multimodals = torch.stack(subprompt_multimodals_list, dim=0)  # (group_size, time_step_max)
        grouped_subprompt_masks = torch.stack(subprompt_masks_list, dim=0)  # (group_size, time_step_max)
        grouped_index = torch.tensor(index_list)                       # (group_size,)
        
        # 堆叠action_label
        grouped_action_label = {}
        for key, value_list in action_label_list.items():
            # 确保所有值都是tensor
            tensor_list = []
            for v in value_list:
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                tensor_list.append(v)
            grouped_action_label[key] = torch.stack(tensor_list, dim=0)
        
        # 验证该组内所有样本的index值相同
        assert torch.all(grouped_index == idx), f"Index mismatch in group: expected {idx}, got {grouped_index}"
        
        # 添加到分组结果中
        grouped_batch = (
            grouped_prompt_tokens,
            grouped_prompt_masks, 
            grouped_obs_token,
            grouped_obs_mask,
            grouped_pre_action,
            grouped_pre_action_mask,
            grouped_action_label,
            grouped_subprompt_multimodals,
            grouped_subprompt_masks,
            grouped_index
        )
        grouped_batches.append(grouped_batch)
        
        print(f"Created group for index {idx.item()}: {len(group_samples)} samples")
    
    return grouped_batches




def validate_batch_data(batch_data, batch_idx, idx):
    """
    精简的批次数据验证函数，检查关键的数据完整性问题
    
    Args:
        batch_data: 批次数据元组 (prompt_tokens, prompt_masks, obs_token, obs_mask, pre_action, pre_action_mask, action_label, index)
        batch_idx: 批次索引
        idx: 当前处理的index值
    
    Returns:
        bool: True表示数据有效，False表示数据异常
    """
    prompt_tokens, prompt_masks, obs_token, obs_mask, pre_action, pre_action_mask, action_label,subprompt_multimodals,subprompt_masks, index = batch_data
    
    # 检查张量的基本属性
    tensors_to_check = [
        ("prompt_tokens", prompt_tokens),
        ("prompt_masks", prompt_masks), 
        ("obs_token", obs_token),
        ("obs_mask", obs_mask),
        ("subprompt_multimodals", subprompt_multimodals),
        ("subprompt_masks", subprompt_masks),
        ("index", index)
    ]
    
    for name, tensor in tensors_to_check:
        # 检查是否为有效张量
        assert torch.is_tensor(tensor), f"{name} is not a tensor: {type(tensor)}"
        
        # 检查是否包含NaN或Inf
        assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf values"
        
        # 检查形状是否合理
        assert tensor.numel() > 0, f"{name} is empty tensor"
    
    # 检查action_label字典中的数据
    for key, value in action_label.items():
        assert torch.is_tensor(value), f"action_label[{key}] is not tensor: {type(value)}"
        assert torch.isfinite(value).all(), f"action_label[{key}] contains NaN/Inf"
        
        # 检查位置数据的合理范围
        if key in ['pose0_position', 'pose1_position']:
            assert (value >= -10).all() and (value <= 10).all(), f"action_label[{key}] out of reasonable range: {value}"
        
        # 检查旋转数据的合理范围  
        if key in ['pose0_rotation', 'pose1_rotation']:
            assert (value >= -5).all() and (value <= 5).all(), f"action_label[{key}] out of reasonable range: {value}"
    
    # 检查index的一致性
    unique_index = torch.unique(index)
    assert len(unique_index) == 1, f"Inconsistent index in batch: {unique_index}"
    assert unique_index[0].item() == idx, f"Index mismatch: expected {idx}, got {unique_index[0].item()}"
    
    print(f"  ✓ Batch {batch_idx} data validation passed")
    return True


def create_dataloader(data_dir, subprompt_dir, num_start,num_data, policy,time_step_max,num_obj_max,num_prompt_max,batch_size=64, shuffle=True, num_workers=0,device='cuda:0'):
    """
    创建使用自定义collate_fn的DataLoader
    Args:
        data_dir (str): 数据文件夹路径
        num_data (int): 要读取的文件数量
        batch_size (int): 批量大小
        shuffle (bool): 是否打乱数据
        num_workers (int): DataLoader 的工作线程数
    Returns:
        DataLoader: 使用按index分组的数据加载器
    """
    dataset = VIMADataset(data_dir, subprompt_dir, num_start , num_data,policy,time_step_max,num_obj_max,num_prompt_max,device=device)
    
    # 使用自定义的collate_fn进行按index分组
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=collate_by_index  # 关键修改：使用自定义collate函数
    )
    return dataloader

def train_policy(policy, dataloader, cfg, num_epochs):
    """
    训练策略模型
    
    Args:
        policy: 策略模型
        dataloader: 数据加载器
        cfg: 配置参数
        num_epochs: 训练轮数
    """
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    # # 初始化 PromptDecoder
    # prompt_decoder = PromptDecoder(
    #     embed_dim=policy.embed_dim,  # 使用 policy 的 embed_dim
    #     t5_model_name="t5-small",
    #     freeze_t5=True  # 冻结 T5 权重，只训练投影层
    # )
    # prompt_decoder.to(cfg.device)
    # prompt_decoder.train()
    
    # # 将 PromptDecoder 的参数也加入优化器
    # optimizer = torch.optim.Adam([
    #     {'params': policy.parameters()},
    #     {'params': prompt_decoder.parameters()}
    # ], lr=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        # dataloader现在返回的是按index分组的batch列表
        for grouped_batches in dataloader:
            # grouped_batches是一个列表，每个元素是一个同构的子batch
            epoch_batch_loss = 0
            
            # 遍历每个同构的子batch
            for batch_idx, batch_data in enumerate(grouped_batches):
                prompt_tokens, prompt_masks, obs_token, obs_mask, pre_action, pre_action_mask, action_label, subprompt_multimodals, subprompt_masks, index = batch_data
                
                # 验证该batch内所有样本的index相同
                unique_index = torch.unique(index)
                assert len(unique_index) == 1, f"Batch should have same index, got {unique_index}"
                idx = unique_index[0].item()
                
                print(f"Processing batch {batch_idx} with index {idx}, batch_size: {len(index)}")
                
                validate_batch_data(batch_data, batch_idx, idx)
                
                # 移动数据到设备
                prompt_tokens = prompt_tokens.to(cfg.device)     # (B, seq_len, 256)
                prompt_masks = prompt_masks.to(cfg.device)       # (B, seq_len)
                obs_token = obs_token.to(cfg.device)             # (B, time_step_max+1, 1, num_obj*2, 256)
                obs_mask = obs_mask.to(cfg.device)               # (B, time_step_max+1, 1, num_obj*2)
                pre_action = pre_action.to(cfg.device)           # (B, time_step_max, 256)
                pre_action_mask = pre_action_mask.to(cfg.device) # (B, time_step_max)
                index = index.to(cfg.device)                     # (B,)
                subprompt_multimodals = subprompt_multimodals.to(cfg.device)         # (B, time_step_max, subseq_len, 256)
                subprompt_masks = subprompt_masks.to(cfg.device)                     # (B, time_step_max, subseq_len,)
                
                # 移动action_label到设备并确保位置维度正确
                action_label_gpu = {}
                for k, v in action_label.items():
                    v_gpu = v.to(cfg.device)
                    # 确保位置只有2个维度
                    if k in ['pose0_position', 'pose1_position']:
                        if v_gpu.shape[-1] > 2:
                            v_gpu = v_gpu[..., :2]
                    action_label_gpu[k] = v_gpu
                
                # 维度重排以符合transformer输入格式
                obs_token = rearrange(obs_token, 'b l 1 q e -> l b q e')    # (time_step_max+1, B, num_obj*2, 256)
                obs_mask = rearrange(obs_mask, 'b l 1 q -> l b q')          # (time_step_max+1, B, num_obj*2)
                pre_action = pre_action.transpose(0, 1)                     # (time_step_max, B, 256)
                subprompt_multimodals = subprompt_multimodals.transpose(0, 1)
                subprompt_masks = subprompt_masks.transpose(0, 1)
                
                # 截取到实际需要的时间步
                # index值表示要预测第index步的动作，所以需要index+1个观测和index个前续动作
                obs_token_truncated = obs_token[:idx+1]      # (idx+1, B, num_obj*2, 256)
                obs_mask_truncated = obs_mask[:idx+1]        # (idx+1, B, num_obj*2)
                pre_action_truncated = pre_action[:idx]      # (idx, B, 256)
                subprompt_multimodals_truncated = subprompt_multimodals[:idx+1]
                subprompt_masks_truncated = subprompt_masks[:idx+1]
                
                print(f"  After truncation: obs_token {obs_token_truncated.shape}, pre_action {pre_action_truncated.shape}, subprompt_multimodals {subprompt_multimodals_truncated.shape}")

                # if torch.any(obs_mask_truncated == False):
                #     #print(f"  Fixing obs_mask: converting all False values to True")
                #     obs_mask_fixed = torch.ones_like(obs_mask_truncated, dtype=torch.bool)
                # else:
                #     #print(f"  obs_mask is already all True, no fixing needed")
                #     obs_mask_fixed = obs_mask_truncated.bool()  # 确保是bool类型

                
                # 前向传播
                predicted_action_tokens, loss_order, loss_attr, loss_subprompt = policy(
                # predicted_action_tokens = policy.forward(
                    obs_token=obs_token_truncated,                          # (idx+1, B, num_obj*2, 256)
                    action_token=pre_action_truncated,                      # (idx, B, 256)
                    prompt_token=prompt_tokens.transpose(0, 1),             # (seq_len, B, 256)
                    prompt_token_mask=prompt_masks,                         # (B, seq_len)
                    obs_mask=obs_mask_truncated,                           # (idx+1, B, num_obj*2)
                    subprompt_multimodals=subprompt_multimodals_truncated,                           # (idx+1, B, num_obj*2)
                    subprompt_masks=subprompt_masks_truncated,                           # (idx+1, B, num_obj*2)
                )
                # 取最后一个时间步的输出作为预测的动作token
                predicted_action_tokens = predicted_action_tokens[-1]      # (B, 256)
                
                print(f"  Predicted action tokens shape: {predicted_action_tokens.shape}")
                
                # 将目标动作转换为token空间
                target_actions_processed = preprocess_actions(action_label_gpu, cfg)
                
                # 使用no_grad确保目标token不参与梯度计算
                with torch.no_grad():
                    target_action_tokens = policy.forward_action_token_train(target_actions_processed)
                
                # 确保目标token不需要梯度
                target_action_tokens = target_action_tokens.detach()       # (B, 256)
                
                print(f"  Target action tokens shape: {target_action_tokens.shape}")
                
                # 验证维度匹配
                assert predicted_action_tokens.shape == target_action_tokens.shape, \
                    f"Shape mismatch: predicted {predicted_action_tokens.shape} vs target {target_action_tokens.shape}"
                
                # 在token空间计算损失
                loss = criterion(predicted_action_tokens, target_action_tokens)
                epoch_batch_loss += (loss + 0.1*loss_order + + 0.1 * loss_subprompt) #0.1*loss_attr 
                
                print(f"  Batch {batch_idx} loss: {loss.item():.6f}")
            
            # 对整个epoch的所有子batch进行反向传播
            optimizer.zero_grad()
            epoch_batch_loss.backward()
            optimizer.step()
            
            total_loss += epoch_batch_loss.item()
            batch_count += len(grouped_batches)
            
            print(f"Completed epoch batch, total loss: {epoch_batch_loss.item():.6f}")

        # 计算平均损失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(policy.state_dict(), f"/path/to/weights.pt")





    # for epoch in range(num_epochs):
    #     total_loss = 0
    #     for prompt_tokens,prompt_masks,obs_token,obs_mask,pre_action,pre_action_mask,action_label,index in dataloader:
    #         # 移动到设备

    #         prompt_tokens = prompt_tokens.to(cfg.device)
    #         prompt_masks = prompt_masks.to(cfg.device)
    #         obs_token = obs_token.to(cfg.device)
    #         obs_mask = obs_mask.to(cfg.device)
    #         pre_action = pre_action.to(cfg.device)
    #         pre_action_mask = pre_action_mask.to(cfg.device)
    #         index = index.to(cfg.device)
    #         # action_label 里的每个值也要转
    #         action_label = {k: v.to(cfg.device) for k, v in action_label.items()}

    #         # # 通过 PromptDecoder 增强 prompt_tokens
    #         # enhanced_prompt_tokens = prompt_decoder(
    #         #     prompt_tokens=prompt_tokens,
    #         #     obs_token=obs_token,
    #         #     prompt_mask=prompt_masks,
    #         #     obs_mask=obs_mask
    #         # )

    #         index = index.squeeze(-1) if index.dim() > 1 else index
    #         unique_indices = torch.unique(index)
    #         batch_loss = 0
    #         for idx in unique_indices:
    #             mask = (index == idx)
    #             # 分组
    #             prompt_tokens_group = prompt_tokens[mask]
    #             # prompt_tokens_group = enhanced_prompt_tokens[mask]  # 使用增强后的 prompt_tokens
    #             prompt_masks_group = prompt_masks[mask]
    #             obs_token_group = obs_token[mask]
    #             obs_mask_group = obs_mask[mask]
    #             pre_action_group = pre_action[mask]
    #             pre_action_mask_group = pre_action_mask[mask]
    #             action_label_group = {k: v[mask] for k, v in action_label.items()}

    #             # 维度重排
    #             obs_token_group = rearrange(obs_token_group, 'b l 1 q e -> l b q e')
    #             obs_mask_group = rearrange(obs_mask_group, 'b l 1 q -> l b q')
    #             pre_action_group = pre_action_group.transpose(0, 1)

    #             obs_token_group = obs_token_group[:idx+1] 
    #             obs_mask_group = obs_mask_group[:idx+1]
    #             pre_action_group = pre_action_group[:idx]
                

    #             # 前向
    #             predicted_action_tokens = policy.forward( #这里的L不代表length，而是代表时间步数
    #                 obs_token=obs_token_group,#L,B,Q,E
    #                 action_token=pre_action_group,#(L, B, E)
    #                 prompt_token=prompt_tokens_group.transpose(0, 1),#(Length, B, E)
    #                 prompt_token_mask=prompt_masks_group,#(B, Length),这个地方token和token_mask形状的排列顺序不一样，需要注意！
    #                 obs_mask=obs_mask_group,#L,B,Q,
    #             )
    #             predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(0)  # (1, B, E)

    #             # 解码
    #             dist_dict = policy.forward_action_decoder(predicted_action_tokens)
    #             predicted_actions = {k: v.mode() for k, v in dist_dict.items()}
    #             predicted_actions = policy._de_discretize_actions(predicted_actions)
    #             predicted_actions = postprocess_actions(predicted_actions, cfg)#照抄的，仅有略微改动，原本用的是np,这用的是tensor，所以某些地方后面添加了.cpu().numpy()

    #             # 计算loss
    #             action_keys = ["pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation"]
    #             loss = None
    #             for key in action_keys:
    #                 if key in predicted_actions and key in action_label_group:
    #                     pred = predicted_actions[key][0]
    #                     target = action_label_group[key]
    #                     if pred.shape[1] == 2:
    #                         pred = pred[..., :2]
    #                         target = target[..., :2]
    #                     if not torch.is_tensor(pred):
    #                         pred = torch.tensor(pred, dtype=torch.float32, device=cfg.device)
    #                     if not torch.is_tensor(target):
    #                         target = torch.tensor(target, dtype=torch.float32, device=cfg.device)
    #                     pred = pred.requires_grad_()
    #                     this_loss = criterion(pred, target)
    #                     if loss is None:
    #                         loss = this_loss
    #                     else:
    #                         loss = loss + this_loss
    #             batch_loss += loss


    #         # 反向传播
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     avg_loss = total_loss / len(dataloader)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    #     # 保存模型
    #     if (epoch + 1) % 1000 == 0:
    #         torch.save(policy.state_dict(), f"/path/to/weights.pt")
    #     # if (epoch + 1) % 10 == 0:
    #     #     torch.save({
    #     #         'policy_state_dict': policy.state_dict(),
    #     #         'prompt_decoder_state_dict': prompt_decoder.state_dict(),
    #     #         'optimizer_state_dict': optimizer.state_dict(),
    #     #         'epoch': epoch,
    #     #         'loss': avg_loss,
    #     #     }, f"/path/to/weights.pth")

# def train_policy_batch2(policy, dataloader, cfg, num_epochs=10):
#     """
#     训练 VIMA 策略模型，支持分batch
#     Args:
#         policy: 策略模型
#         dataloader: 数据加载器
#         cfg: 配置参数
#         num_epochs: 训练轮数
#     """
    

#     # for name, param in policy.named_parameters():
#     #     # param.requires_grad = False
#     #     # if not name.startswith("test_layer"):
#     #     if "test_layer" not in name:
#     #         param.requires_grad = False

    
#     policy.train()
#     optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
#     # optimizer = torch.optim.Adam(
#     #     filter(lambda p: p.requires_grad, policy.parameters()), 
#     #     lr=1e-4
#     # )
#     criterion = torch.nn.MSELoss()

#     # print("Trainable parameters:")
#     # for name, param in policy.named_parameters():
#     #     if param.requires_grad:
#     #         print(" -", name)



#     for epoch in range(num_epochs):
#         total_loss = 0
#         for prompt_tokens,prompt_masks,obs_token,obs_mask,pre_action,pre_action_mask,action_label,index in dataloader:
#             # 移动到设备
            
            

#             prompt_tokens = prompt_tokens.to(cfg.device)
#             prompt_masks = prompt_masks.to(cfg.device)
#             obs_token = obs_token.to(cfg.device)
#             obs_mask = obs_mask.to(cfg.device)
#             pre_action = pre_action.to(cfg.device)
#             pre_action_mask = pre_action_mask.to(cfg.device)
#             index = index.to(cfg.device)
#             # action_label 里的每个值也要转
#             action_label = {k: v.to(cfg.device) for k, v in action_label.items()}


#             index = index.squeeze(-1) if index.dim() > 1 else index
#             unique_indices = torch.unique(index)
#             # batch_loss = 0
#             loss = 0
#             for idx in unique_indices:
#                 mask = (index == idx)
#                 # 分组
#                 prompt_tokens_group = prompt_tokens[mask]
#                 prompt_masks_group = prompt_masks[mask]
#                 obs_token_group = obs_token[mask]
#                 obs_mask_group = obs_mask[mask]
#                 pre_action_group = pre_action[mask]
#                 pre_action_mask_group = pre_action_mask[mask]
#                 action_label_group = {k: v[mask] for k, v in action_label.items()}

#                 # 维度重排
#                 obs_token_group = rearrange(obs_token_group, 'b l 1 q e -> l b q e')
#                 obs_mask_group = rearrange(obs_mask_group, 'b l 1 q -> l b q')
#                 pre_action_group = pre_action_group.transpose(0, 1)

#                 obs_token_group = obs_token_group[:idx+1] 
#                 obs_mask_group = obs_mask_group[:idx+1]
#                 pre_action_group = pre_action_group[:idx]
                

#                 # 前向
#                 predicted_action_tokens = policy.forward( #这里的L不代表length，而是代表时间步数
#                     obs_token=obs_token_group,#L,B,Q,E
#                     action_token=pre_action_group,#(L, B, E)
#                     prompt_token=prompt_tokens_group.transpose(0, 1),#(Length, B, E)
#                     prompt_token_mask=prompt_masks_group,#(B, Length),这个地方token和token_mask形状的排列顺序不一样，需要注意！
#                     obs_mask=obs_mask_group,#L,B,Q,
#                 )
#                 predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(0)  # (1, B, E)

#                 # 解码
#                 dist_dict = policy.forward_action_decoder(predicted_action_tokens)
#                 predicted_actions = {k: v.mode() for k, v in dist_dict.items()}
#                 predicted_actions = policy._de_discretize_actions(predicted_actions)
#                 predicted_actions = postprocess_actions(predicted_actions, cfg)#照抄的，仅有略微改动，原本用的是np,这用的是tensor，所以某些地方后面添加了.cpu().numpy()

#                 # 计算loss
#                 action_keys = ["pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation"]
#                 # loss = None
                
#                 for key in action_keys:
#                     if key in predicted_actions and key in action_label_group:
#                         pred = predicted_actions[key][0]
#                         target = action_label_group[key]
#                         # if pred.shape[1] == 2:
#                         #     pred = pred[..., :2]
#                         #     target = target[..., :2]
#                         target = target[..., :pred.shape[-1]]
#                         if not torch.is_tensor(pred):
#                             pred = torch.tensor(pred, dtype=torch.float32, device=cfg.device)
#                         if not torch.is_tensor(target):
#                             target = torch.tensor(target, dtype=torch.float32, device=cfg.device)
#                         # pred = pred.requires_grad_()
#                         this_loss = criterion(pred, target)
#                         loss = loss + this_loss
#                         # if loss is None:
#                         #     loss = this_loss
#                         # else:
#                         #     loss += this_loss
#                 # batch_loss += loss



#             # 反向传播

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # for name, param in policy.named_parameters():
#             #     if param.grad is not None:
#             #         print(f"{name}: grad norm = {param.grad.norm().item()}")
#             #     else:
#             #         print(f"{name}: NO GRAD")


#             total_loss += loss.item()

#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        


#         # 保存模型
#         if (epoch + 1) % 1000 == 0:
#             torch.save(policy.state_dict(), f"/path/to/weights.pt")

def main(cfg):
    """
    主函数：训练 VIMA 策略
    Args:
        cfg: 配置参数
    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载预训练策略模型
    print("Loading policy...")
    policy = create_policy_from_ckpt(cfg.ckpt, cfg.device)

    # 创建 DataLoader
    data_dir = cfg.data_dir
    num_data = cfg.num_data
    num_start = cfg.num_start
    subprompt_dir = cfg.subprompt_dir
    task = cfg.task
    with open(cfg.config_dir, 'r') as f:
        padding_stats = json.load(f)
    # 取出与 task 对应的内容
    with open(cfg.prompt_max_dir, 'r') as f:
        prompt_max_stats = json.load(f)
    task_stats = padding_stats.get(cfg.task)
    prompt_max_stats = prompt_max_stats.get(cfg.task)
    print(f"Stats for task '{cfg.task}': {task_stats}")
    time_step_max = task_stats['rgb_padding'] - 1#最大的时间步数
    num_obj_max = task_stats['max_segm_front']#最大的物体数
    num_prompt_max = prompt_max_stats['max_prompt_length']#prompt的最大长度
    dataloader = create_dataloader(
        data_dir, subprompt_dir, num_start, num_data, policy, 
        time_step_max=time_step_max,
        num_obj_max=num_obj_max,
        num_prompt_max=num_prompt_max, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0,
        device=cfg.device
    )
    # 训练模型
    train_policy(policy, dataloader, cfg, num_epochs=10)

def parse_args():
    parser = argparse.ArgumentParser(description="Run VIMA task inference or analysis.")

    # ===== 基本参数 =====
    parser.add_argument("--task", type=str, default="rotate", help="任务名称，如 follow_order, twist 等。")
    parser.add_argument("--device", type=str, default="cuda:2", help="运行GPU设备。")

    # ===== 路径参数 =====
    #base_dir = "/home/hjy/zhouleyu"
    parser.add_argument("--ckpt", type=str, default="/path/to/workspacevima_code/configs/2M.ckpt", help="模型权重路径。")
    parser.add_argument("--config_dir", type=str, default="/path/to/workspacevima_code/configs/padding_stats.json", help="任务最大物体数、最大时间步配置文件。")
    parser.add_argument("--prompt_max_dir", type=str, default="/path/to/workspacevima_code/configs/padding_prompt_length.json", help="任务最大 prompt 长度配置文件。")
    parser.add_argument("--data_root", type=str, default="/path/to/workspaceVIMA-Data/packed_zips", help="数据根目录。")
    parser.add_argument("--subprompt_root", type=str, default="/path/to/workspacevima_code/subprompt_out", help="subprompt目录。")

    # ===== 数据加载参数 =====
    parser.add_argument("--num_data", type=int, default=64, help="每次读取的数据数量。")
    # parser.add_argument("--num_start", type=int, default=22890, help="数据起始索引。")
    parser.add_argument("--num_start", type=int, default=0, help="数据起始索引。")

    args = parser.parse_args()
    # args.num_start = 22980

    # ===== 自动拼接具体任务路径 =====
    args.data_dir = os.path.join(args.data_root, args.task)
    args.subprompt_dir = os.path.join(args.subprompt_root, args.task)

    return args

if __name__ == "__main__":
    # arg = argparse.ArgumentParser()
    # # arg.add_argument("--task", type=str, default="follow_order")
    # arg.add_argument("--task", type=str, default="twist")
    # # arg.add_argument("--ckpt", type=str, required=True)
    # arg.add_argument("--ckpt", type=str, default='/home/hjy/zhouleyu/vima/VIMA/scripts/2M.ckpt')
    # # arg.add_argument("--data_dir", type=str, default="/home/hjy/zhouleyu/vima_v6_preprocessed/twist")#任务的路径
    # arg.add_argument("--config_dir",type=str,default = "/home/hjy/zhouleyu/padding_stats.json")#这个东西代表每个任务的最大值->最大物体数，最大时间步数
    # arg.add_argument('--prompt_max_dir', type=str, default="/home/hjy/zhouleyu/padding_prompt_length.json")#这个东西代表每个任务的最大prompt长度
    # arg.add_argument('--num_data', type=int, default=128)#这个是每次读取多少个数据
    # arg.add_argument('--num_start', type=int, default=0)#每次读取多少个数据的开始值
    # arg.add_argument("--device", default="cuda:0")
    # arg = arg.parse_args()
    # arg.data_dir = "/home/hjy/zhouleyu/vima_v6_preprocessed/"+arg.task
    # main(arg)
    args = parse_args()
    main(args)





    