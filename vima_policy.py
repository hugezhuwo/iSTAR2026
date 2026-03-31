from __future__ import annotations
import os
import json
import torch
import torch.nn as nn
from einops import rearrange
import sys
import vima.nn as vnn
from vima.utils import *
# import xattn_gpt as vnn2 # 后续可能要注释掉，不然会有问题
import copy
from nlir_decomposer import NLIRDecomposer
from nlir_decomposer_frozeT5 import NLIRDecomposer_t5
from graph_dynamic_pe import GraphDynamicPE
from typing import Optional

class VIMAPolicy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        xf_n_layers: int,
        sattn_n_heads: int,
        xattn_n_heads: int,
        nlir_decomposer = "NLIRDecomposer_t5",
        is_pe = True,
        save_dir="./"
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.is_pe = is_pe
        # self.xattn_gpt = vnn2.XAttnGPT( #model
        self.xattn_gpt = vnn.XAttnGPT( #model
            embed_dim,
            n_layer=xf_n_layers,
            n_head=sattn_n_heads,
            dropout=0.1,
            xattn_n_head=xattn_n_heads,
            xattn_ff_expanding=4,
            xattn_n_positions=256,
            use_geglu=True,
        )
        self.subtask_projection = nn.Linear(
            embed_dim, 
            768
        )
        if nlir_decomposer == "NLIRDecomposer":
            self.nlir_decomposer = NLIRDecomposer(embed_dim=embed_dim, t5_model_name="t5-base", freeze_t5=False)
            print("nlir_decomposer == NLIRDecomposer")
        elif nlir_decomposer == "NLIRDecomposer_t5":
            decomposer = NLIRDecomposer_t5(
                freeze_t5=True,
                t5_model_name="t5-small",
            )

            num_enc = decomposer.t5_model.config.num_layers  # 或 len(decomposer.t5_model.encoder.block)
            self.nlir_decomposer = NLIRDecomposer_t5(
                embed_dim=embed_dim,                # 嵌入维度
                t5_model_name="t5-base",            # 使用 t5-base 模型
                freeze_t5=True,                     # 冻结 T5 的大部分参数
                vima_vocab_size=32142,              # VIMA 词表大小
                t5_train_encoder_layers=list(range(num_enc)),       # 解冻 T5 编码器的最后一层
                t5_train_decoder_layers=[-1],       # 解冻 T5 解码器的最后一层
                t5_train_shared_embeddings=False,   # 不训练共享嵌入
                t5_train_lm_head=False,             # 不训练语言模型头
            )
            # 记录参数到字典
            self.nlir_decomposer_config = {
                "embed_dim": embed_dim,
                "t5_model_name": "t5-base",
                "freeze_t5": True,
                "vima_vocab_size": 32142,
                "t5_train_encoder_layers": list(range(num_enc)),
                "t5_train_decoder_layers": [-1],
                "t5_train_shared_embeddings": False,
                "t5_train_lm_head": False,
            }


            print("nlir_decomposer == NLIRDecomposer_t5")#

            # 保存参数到 JSON 文件
            # json_path = os.path.join(save_dir, "nlir_decomposer_t5_config.json")
            # with open(json_path, "w") as f:
            #     json.dump(self.nlir_decomposer_config, f, indent=4)
            # print(f"NLIRDecomposer_t5 config saved to {json_path}")
        else:
            print("nlir选择错了")
            sys.exit()
            

        if self.is_pe:
            self.graph_pe = GraphDynamicPE(embed_dim=768, n_heads=4)

        
        self.sent_decomposer = copy.deepcopy(self.xattn_gpt)

        self.action_generator = copy.deepcopy(self.xattn_gpt)
        for p in self.action_generator.parameters():
            p.requires_grad = False

        self.obj_encoder = vnn.ObjEncoder( #视觉编码器
            transformer_emb_dim=embed_dim,
            views=["front", "top"],
            vit_output_dim=768,
            vit_resolution=32,
            vit_patch_size=16,
            vit_width=768,
            vit_layers=4,
            vit_heads=24,
            bbox_mlp_hidden_dim=768,
            bbox_mlp_hidden_depth=2,
        )

        self.end_effector_encoder = vnn.Embedding(num_embeddings=2, embedding_dim=2)#末端执行，给索引，输出词的embedding

        self.obs_fusion_layer = nn.Linear(self.obj_encoder.output_dim + 2, embed_dim)#把视觉和末端执行统一维度

        self.action_encoder = vnn.ActionEmbedding(#把不同的动作进行综合embedding->进行把结果拼接在一起
            output_dim=embed_dim,
            embed_dict={
                "pose0_position": vnn.ContinuousActionEmbedding( #把连续动作编码成特征
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose0_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
            },
        )
        self.action_decoder = vnn.ActionDecoder(#连续的token变成离散的动作
            input_dim=embed_dim,
            action_dims={
                "pose0_position": [50, 100],
                "pose0_rotation": [50] * 4,
                "pose1_position": [50, 100],
                "pose1_rotation": [50] * 4,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )

        self.prompt_embedding = vnn.WordEmbedding()#单词的encoder
        self.t5_prompt_encoder = vnn.T5PromptEncoder()#
        self.t5_prompt_encoder_post_layer = (
            nn.Identity()
            if embed_dim == self.t5_prompt_encoder.output_dim
            else nn.Linear(self.t5_prompt_encoder.output_dim, embed_dim, bias=False)
        )#用来统一维度

        # self.test_layer = nn.Linear(embed_dim, embed_dim)   

        

        self.prompt_obj_post_layer = vnn.build_mlp(
            self.obj_encoder.output_dim,
            hidden_dim=768,
            output_dim=768,
            hidden_depth=2,
        )


                

        # ... existing code ...

        # =========================================================
        # 1. 明确定义需要冻结的模块 (VIMA Backbone & Shared Modules)
        # =========================================================
        modules_to_freeze = [
            self.xattn_gpt,                  # VIMA 主干 Transformer
            self.obj_encoder,                # 视觉编码器 (Shared)
            self.end_effector_encoder,       # 末端执行器编码 (Shared)
            self.obs_fusion_layer,           # 融合层 (Shared)
            self.prompt_embedding,           # 词向量 (Shared)
            self.prompt_obj_post_layer,      # 视觉后处理 (Shared)
            
            # --- 之前遗漏的关键共享模块 ---
            self.t5_prompt_encoder,          # T5 编码器 (CRITICAL SHARED)
            self.t5_prompt_encoder_post_layer, # T5 后处理 (CRITICAL SHARED)
            self.action_encoder,             # 动作编码 (Shared)
            self.action_decoder,             # 动作解码 (Shared)
            
            # --- Subprompt 路径中的固定参考模块 ---
            self.action_generator,           # 它是 xattn_gpt 的副本，作为固定的动作生成器用于 teacher-forcing
        ]

        # 执行冻结
        print("Freezing VIMA backbone and shared modules...")
        for m in modules_to_freeze:
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
        
        # =========================================================
        # 2. 明确定义需要训练的模块 (Subprompt Exclusive)
        # =========================================================
        # 这一步不是必须的（因为剩下默认就是 True），但为了打印确认，写在这里
        modules_to_train = [
            self.sent_decomposer,            # 负责将 obs/action 分解为子任务隐变量
            self.subtask_projection,         # 线性层
            self.nlir_decomposer,            # 生成子任务 prompt/embedding
        ]
        if self.is_pe:
            modules_to_train.append(self.graph_pe) # 图位置编码

        # 打印确认 (Debug用)
        print("Modules allowed to train (Subprompt):")
        for m in modules_to_train:
            # 确保它们是开启梯度的
            for p in m.parameters():
                p.requires_grad = True 
            print(f" - {type(m).__name__}")

        self._views = ["front", "top"]
        # ... existing code ...

        self._views = ["front", "top"]#两种视角
        self._n_discrete_x_bins = 50
        self._n_discrete_y_bins = 100
        self._n_discrete_z_bins = 50
        self._n_discrete_rot_bins = 50
        self.default_prompt_token = nn.Parameter(torch.zeros(1, self.embed_dim))

    

    def forward_gate(
        self,
        obs_token: torch.Tensor,               # [L_obs, B, Q, E]
        obs_mask: torch.Tensor,                # [L_obs, B, Q]
        action_token: torch.Tensor | None,     # [L_act, B, E] 或 None（历史动作）
        prompt_token: torch.Tensor,            # [L_prompt, B, E]
        prompt_token_mask: torch.Tensor,       # [B, L_prompt]
        subprompt_multimodals: torch.Tensor,   # [L_obs, B, S, 768]
        subprompt_masks: torch.Tensor,         # [L_obs, B, S]
        gate: float | torch.Tensor = 0.5,      # 融合系数：gate * vima + (1-gate) * subprompt(+residual)
    ):
        device = obs_token.device
        dtype  = obs_token.dtype
        L_obs, B, Q, E = obs_token.shape
        assert E == self.embed_dim, f"embed_dim mismatch: {E} vs {self.embed_dim}"
    
        # ===============================
        # A) 交错序列（固定 step = Q+1）
        # ===============================
        step = Q + 1
        L_total = L_obs * step
    
        # 观测展平到交错序列的布局
        obs_token_r = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token_r = rearrange(obs_token_r, "B L Q E -> B (L Q) E")
        obs_token_r = rearrange(obs_token_r, "B L E -> L B E")                     # [L_obs*Q, B, E]
    
        obs_mask_r  = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask_r  = rearrange(obs_mask_r, "B L Q -> B (L Q)")
        obs_mask_r  = rearrange(obs_mask_r, "B L -> L B")                          # [L_obs*Q, B]
    
        # 历史动作：对齐到每个时间步的“动作位”
        if action_token is not None and action_token.numel() > 0:
            L_act = action_token.shape[0]
            action_in      = torch.zeros(L_obs, B, E, device=device, dtype=dtype)
            action_in_mask = torch.zeros(L_obs, B, dtype=torch.bool, device=device)
            take = min(L_act, L_obs)
            action_in[:take] = action_token[:take]
            action_in_mask[:take] = True
        else:
            action_in      = torch.zeros(L_obs, B, E, device=device, dtype=dtype)
            action_in_mask = torch.zeros(L_obs, B, dtype=torch.bool, device=device)
    
        # 拼交错序列（obs 位于每步 0..Q-1；动作位于每步 Q）
        seq_tokens = torch.zeros(L_total, B, E, device=device, dtype=dtype)
        seq_masks  = torch.zeros(L_total, B, dtype=torch.bool, device=device)
    
        for q in range(Q):
            seq_tokens[q::step] = obs_token_r[q::Q]     # 每步第 q 个对象位
            seq_masks[q::step]  = obs_mask_r[q::Q]
    
        seq_tokens[Q::step] = action_in                 # 每步的动作位（可能全 0）
        seq_masks[Q::step]  = action_in_mask
    
        # 位置 id
        position_ids = torch.cumsum(seq_masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1
    
        # ===============================
        # B) VIMA 主干（原版路径）
        # ===============================
        for p in self.xattn_gpt.parameters():
            p.requires_grad = False
        with torch.no_grad():
            tokens_out_vima = self.xattn_gpt(
                obs_action_tokens=seq_tokens,                          # [L_total, B, E]
                prompt_tokens=prompt_token,                            # [L_prompt, B, E]
                prompt_mask=prompt_token_mask,                         # [B, L_prompt]
                obs_action_masks=seq_masks.transpose(0, 1),            # [B, L_total]
                obs_action_position_ids=position_ids.transpose(0, 1),  # [B, L_total]
                prompt_position_ids=prompt_position_ids,               # [B, L_prompt]
            )
        # 取各时间步的“动作输出位”——与 forward_test_res 保持一致：最后一个对象位（Q-1）
        vima_out = tokens_out_vima[Q - 1 :: step]                 # [L_obs, B, E]
    
        # ============================================
        # C) 子提示增强路径（保持你现有 forward 逻辑）
        # ============================================
        # sent_decomposer：序列头 pad 一步；reshape 前去掉 pad（保证 [L_obs*(Q+1), B, E] 可重排）
        pad = torch.zeros(1, B, E, device=device, dtype=dtype)
        seq_masks_padded   = torch.cat([seq_masks[0:1], seq_masks], dim=0)                 # [L_total+1, B]
        position_ids_padded = torch.cumsum(seq_masks_padded, dim=0) - 1
        position_ids_padded = position_ids_padded.long()
    
        subtask_tokens = self.sent_decomposer(
            obs_action_tokens=torch.cat([pad, seq_tokens], dim=0),                         # [L_total+1, B, E]
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
            obs_action_masks=seq_masks_padded.transpose(0, 1),
            obs_action_position_ids=position_ids_padded.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )                                                                                  # [L_total+1, B, E]
    
        # 残差：冻结 action_generator 最后一步输出
        with torch.no_grad():
            subtask_tokens_residual = self.action_generator(
                obs_action_tokens=torch.cat([pad, seq_tokens], dim=0),
                prompt_tokens=prompt_token,
                prompt_mask=prompt_token_mask,
                obs_action_masks=seq_masks_padded.transpose(0, 1),
                obs_action_position_ids=position_ids_padded.transpose(0, 1),
                prompt_position_ids=prompt_position_ids,
            )
        predicted_action_tokens_residual = subtask_tokens_residual[-1]                     # [B, E]
    
        # 去掉 pad 再重排为 [n_obj+1, L_obs*B, E]，方便做 NLIR/PE
        subtask_wo_pad = subtask_tokens[1:]                                                # [L_total, B, E] = L_obs*(Q+1)
        subtask_tokens_out = subtask_wo_pad.reshape(L_obs, step, B, E).transpose(0, 1)     # [Q+1, L_obs, B, E]
        subtask_tokens_out = subtask_tokens_out.reshape(step, L_obs * B, E)                # [Q+1, L_obs*B, E]
        subtask_tokens_out = self.subtask_projection(subtask_tokens_out)                   # [Q+1, L_obs*B, 768]
    
        # Graph Dynamic PE（可选）
        if getattr(self, "is_pe", True):
            S = subprompt_multimodals.shape[2]
            subprompt_tokens_768 = subprompt_multimodals.reshape(-1, S, 768).transpose(0, 1)  # [S, L_obs*B, 768]
            subprompt_masks_SxB  = subprompt_masks.reshape(-1, S).transpose(0, 1)             # [S, L_obs*B]
            subtask_tokens_out, gated_node, gated_node_pe, loss_order, loss_attr, alpha = self.graph_pe(
                subtask_tokens_out, subprompt_tokens_768, L_obs=L_obs, B=B
            )
        else:
            loss_order = torch.zeros((), device=device)
            loss_attr  = torch.zeros((), device=device)
    
        # NLIR（teacher forcing）与重建损失
        S = subprompt_multimodals.shape[2]
        mm_tf   = subprompt_multimodals.reshape(-1, S, 768).transpose(0, 1)                 # [S, L_obs*B, 768]
        mask_tf = subprompt_masks.reshape(-1, S).transpose(0, 1)                             # [S, L_obs*B]
        subprompt_raw, subprompt_pred = self.nlir_decomposer(subtask_tokens_out, mm_tf, mask_tf)  # [L_obs*B, S, 768]
    
        with torch.no_grad():
            target_subprompt = mm_tf.transpose(0, 1)                                        # [L_obs*B, S, 768]
            target_mask = mask_tf.transpose(0, 1).unsqueeze(-1)                              # [L_obs*B, S, 1]
        cos_sim = torch.nn.functional.cosine_similarity(subprompt_raw, target_subprompt, dim=-1)
        loss_subprompt = ((1.0 - cos_sim).unsqueeze(-1) * target_mask).sum() / target_mask.sum().clamp_min(1.0)
    
        # T5 prompt encoder 回到策略维度 E
        prompt_emb = subprompt_raw.transpose(0, 1)                                          # [S, L_obs*B, 768]
        prompt_emb = self.t5_prompt_encoder(prompt_emb, attention_mask=mask_tf.transpose(0, 1), batch_first=False)
        prompt_emb = self.t5_prompt_encoder_post_layer(prompt_emb)                           # [S, L_obs*B, E]
    
        # 展平观测到 B' = L_obs*B，供 action_generator 使用
        obs_token_flat = obs_token.permute(2, 0, 1, 3).contiguous().view(Q, L_obs * B, E)   # [Q, B', E]
        obs_mask_flat  = obs_mask.permute(2, 0, 1).contiguous().view(Q, L_obs * B)          # [Q, B']
        obs_pos_ids    = (torch.cumsum(obs_mask_flat, dim=0) - 1).long()                    # [Q, B']
        prompt_pos_ids = (torch.cumsum(mask_tf, dim=0) - 1).long()                          # [S, L_obs*B]
    
        tokens_out_sub = self.action_generator(
            obs_action_tokens=obs_token_flat,                           # [Q, B', E]
            prompt_tokens=prompt_emb,                                   # [S, B', E]
            prompt_mask=mask_tf.transpose(0, 1),                        # [B', S]
            obs_action_masks=obs_mask_flat.bool().transpose(0, 1),      # [B', Q]
            obs_action_position_ids=obs_pos_ids.transpose(0, 1),        # [B', Q]
            prompt_position_ids=prompt_pos_ids.transpose(0, 1),         # [B', S]
        )
        pred_sub_last = tokens_out_sub[-1]                               # [B', E], B' = L_obs*B
    
        # 残差对齐并相加
        residual_rep = predicted_action_tokens_residual.unsqueeze(1).repeat(1, L_obs, 1).reshape(B * L_obs, E)  # [B', E]
        subprompt_out = (pred_sub_last + residual_rep).view(B, L_obs, E).permute(1, 0, 2).contiguous()          # [L_obs, B, E]
    
        # ===============================
        # D) 最终加权融合
        # ===============================
        if not torch.is_tensor(gate):
            gate_t = torch.tensor(float(gate), device=device, dtype=vima_out.dtype)
        else:
            gate_t = gate.to(device=device, dtype=vima_out.dtype)
        gate_t = torch.clamp(gate_t, 0.0, 1.0)
        while gate_t.dim() < 3:
            gate_t = gate_t.unsqueeze(-1)  # 支持标量或 [L_obs,B,1] 广播
    
        fused = gate_t * vima_out + (1.0 - gate_t) * subprompt_out       # [L_obs, B, E]
        return fused, loss_order, loss_attr, loss_subprompt
   

    def forward_test_res_gate(
        self,
        obs_token: torch.Tensor,            # [L_obs, B, Q, E]
        obs_mask: torch.Tensor,             # [L_obs, B, Q]
        action_token: Optional[torch.Tensor],  # [L_act, B, E] 或 None
        prompt_token: torch.Tensor,         # [L_prompt, B, E]
        prompt_token_mask: torch.Tensor,    # [B, L_prompt]
        subprompt_multimodals: Optional[torch.Tensor] = None,  # [L_obs,B,S,768] 或 None
        subprompt_masks: Optional[torch.Tensor] = None,        # [L_obs,B,S] 或 None
        gen_len: int = 15,
        gate=0.95,
    ):
        """
        与 forward_gate 对齐的 test 前向（保持 forward_test_res 行为并加 gate 融合）：
        - vima 主干：交错序列 -> xattn_gpt -> 取动作位
        - subprompt 分支：严格复用 forward_test_res（sent_decomposer -> subtask_projection -> NLIR -> T5 -> action_generator）
        - 残差与 subprompt_out 相加，最终 fused = gate*vima_out + (1-gate)*subprompt_out
        返回: fused [L_obs, B, E], loss_order, loss_attr, loss_subprompt
        """
        device = obs_token.device
        dtype = obs_token.dtype
        L_obs, B, Q, E = obs_token.shape
        assert E == self.embed_dim

        # build interleaved sequence (step = Q + 1)
        step = Q + 1
        L_total = L_obs * step

        obs_token_r = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token_r = rearrange(obs_token_r, "B L Q E -> B (L Q) E")
        obs_token_r = rearrange(obs_token_r, "B L E -> L B E")                     # [L_obs*Q, B, E]

        obs_mask_r  = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask_r  = rearrange(obs_mask_r, "B L Q -> B (L Q)")
        obs_mask_r  = rearrange(obs_mask_r, "B L -> L B")                          # [L_obs*Q, B]

        # align historical actions to per-step action slots
        if action_token is not None and action_token.numel() > 0:
            L_act = action_token.shape[0]
            action_in      = torch.zeros(L_obs, B, E, device=device, dtype=dtype)
            action_in_mask = torch.zeros(L_obs, B, dtype=torch.bool, device=device)
            take = min(L_act, L_obs)
            action_in[:take] = action_token[:take]
            action_in_mask[:take] = True
        else:
            action_in      = torch.zeros(L_obs, B, E, device=device, dtype=dtype)
            action_in_mask = torch.zeros(L_obs, B, dtype=torch.bool, device=device)

        seq_tokens = torch.zeros(L_total, B, E, device=device, dtype=dtype)
        seq_masks  = torch.zeros(L_total, B, dtype=torch.bool, device=device)

        for q in range(Q):
            seq_tokens[q::step] = obs_token_r[q::Q]
            seq_masks[q::step]  = obs_mask_r[q::Q]
        seq_tokens[Q::step] = action_in
        seq_masks[Q::step]  = action_in_mask

        position_ids = torch.cumsum(seq_masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1

        # -----------------------
        # subtask / residual path (mirror forward_test_res)
        # -----------------------
        pad = torch.zeros(1, B, E, device=device, dtype=dtype)
        seq_masks_padded = torch.cat([seq_masks[0:1], seq_masks], dim=0)                 # [L_total+1, B]
        position_ids_padded = torch.cumsum(seq_masks_padded, dim=0) - 1
        position_ids_padded = position_ids_padded.long()

        sent_out_raw = self.sent_decomposer(
            obs_action_tokens=torch.cat([pad, seq_tokens], dim=0),
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
            obs_action_masks=seq_masks_padded.transpose(0, 1),
            obs_action_position_ids=position_ids_padded.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )  # [L_total+1, B, E]

        predicted_action_tokens_residual = sent_out_raw[-1]  # [B, E]

        # -------- 修复：去掉 pad 后再 reshape（与 forward_gate/forward_test_res 对齐） ----------
        subtask_wo_pad = sent_out_raw[1:]                                 # [L_total, B, E] == [L_obs*step, B, E]
        assert subtask_wo_pad.shape[0] == L_obs * step, f"len mismatch {subtask_wo_pad.shape[0]} vs {L_obs*step}"
        sent_out = subtask_wo_pad.reshape(L_obs, step, B, E).transpose(0, 1)   # [step, L_obs, B, E]
        sent_out = sent_out.reshape(step, L_obs * B, E)                        # [step, L_obs*B, E]
        sent_out = self.subtask_projection(sent_out)   

        # NLIR: teacher forcing or forward_test
        have_subprompt = (
            (subprompt_multimodals is not None)
            and (subprompt_multimodals.numel() > 0)
            and (subprompt_multimodals.shape[2] > 0)
        )

        if have_subprompt:
            S = subprompt_multimodals.shape[2]
            mm_tf   = subprompt_multimodals.reshape(-1, S, 768).transpose(0, 1)   # [S, L_obs*B, 768]
            mask_tf = subprompt_masks.reshape(-1, S).transpose(0, 1)              # [S, L_obs*B]
            subtask_prompt_embedding_raw, subtask_prompt_predictions = self.nlir_decomposer(
                sent_out, mm_tf, mask_tf
            )  # [L_obs*B, S, 768]
            prompt_mask_for_attn = mask_tf.transpose(0, 1)                         # [L_obs*B, S]
        else:
            subtask_prompt_embedding_raw, _ ,_ = self.nlir_decomposer.forward_test(
                sent_out, subtask_decoder_input=None, max_gen_steps=gen_len
            )  # [L_obs*B, gen_len, 768]
            S = subtask_prompt_embedding_raw.shape[1]
            prompt_mask_for_attn = torch.ones(L_obs * B, S, dtype=torch.bool, device=device)

        # compute loss_subprompt when teacher-forcing
        if have_subprompt:
            with torch.no_grad():
                target_subprompt = mm_tf.transpose(0, 1)                    # [L_obs*B, S, 768]
                target_mask = mask_tf.transpose(0, 1).unsqueeze(-1)         # [L_obs*B, S, 1]
            cos_sim = torch.nn.functional.cosine_similarity(subtask_prompt_embedding_raw, target_subprompt, dim=-1)
            loss_subprompt = ((1.0 - cos_sim).unsqueeze(-1) * target_mask).sum() / target_mask.sum().clamp_min(1.0)
        else:
            loss_subprompt = torch.zeros((), device=device)

        # T5 encode -> strategy dim
        prompt_emb = subtask_prompt_embedding_raw.transpose(0, 1)  # [S, L_obs*B, 768]
        prompt_emb = self.t5_prompt_encoder(prompt_emb, attention_mask=prompt_mask_for_attn, batch_first=False)
        prompt_emb = self.t5_prompt_encoder_post_layer(prompt_emb)                # [S, L_obs*B, E]

        # prepare flattened obs for action_generator (B' = L_obs * B)
        obs_token_flat = obs_token.permute(2, 0, 1, 3).contiguous().view(Q, L_obs * B, E)   # [Q, B', E]
        obs_mask_flat  = obs_mask.permute(2, 0, 1).contiguous().view(Q, L_obs * B)          # [Q, B']
        obs_pos_ids    = (torch.cumsum(obs_mask_flat, dim=0) - 1).long()                    # [Q, B']
        prompt_pos_ids = (torch.cumsum(prompt_mask_for_attn.transpose(0,1), dim=0) - 1).long() if prompt_mask_for_attn is not None else None

        tokens_out_sub = self.action_generator(
            obs_action_tokens=obs_token_flat,
            prompt_tokens=prompt_emb,
            prompt_mask=prompt_mask_for_attn,
            obs_action_masks=obs_mask_flat.bool().transpose(0, 1),
            obs_action_position_ids=obs_pos_ids.transpose(0, 1),
            prompt_position_ids=prompt_pos_ids.transpose(0, 1) if prompt_pos_ids is not None else None,
        )
        pred_sub_last = tokens_out_sub[-1]                               # [B', E]

        residual_rep = predicted_action_tokens_residual.unsqueeze(1).repeat(1, L_obs, 1).reshape(B * L_obs, E)
        subprompt_out = (pred_sub_last + residual_rep).view(B, L_obs, E).permute(1, 0, 2).contiguous()  # [L_obs, B, E]

        # -----------------------
        # VIMA 主干（与 forward_gate 对齐）
        # -----------------------
        tokens_out_vima = self.xattn_gpt(
            obs_action_tokens=seq_tokens,
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
            obs_action_masks=seq_masks.transpose(0, 1),
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )
        vima_out = tokens_out_vima[Q - 1 :: step]  # [L_obs, B, E]

        # -----------------------
        # 融合（gate)
        # -----------------------
        if not torch.is_tensor(gate):
            gate_t = torch.tensor(float(gate), device=device, dtype=vima_out.dtype)
        else:
            gate_t = gate.to(device=device, dtype=vima_out.dtype)
        gate_t = torch.clamp(gate_t, 0.0, 1.0)
        while gate_t.dim() < 3:
            gate_t = gate_t.unsqueeze(-1)

        fused = gate_t * vima_out + (1.0 - gate_t) * subprompt_out
        #fused = vima_out
        return fused, 0, 0, loss_subprompt

        # # ====== 使用 gate 加权 residual（gate=1.0 表示不使用 residual） ======
        # if not torch.is_tensor(gate):
        #     gate_t = torch.tensor(float(gate), device=pred_last.device, dtype=pred_last.dtype)
        # else:
        #     gate_t = gate.to(device=pred_last.device, dtype=pred_last.dtype)
        # gate_t = torch.clamp(gate_t, 0.0, 1.0)
        # # 广播 gate 到 pred_last 形状（常用标量）
        # while gate_t.dim() < pred_last.dim():
        #     gate_t = gate_t.unsqueeze(-1)
        # # 最终按 gate 加权：pred_last + (1 - gate) * residual_rep
        # pred_last = pred_last + (1.0 - gate_t) * residual_rep

        # # 还原为 [L_obs, B, E]
        # out = pred_last.view(B, L_obs, self.embed_dim).permute(1, 0, 2).contiguous()
        # loss_order = torch.zeros((), device=out.device)
        # loss_attr  = torch.zeros((), device=out.device)
        # return out, loss_order, loss_attr, 1
    def forward(
        self,
        obs_token: torch.Tensor,
        obs_mask: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
        subprompt_multimodals: torch.Tensor,# [L_obs,B,S,768] S是子提示长度
        subprompt_masks: torch.Tensor,
    ):
        L_obs, B = obs_token.shape[:2] #L_obs时间步数，B批量大小
        L_action = 0 if action_token is None else action_token.shape[0]
        n_max_objs = obs_token.shape[-2]#最大物体数
        L = L_obs * n_max_objs + L_action#总token数

        tokens = torch.empty(
            L, B, self.embed_dim, dtype=torch.float32, device=obs_token.device
        )
        masks = torch.ones(L, B, dtype=torch.bool, device=obs_token.device)
        obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")
        obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = rearrange(obs_mask, "B L -> L B")
        for q in range(n_max_objs):
            tokens[q :: n_max_objs + 1] = obs_token[q::n_max_objs]#插入观察和mask
            masks[q :: n_max_objs + 1] = obs_mask[q::n_max_objs]
        if action_token is not None:
            tokens[n_max_objs :: n_max_objs + 1] = action_token

        position_ids = torch.cumsum(masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1#生成位置编码



        pad = torch.zeros(1, B, self.embed_dim, device=tokens.device, dtype=tokens.dtype)
        # tokens_padded = torch.cat([pad, subtask_tokens_out], dim=0)
        mask_padded = torch.cat([masks[0:1], masks], dim=0)
        position_ids_padded = torch.cumsum(mask_padded, dim=0) - 1
        position_ids_padded = position_ids_padded.long()

        subtask_tokens = self.sent_decomposer(
            obs_action_tokens=torch.cat([pad, tokens], dim=0),#obj_num*time_step,batch_size,embed_dim
            prompt_tokens=prompt_token,#total_length, batch_size, embed_dim
            prompt_mask=prompt_token_mask,
            obs_action_masks=mask_padded.transpose(0, 1),#1,6
            # obs_action_masks=masks.transpose(0, 1),#1,6
            obs_action_position_ids=position_ids_padded.transpose(0, 1),
            # obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )#n_max_objs*time_step(L_obs)+1, batch_size, embed_dim


        use_residual = True #是否使用残差连接
        residual_mode = "frozen" #残差连接模式
        predicted_action_tokens_residual = 0
        if use_residual:
            if residual_mode == "frozen":
                # subtask_tokens_residual = self.sent_decomposer(
                subtask_tokens_residual = self.action_generator(
                    obs_action_tokens=torch.cat([pad, tokens], dim=0),#obj_num*time_step,batch_size,embed_dim
                    prompt_tokens=prompt_token,#total_length, batch_size, embed_dim
                    prompt_mask=prompt_token_mask,
                    obs_action_masks=mask_padded.transpose(0, 1),#1,6
                    # obs_action_masks=masks.transpose(0, 1),#1,6
                    obs_action_position_ids=position_ids_padded.transpose(0, 1),
                    # obs_action_position_ids=position_ids.transpose(0, 1),
                    prompt_position_ids=prompt_position_ids,
                )#n_max_objs*time_step(L_obs)+1, batch_size, embed_dim
                predicted_action_tokens_residual = subtask_tokens_residual[-1]
            else:
                predicted_action_tokens_residual = subtask_tokens[-1]
        



        subtask_tokens_out = subtask_tokens.reshape(L_obs, n_max_objs+1, B, self.embed_dim).transpose(0, 1) # obj_num+1, time_step, batch_size, embed_dim
        subtask_tokens_out = subtask_tokens_out.reshape(n_max_objs+1, L_obs*B, self.embed_dim)
        subseqlen = subprompt_multimodals.shape[2]
        subprompt_multimodals = subprompt_multimodals.reshape(-1, subseqlen, 768).transpose(0, 1)
        subprompt_masks = subprompt_masks.reshape(-1, subseqlen).transpose(0, 1)
        # fake t5-decoder inputs for debugging
        # subtask_decoder_input = torch.randn(
        #     subtask_tokens_out.shape[0],
        #     subtask_tokens_out.shape[1],
        #     768,
        #     device=subtask_tokens_out.device,
        #     dtype=torch.float32,
        # )
        # subtask_decoder_mask = torch.ones(
        #     subtask_tokens_out.shape[0],
        #     subtask_tokens_out.shape[1],
        #     device=subtask_tokens_out.device,
        #     dtype=torch.bool,
        # )
        subtask_tokens_out = self.subtask_projection(subtask_tokens_out) 
        if self.is_pe:
            subtask_tokens_out, gated_node, gated_node_pe, loss_order, loss_attr, alpha = self.graph_pe(
                subtask_tokens_out, subprompt_multimodals, L_obs=L_obs, B=B
            )
        else:
            # 没有图位置编码时，占位 0，避免上游解包出错
            loss_order = torch.zeros((), device=subtask_tokens_out.device)
            loss_attr = torch.zeros((), device=subtask_tokens_out.device)

        # NLIRDecomposer：从 subtask_tokens_out 生成 subprompt embedding/logits
        # subprompt_multimodals / subprompt_masks 在前面已经 reshape 为:
        #   subprompt_multimodals: [S, L_obs*B, 768]
        #   subprompt_masks:      [S, L_obs*B]
        subtask_prompt_embedding_raw, subtask_prompt_predictions = self.nlir_decomposer(
            subtask_tokens_out, subprompt_multimodals, subprompt_masks
        )  # [L_obs*B, S, 768]（见 NLIRDecomposer 实现）

        # 计算一个简单的 subprompt 重建损失：
        # 直接在 embedding 空间对齐 decoder 输出与 teacher-forcing 的多模态真值
        # 注意：NLIRDecomposer 内部做了 shift-right，这里我们对齐相同长度，忽略 mask 为 0 的位置
        with torch.no_grad():
            # 目标：原始多模态 embedding，转成 [L_obs*B, S, 768]
            target_subprompt = subprompt_multimodals.transpose(0, 1)  # [L_obs*B, S, 768]
            target_mask = subprompt_masks.transpose(0, 1).unsqueeze(-1)  # [L_obs*B, S, 1]
        # diff = (subtask_prompt_embedding_raw - target_subprompt) ** 2
        # diff = diff * target_mask
        # denom = target_mask.sum().clamp_min(1.0)
        # loss_subprompt = diff.sum() / denom

        cos_sim = torch.nn.functional.cosine_similarity(
            subtask_prompt_embedding_raw, target_subprompt, dim=-1
        ) # 结果形状: [L_obs*B, S]

        # 2. 转换为 Loss (1 - cos_sim)，范围 [0, 2]
        # 此时需要扩展一维以匹配 target_mask 的形状 [L_obs*B, S, 1]
        loss_per_token = (1.0 - cos_sim).unsqueeze(-1)

        # 3. 应用 Mask 并计算平均值
        loss_masked = loss_per_token * target_mask
        denom = target_mask.sum().clamp_min(1.0)
        loss_subprompt = loss_masked.sum() / denom

        # 下游 action_generator 仍然使用 T5PromptEncoder 处理后的 embedding
        subtask_prompt_embedding = subtask_prompt_embedding_raw.transpose(0, 1)  # obj_num+1, time_step*batch_size, Ht5 (768)
        # subtask_prompt_embedding = self.t5_prompt_encoder(
        #         subtask_prompt_embedding, 
        #         attention_mask=subtask_decoder_mask.transpose(0, 1), 
        #         batch_first=False
        #     )
        subtask_prompt_embedding = self.t5_prompt_encoder(
                subtask_prompt_embedding, 
                attention_mask=subprompt_masks.transpose(0, 1), 
                batch_first=False
            )
        subtask_prompt_embedding = self.t5_prompt_encoder_post_layer(subtask_prompt_embedding)

        obs_token = obs_token.reshape(L_obs, n_max_objs, B, self.embed_dim).transpose(0, 1) # obj_num, time_step, batch_size, embed_dim
        obs_token = obs_token.reshape(n_max_objs, L_obs*B, self.embed_dim)
        obs_mask = obs_mask.reshape(L_obs, n_max_objs, B).transpose(0, 1)
        obs_mask = obs_mask.reshape(n_max_objs, L_obs*B)

        obs_position_ids = torch.cumsum(obs_mask, dim=0) - 1
        obs_position_ids = obs_position_ids.long()
        # subtask_prompt_position_ids = torch.cumsum(subtask_decoder_mask, dim=0) - 1
        subtask_prompt_position_ids = torch.cumsum(subprompt_masks, dim=0) - 1


        tokens_out = self.action_generator(
            obs_action_tokens= obs_token,#obj_num*time_step,batch_size,embed_dim
            prompt_tokens=subtask_prompt_embedding,#total_length, batch_size, embed_dim
            prompt_mask=subprompt_masks.transpose(0, 1),
            # prompt_mask=subtask_decoder_mask.transpose(0, 1),
            obs_action_masks=obs_mask.bool().transpose(0, 1),#1,6
            obs_action_position_ids=obs_position_ids.transpose(0, 1),
            prompt_position_ids=subtask_prompt_position_ids.transpose(0, 1),
        )
        predicted_action_tokens = tokens_out[-1] + predicted_action_tokens_residual

        

        # tokens_out = self.xattn_gpt(
        #     obs_action_tokens= tokens,#obj_num*time_step,batch_size,embed_dim
        #     prompt_tokens=prompt_token,#total_length, batch_size, embed_dim
        #     prompt_mask=prompt_token_mask,
        #     obs_action_masks=masks.transpose(0, 1),#1,6
        #     obs_action_position_ids=position_ids.transpose(0, 1),
        #     prompt_position_ids=prompt_position_ids,
        # )

        # predicted_action_tokens = tokens_out[n_max_objs - 1 :: n_max_objs + 1]#每隔n_max_objs + 1 步取一个token,其实就是取最后一个tokens
        # predicted_action_tokens = self.test_layer(predicted_action_tokens)
        return (
            predicted_action_tokens.reshape([L_obs, B, self.embed_dim]),
            loss_order,
            loss_attr,
            loss_subprompt,
        )

        
    def forward_new(
        self,
        obs_token: torch.Tensor,
        obs_mask: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
        index: torch.Tensor | None = None,
    ):
        
        L_obs, B = obs_token.shape[:2]
        L_action = 0 if action_token is None else action_token.shape[0]
        n_max_objs = obs_token.shape[-2]#最大物体数
        L = L_obs * n_max_objs + L_action#总token数

        tokens = torch.empty(
            L, B, self.embed_dim, dtype=torch.float32, device=obs_token.device
        )
        masks = torch.ones(L, B, dtype=torch.bool, device=obs_token.device)
        obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")
        obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = rearrange(obs_mask, "B L -> L B")
        for q in range(n_max_objs):
            tokens[q :: n_max_objs + 1] = obs_token[q::n_max_objs]#插入观察和mask
            masks[q :: n_max_objs + 1] = obs_mask[q::n_max_objs]
        if action_token is not None:
            tokens[n_max_objs :: n_max_objs + 1] = action_token

        position_ids = torch.cumsum(masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1#生成位置编码

        tokens_out = self.xattn_gpt(
            obs_action_tokens=tokens,#obj_num,time_step,embed_dim
            prompt_tokens=prompt_token,#total_length, batch_size, embed_dim
            prompt_mask=prompt_token_mask,
            obs_action_masks=masks.transpose(0, 1),#1,6
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )

        predicted_action_tokens = tokens_out[n_max_objs - 1 :: n_max_objs + 1]#每隔n_max_objs + 1 步取一个token,其实就是取最后一个tokens
        return predicted_action_tokens

    def forward_prompt_assembly(self, prompts, is_encoding = True):#让图像和文字的token统一
        raw_prompts_token_type, word_batch, image_batch = prompts
        batch_word_emb = self.prompt_embedding(word_batch)
        batch_image_emb = self.obj_encoder(**image_batch)
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)#mlp
        n_max_objs = batch_image_emb.shape[-2]

        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0:
                    L_this += 1
                elif item == 1:
                    L_this += n_max_objs
                else:
                    raise ValueError(f"Invalid prompt token type {item}")
            L_max = max(L_max, L_this)

        prompt_tokens, prompt_masks = [], []
        word_ptr, img_ptr = 0, 0
        for raw_prompt in raw_prompts_token_type:
            assembled_prompt = []
            assembled_mask = []
            for item in raw_prompt:
                if item == 0:
                    assembled_prompt.append(batch_word_emb[word_ptr])
                    word_ptr += 1
                    assembled_mask.append(True)
                elif item == 1:
                    obj_mask = any_concat(
                        [
                            image_batch["mask"][view][img_ptr]
                            for view in sorted(self._views)
                        ],
                        dim=-1,
                    )
                    for q in range(n_max_objs):
                        assembled_prompt.append(batch_image_emb[img_ptr][q])
                        assembled_mask.append(obj_mask[q])
                    img_ptr += 1
                else:
                    raise ValueError(f"Invalid type: {type(item)}")
            num_padding = L_max - len(assembled_prompt)
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=assembled_prompt.device,
            )
            assembled_prompt = torch.cat([assembled_prompt, required_padding], dim=0)
            prompt_tokens.append(assembled_prompt)

            prompt_masks.append(
                torch.cat(
                    [
                        any_to_torch_tensor(
                            assembled_mask,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                        torch.zeros(
                            num_padding,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                    ],
                    dim=0,
                )
            )

        prompt_tokens = torch.stack(prompt_tokens, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)
        prompt_tokens = prompt_tokens.transpose(0, 1)

    
        if (self.t5_prompt_encoder is not None) and is_encoding:
            prompt_tokens = self.t5_prompt_encoder(
                prompt_tokens, attention_mask=prompt_masks, batch_first=False
            )
            prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens, prompt_masks

    def forward_obs_token(self, obs):#物品变成token
        objects, ee = obs["objects"], obs["ee"]
        leading_dims = ee.shape[:2]

        objects = objects.map_structure(func=lambda x: x.reshape(-1, *x.shape[2:]))
        img_feats = self.obj_encoder(**objects)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])
        obj_mask = {
            k: objects["mask"][k].reshape(*leading_dims, -1) for k in objects["mask"]
        }

        ee_feats = self.end_effector_encoder(ee)
        ee_feats = ee_feats.unsqueeze(2).repeat(1, 1, img_feats.shape[-2], 1)

        obs_feats = self.obs_fusion_layer(torch.cat([img_feats, ee_feats], dim=-1))

        obj_mask = any_concat([obj_mask[view] for view in sorted(self._views)], dim=-1)
        return obs_feats, obj_mask

    def forward_action_token(self, action):#离散的动作变成token
        return self.action_encoder(self._de_discretize_actions(action))
    
    def forward_action_token_train(self, action):#离散的动作变成token
        return self.action_encoder(action)

    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):#反过来
        return self.action_decoder(predicted_action_tokens)

    def discretize_action(self, action):#连续动作->离散动作
        device = action["pose0_position"].device
        boundary_x = torch.linspace(
            start=0, end=1, steps=self._n_discrete_x_bins, device=device
        )
        boundary_y = torch.linspace(
            start=0, end=1, steps=self._n_discrete_y_bins, device=device
        )
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    def _de_discretize_actions(self, actions):
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = (
            actions["pose0_rotation"] / self._n_discrete_rot_bins
        )

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = (
            actions["pose1_rotation"] / self._n_discrete_rot_bins
        )
        return actions
