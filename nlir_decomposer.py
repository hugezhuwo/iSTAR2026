import torch
import torch.nn as nn
# import torch.nn.functional as F

from typing import Optional
try:
    # 仅在可用时导入，用于包装 encoder_hidden_states 为标准输出结构
    from transformers.modeling_outputs import BaseModelOutput
except Exception:
    BaseModelOutput = None

class NLIRDecomposer(nn.Module):
    def __init__(self, embed_dim=256, t5_model_name="t5-small", freeze_t5=False, vima_vocab_size=32142):
        super().__init__()
        # 导入 T5 decoder
        try:
            from transformers import T5ForConditionalGeneration
            self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
            # self.t5_decoder = self.t5_model.decoder
            # tensor 友好的包装器
            self.tensor_decoder = T5TensorDecoder(self.t5_model, freeze_t5=freeze_t5)
        except ImportError:
            raise ImportError("请安装 transformers 库: pip install transformers")

        self.input_projection = nn.Linear(
            embed_dim, 
            self.tensor_decoder.config.hidden_size
        )
        # 自定义 VIMA vocab 的输出头，用于根据外部字典生成 logits
        self.vima_vocab_size = vima_vocab_size
        if vima_vocab_size is not None:
            self.vima_lm_head = nn.Linear(self.tensor_decoder.config.hidden_size, vima_vocab_size, bias=False)
        
    def forward(self, subtask_tokens_out, subtask_decoder_input, subtask_decoder_mask=None, subtask_tokens_mask=None):
        """
        使用 self.tensor_decoder 生成：
        - subtask_prompt_embedding: 解码器每步的可导 embedding（[B, T, E]）
        - subtask_prompt_predictions: 解码器每步的词表 logits（[B, T, V]）

        Args:
            subtask_tokens_out: [L, B, E] 的 encoder 端外部嵌入（必填）
            subtask_decoder_input: [L, B, E] 的 decoder 端目标嵌入（teacher forcing）
                说明：本实现仅支持以目标嵌入形式喂入 decoder；不接受 token ids。
            subtask_decoder_mask: 可选 [B, T]，1 表示可见/参与注意力与损失
            subtask_tokens_mask: 可选 [B, L]，1 表示有效的 encoder token

        Returns:
            (subtask_prompt_embedding, subtask_prompt_predictions)
        """
        device = subtask_tokens_out.device

        # 两者均为 [L, B, E]
        if subtask_tokens_out.dim() != 3 or subtask_decoder_input.dim() != 3:
            raise ValueError("subtask_tokens_out 和 subtask_decoder_input 必须为 [L, B, E]")

        _, B, E = subtask_tokens_out.shape
        # if subtask_decoder_input.shape[:2] != (L, B):
        #     raise ValueError("subtask_decoder_input 的前两维需与 subtask_tokens_out 匹配为 [L, B]")

        Ht5 = self.tensor_decoder.config.hidden_size

        # encoder hidden: 若 E 与 Ht5 不同则投影
        if E != Ht5:
            enc_hidden = self.input_projection(subtask_tokens_out)  # [L, B, Ht5]
        else:
            enc_hidden = subtask_tokens_out  # [L, B, Ht5]
        enc_hidden = enc_hidden.transpose(0, 1).contiguous()  # [B, L, Ht5]

        # encoder attention mask: [B, L] 或 None
        enc_mask = subtask_tokens_mask.transpose(0, 1).contiguous() if subtask_tokens_mask is not None else None

        cfg = self.t5_model.config
        pad_id = cfg.pad_token_id
        start_id = cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else pad_id

        # # decoder teacher forcing: 直接使用嵌入 [L, B, E]
        # if subtask_decoder_input.shape != (L, B, E):
        #     raise ValueError("当传入嵌入时，subtask_decoder_input 必须为 [L, B, E]")

        # tgt = subtask_decoder_input  # [L, B, E]
        # if E != Ht5:
        #     tgt_proj = self.input_projection(tgt)  # [L, B, Ht5]
        # else:
        #     tgt_proj = tgt  # [L, B, Ht5]
        subtask_decoder_input = subtask_decoder_input.transpose(0, 1).contiguous()  # [B, T, Ht5]

        # shift-right 起始 embedding
        start_embed_vec = self.t5_model.shared.weight[start_id].unsqueeze(0).unsqueeze(0).expand(B, 1, -1).contiguous()
        subtask_decoder_input = torch.cat([start_embed_vec.to(subtask_decoder_input.device), subtask_decoder_input[:, :-1, :]], dim=1)  # [B, T, Ht5]

        # 可选 decoder attention mask
        subtask_decoder_mask = subtask_decoder_mask.transpose(0, 1).contiguous() if subtask_decoder_mask is not None else None  # [B, T]

        dec_out = self.tensor_decoder(
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
            decoder_inputs_embeds=subtask_decoder_input,
            decoder_attention_mask=subtask_decoder_mask,
            return_dict=True,
            output_hidden_states=True
        )

        # 取 decoder 最后一层隐状态作为可导 embedding
        pred_embeds = dec_out.decoder_hidden_states[-1]  # [B, T, Ht5]
        # 用外部 VIMA 字典生成 logits（不使用 T5 的 lm_head）
        if getattr(self, "vima_lm_head", None) is None:
            # raise ValueError("请在 NLIRDecomposer 初始化时提供 vima_vocab_size 以生成 VIMA logits")
            logits= dec_out.logits
        else:
            logits = self.vima_lm_head(pred_embeds)  # [B, T, |V_vima|]
        return pred_embeds, logits

    # 在 NLIRDecomposer 类内添加 forward_test，自回归测试用
    # ...existing code...
    def forward_test(
        self,
        subtask_tokens_out: torch.Tensor,
        subtask_decoder_input: Optional[torch.Tensor] = None,
        subtask_tokens_mask: Optional[torch.Tensor] = None,
        subtask_decoder_mask: Optional[torch.Tensor] = None,
        max_gen_steps: int = 16,
        greedy: bool = True,
        return_logits: bool = True,
    ):
        """
        测试前向:
        - Teacher forcing: 传入 subtask_decoder_input（[L_dec,B,E]），走并行解码
        - 否则自回归生成 max_gen_steps
        返回:
          pred_embeds: [B, T, Ht5]
          logits: [B, T, V] 或 None
          decoder_input_ids: 自回归时的 token 序列 (TF 返回 None)
        """
        assert subtask_tokens_out.dim() == 3, "subtask_tokens_out 需要 [L_enc,B,E]"
        d_model = self.t5_model.config.d_model

        # 编码侧投影
        if subtask_tokens_out.size(-1) != d_model:
            subtask_tokens_out = self.input_projection(subtask_tokens_out)
        enc_hidden = subtask_tokens_out.transpose(0, 1).contiguous()          # [B,L_enc,H]
        enc_mask = subtask_tokens_mask.transpose(0, 1) if subtask_tokens_mask is not None else None

        # ===== Teacher Forcing =====
        if subtask_decoder_input is not None:
            assert subtask_decoder_input.dim() == 3, "decoder_input 需 [L_dec,B,E]"
            dec_in = subtask_decoder_input
            if dec_in.size(-1) != d_model:
                dec_in = self.input_projection(dec_in)
            dec_in = dec_in.transpose(0, 1).contiguous()                      # [B,T,H]
            start_id = self.t5_model.config.decoder_start_token_id or self.t5_model.config.pad_token_id
            start_embed = self.t5_model.shared.weight[start_id].unsqueeze(0).unsqueeze(0).expand(dec_in.size(0), 1, -1)
            dec_shift = torch.cat([start_embed.to(dec_in.device), dec_in[:, :-1, :]], dim=1)
            dec_mask = subtask_decoder_mask.transpose(0, 1) if subtask_decoder_mask is not None else None

            out = self.tensor_decoder(
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                decoder_inputs_embeds=dec_shift,
                decoder_attention_mask=dec_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            pred_embeds = out.decoder_hidden_states[-1]                      # [B,T,H]
            if return_logits:
                logits = self.vima_lm_head(pred_embeds) if hasattr(self, "vima_lm_head") else out.logits
            else:
                logits = None
            return pred_embeds, logits, None

        # ===== 自回归生成 =====
        start_id = self.t5_model.config.decoder_start_token_id or self.t5_model.config.pad_token_id
        decoder_input_ids = torch.full(
            (enc_hidden.size(0), 1),
            start_id,
            dtype=torch.long,
            device=enc_hidden.device
        )
        past = None
        gen_hiddens = []
        logits_list = [] if return_logits else None
        for _ in range(max_gen_steps):
            out = self.tensor_decoder(
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out.decoder_hidden_states[-1][:, -1, :]            # [B,H]
            gen_hiddens.append(last_hidden)
            next_logits = out.logits[:, -1, :]                               # [B,V]
            if return_logits:
                logits_list.append(next_logits)
            # 选下一个 token
            if greedy:
                next_token = next_logits.argmax(-1)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)
            past = out.past_key_values

        pred_embeds = torch.stack(gen_hiddens, dim=1)                        # [B,T,H]
        logits = torch.stack(logits_list, dim=1) if return_logits else None  # [B,T,V] 或 None
        return pred_embeds, logits, decoder_input_ids
# ...existing code...

    def forward_new(
        self,
        subtask_tokens_out: torch.Tensor,
        subtask_decoder_input: Optional[torch.Tensor] = None,
        subtask_decoder_mask: Optional[torch.Tensor] = None,
        subtask_tokens_mask: Optional[torch.Tensor] = None,
        gen_len: int = 1,
        greedy: bool = True,
        return_logits: bool = True,
    ):
        """
        支持:
        - Teacher forcing: 提供 subtask_decoder_input
        - 缺失 decoder 输入: 自动用 start token 生成 gen_len 步(=1 相当于占位符)
        返回:
          pred_embeds: [B,T,Ht5]
          logits: [B,T,V] 或 None
        """
        assert subtask_tokens_out.dim() == 3
        d_model = self.t5_model.config.d_model

        # --- Encoder 侧投影 ---
        if subtask_tokens_out.size(-1) != d_model:
            subtask_tokens_out = self.input_projection(subtask_tokens_out)
        enc_hidden = subtask_tokens_out.transpose(0,1).contiguous()          # [B,L_enc,H]
        enc_mask = subtask_tokens_mask.transpose(0,1) if subtask_tokens_mask is not None else None

        # ================= Teacher Forcing =================
        if subtask_decoder_input is not None:
            assert subtask_decoder_input.dim() == 3
            dec_in = subtask_decoder_input
            if dec_in.size(-1) != d_model:
                dec_in = self.input_projection(dec_in)
            dec_in = dec_in.transpose(0,1).contiguous()                      # [B,T,H]
            start_id = self.t5_model.config.decoder_start_token_id or self.t5_model.config.pad_token_id
            start_embed = self.t5_model.shared.weight[start_id].unsqueeze(0).unsqueeze(0).expand(dec_in.size(0),1,-1)
            dec_shift = torch.cat([start_embed.to(dec_in.device), dec_in[:, :-1, :]], dim=1)
            dec_mask = subtask_decoder_mask.transpose(0,1) if subtask_decoder_mask is not None else None
            out = self.tensor_decoder(
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                decoder_inputs_embeds=dec_shift,
                decoder_attention_mask=dec_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            pred_embeds = out.decoder_hidden_states[-1]                      # [B,T,H]
            logits = (self.vima_lm_head(pred_embeds) if (return_logits and hasattr(self,"vima_lm_head"))
                      else (out.logits if return_logits else None))
            return pred_embeds, logits

        # ================= 缺失 decoder 输入，占位/轻量生成 =================
        # gen_len = 1: 仅占位；>1: 简易贪心展开(轻量自回归)
        start_id = self.t5_model.config.decoder_start_token_id or self.t5_model.config.pad_token_id
        decoder_input_ids = torch.full(
            (enc_hidden.size(0), 1),
            start_id,
            dtype=torch.long,
            device=enc_hidden.device
        )
        past = None
        gen_hiddens = []
        for _ in range(gen_len):
            out = self.tensor_decoder(
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out.decoder_hidden_states[-1][:, -1, :]            # [B,H]
            gen_hiddens.append(last_hidden)
            if _ < gen_len - 1:  # 需要继续生成
                next_logits = out.logits[:, -1, :]
                if greedy:
                    next_token = next_logits.argmax(-1)
                else:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).squeeze(-1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)
                past = out.past_key_values
        pred_embeds = torch.stack(gen_hiddens, dim=1)                        # [B,gen_len,H]
        logits = self.vima_lm_head(pred_embeds) if (return_logits and hasattr(self,"vima_lm_head")) else None
        
        # 在 forward_new 返回前增加这一段裁剪：
        # 假设 pred_embeds 形状为 [B, total_len, 768]，total_len = 1 + gen_len 时
        if pred_embeds.shape[0] > gen_len:
            # 丢弃最前面可能的起始/BOS，占位只保留最后 gen_len 个生成 token
            pred_embeds = pred_embeds[-gen_len:, :, :]
            if logits is not None and logits.shape[0] > gen_len:
                logits = logits[-gen_len:,: , :]
        return pred_embeds, logits
    

# class CueDisentangler(nn.Module):
#     def __init__(self, vision_dim, hidden_dim):
#         super().__init__()
#         self.slot_proj = nn.Linear(vision_dim, hidden_dim)

#         # 动作约束路
#         self.action_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
#         self.action_gate = nn.Linear(hidden_dim, 1)

#         # 目标对象路
#         self.target_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
#         self.target_gate = nn.Linear(hidden_dim, 1)

#     def forward(self, vision_slots, text_embed):
#         """
#         vision_slots: [B, N_slots, vision_dim]
#         text_embed:   [B, T, hidden_dim] 来自指令编码
#         """
#         slots = self.slot_proj(vision_slots)  # [B, N, H]

#         # 动作路
#         action_feat, _ = self.action_attn(text_embed, slots, slots)
#         action_weight = torch.sigmoid(self.action_gate(action_feat))
#         action_cues = (action_feat * action_weight).mean(dim=1)  # [B, H]

#         # 目标路
#         target_feat, _ = self.target_attn(text_embed, slots, slots)
#         target_weight = torch.sigmoid(self.target_gate(target_feat))
#         target_cues = (target_feat * target_weight).mean(dim=1)  # [B, H]

#         return action_cues, target_cues


class T5TensorDecoder(nn.Module):
    """
    一个包装器，使 T5 解码器可以直接接受 PyTorch tensor：
    - encoder_hidden_states: [B, S, H]
    - decoder_input_ids 或 decoder_inputs_embeds

    支持从官方预训练权重（如 t5-small）加载，并可与现有 T5 模型共享权重。
    """
    def __init__(self, t5_model=None, t5_model_name="t5-small", freeze_t5=True):
        super().__init__()
        try:
            from transformers import T5ForConditionalGeneration
        except ImportError:
            raise ImportError("请安装 transformers 库: pip install transformers")

        if t5_model is None:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        else:
            self.t5_model = t5_model

        # 暴露 config 以便上层模块访问 hidden_size 等配置
        self.config = self.t5_model.config

        if freeze_t5:
            for p in self.t5_model.parameters():
                p.requires_grad = False

    def forward(
        self,
        encoder_hidden_states,
        encoder_attention_mask=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        # 将 tensor 包装成与 transformers 兼容的 encoder_outputs 结构
        if isinstance(encoder_hidden_states, torch.Tensor):
            if BaseModelOutput is not None:
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            else:
                class _Wrapper:
                    def __init__(self, last_hidden_state):
                        self.last_hidden_state = last_hidden_state
                encoder_outputs = _Wrapper(encoder_hidden_states)
        else:
            encoder_outputs = encoder_hidden_states

        return self.t5_model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

# class PromptDecoder(nn.Module):
#     """
#     使用 T5 模型处理已经编码的 prompt_tokens 和 obs_token
#     生成新的 prompt representation
#     """
#     def __init__(self, embed_dim=512, t5_model_name="t5-small", freeze_t5=True):
#         super().__init__()
        
#         # 导入 T5 decoder
#         try:
#             from transformers import T5ForConditionalGeneration
#             self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
#             self.t5_decoder = self.t5_model.decoder
#             # tensor 友好的包装器
#             self.tensor_decoder = T5TensorDecoder(self.t5_model, freeze_t5=freeze_t5)
#         except ImportError:
#             raise ImportError("请安装 transformers 库: pip install transformers")
        
#         # 如果冻结 T5 权重
#         if freeze_t5:
#             for param in self.t5_decoder.parameters():
#                 param.requires_grad = False
        
#         # 输入投影层：将 VIMA 的 embed_dim 映射到 T5 的 hidden_size
#         self.input_projection = nn.Linear(
#             embed_dim, 
#             self.t5_decoder.config.hidden_size
#         )
        
#         # 输出投影层：将 T5 的输出映射回 VIMA 的 embed_dim
#         self.output_projection = nn.Linear(
#             self.t5_decoder.config.hidden_size, 
#             embed_dim
#         )
        
#         # 融合层：将 prompt 和 obs 信息融合
#         self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        
#         # 位置编码（如果需要）
#         self.position_embedding = nn.Embedding(512, embed_dim)
        
#     def forward(self, prompt_tokens, obs_token, prompt_mask=None, obs_mask=None):
#         """
#         Args:
#             prompt_tokens: [L, B, E] - 已经编码的 prompt tokens
#             obs_token: [L, B, Q, E] - 已经编码的 observation tokens  
#             prompt_mask: [B, L] - prompt 的 mask
#             obs_mask: [L, B, Q] - obs 的 mask
        
#         Returns:
#             enhanced_prompt: [L, B, E] - 增强后的 prompt representation
#         """
#         L, B, E = prompt_tokens.shape
        
#         # 1. 将 prompt_tokens 投影到 T5 的维度
#         prompt_projected = self.input_projection(prompt_tokens)  # [L, B, T5_H]
        
#         # 2. 处理 obs_token：取平均或最大池化
#         if obs_token.dim() == 4:  # [L, B, Q, E]
#             # 对物体维度取平均
#             obs_avg = obs_token.mean(dim=2)  # [L, B, E]
#         else:
#             obs_avg = obs_token  # [L, B, E]
        
#         obs_projected = self.input_projection(obs_avg)  # [L, B, T5_H]
        
#         # 3. 融合 prompt 和 obs 信息
#         # 将 prompt 和 obs 拼接作为 T5 的输入
#         combined_input = torch.cat([prompt_projected, obs_projected], dim=0)  # [2L, B, T5_H]
        
#         # 4. 通过 T5 模型
#         # 创建 attention mask
#         if prompt_mask is not None and obs_mask is not None:
#             # 将两个 mask 拼接
#             obs_mask_avg = obs_mask.any(dim=-1) if obs_mask.dim() == 3 else obs_mask  # [L, B]
#             combined_mask = torch.cat([prompt_mask.transpose(0, 1), obs_mask_avg], dim=0)  # [2L, B]
#         else:
#             combined_mask = None
        
#         # 通过 T5 decoder
#         t5_outputs = self.t5_decoder(
#             inputs_embeds=combined_input,
#             attention_mask=combined_mask,
#             return_dict=True
#         )
        
#         # 5. 将输出投影回原始维度
#         enhanced_combined = self.output_projection(t5_outputs.last_hidden_state)  # [2L, B, E]
        
#         # 6. 分离 prompt 和 obs 的输出
#         enhanced_prompt = enhanced_combined[:L]  # [L, B, E]
#         enhanced_obs = enhanced_combined[L:]     # [L, B, E]
        
#         # 7. 最终融合（可选）
#         final_prompt = self.fusion_layer(
#             torch.cat([enhanced_prompt, enhanced_obs], dim=-1)
#         )  # [L, B, E]
        
#         return final_prompt
    
#     def generate(
#         self,
#         prompt_tokens,
#         obs_token,
#         prompt_mask=None,
#         obs_mask=None,
#         max_new_tokens=32,
#         num_beams=1,
#         do_sample=False,
#         temperature=1.0,
#         top_p=1.0,
#         eos_token_id=None,
#         pad_token_id=None,
#         **generate_kwargs
#     ):
#         """
#         使用自回归方式变长生成。
#         输入为已编码的 embeddings（与 forward 相同），输出为生成的 token ids。

#         Args:
#             prompt_tokens: [L, B, E]
#             obs_token: [L, B, Q, E] or [L, B, E]
#             prompt_mask: [B, L] (1 表示有效)
#             obs_mask: [L, B, Q] or [L, B]
#         Returns:
#             generated_ids: [B, T_out]
#         """
#         device = prompt_tokens.device
#         L, B, E = prompt_tokens.shape

#         # 投影到 T5 hidden 维度
#         prompt_projected = self.input_projection(prompt_tokens)  # [L, B, Ht5]

#         if obs_token.dim() == 4:  # [L, B, Q, E]
#             obs_avg = obs_token.mean(dim=2)  # [L, B, E]
#         else:
#             obs_avg = obs_token  # [L, B, E]
#         obs_projected = self.input_projection(obs_avg)  # [L, B, Ht5]

#         # 拼接作为 encoder 输入，然后转为 batch-first 供 generate 使用
#         combined_input = torch.cat([prompt_projected, obs_projected], dim=0)  # [2L, B, Ht5]
#         combined_input = combined_input.transpose(0, 1).contiguous()  # [B, 2L, Ht5]

#         # 构造 attention mask，batch-first
#         if prompt_mask is not None and obs_mask is not None:
#             obs_mask_avg = obs_mask.any(dim=-1) if obs_mask.dim() == 3 else obs_mask  # [L, B]
#             combined_mask = torch.cat([prompt_mask.transpose(0, 1), obs_mask_avg], dim=0)  # [2L, B]
#             combined_mask = combined_mask.transpose(0, 1).contiguous()  # [B, 2L]
#         else:
#             combined_mask = None

#         # 生成
#         gen_outputs = self.t5_model.generate(
#             inputs_embeds=combined_input,
#             attention_mask=combined_mask,
#             max_new_tokens=max_new_tokens,
#             num_beams=num_beams,
#             do_sample=do_sample,
#             temperature=temperature,
#             top_p=top_p,
#             eos_token_id=eos_token_id if eos_token_id is not None else self.t5_model.config.eos_token_id,
#             pad_token_id=pad_token_id if pad_token_id is not None else self.t5_model.config.pad_token_id,
#             **generate_kwargs
#         )

#         # 返回生成的 token ids（[B, T_out]）
#         return gen_outputs

#     def generate_embeddings(
#         self,
#         prompt_tokens,
#         obs_token,
#         prompt_mask=None,
#         obs_mask=None,
#         decoder_inputs=None,
#         max_new_tokens=32,
#         num_beams=1,
#         do_sample=False,
#         temperature=1.0,
#         top_p=1.0,
#         eos_token_id=None,
#         pad_token_id=None,
#         **kwargs
#     ):
#         """
#         自回归生成但返回每步的投影后 embedding（不暴露 token ids）。

#         Returns:
#             generated_embeddings: [B, T_gen, E]（按批内最大 T_gen 右侧填充）
#             lengths: [B] 每个样本真正生成的长度（遇到 eos 会提前停止）
#         支持两种模式：
#         - 若提供 decoder_inputs（[B, T, E] 张量），则直接以张量作为解码输入（teacher forcing），一次性前向；
#         - 否则走原有自回归按 token ids 生成的路径。
#         """
#         device = prompt_tokens.device
#         L, B, E = prompt_tokens.shape

#         # 1) 编码器输入（来自外部的 embeddings）
#         prompt_projected = self.input_projection(prompt_tokens)  # [L, B, Ht5]
#         if obs_token.dim() == 4:  # [L, B, Q, E]
#             obs_avg = obs_token.mean(dim=2)
#         else:
#             obs_avg = obs_token
#         obs_projected = self.input_projection(obs_avg)  # [L, B, Ht5]

#         encoder_inputs = torch.cat([prompt_projected, obs_projected], dim=0)  # [2L, B, Ht5]
#         encoder_inputs = encoder_inputs.transpose(0, 1).contiguous()  # [B, 2L, Ht5]

#         if prompt_mask is not None and obs_mask is not None:
#             obs_mask_avg = obs_mask.any(dim=-1) if obs_mask.dim() == 3 else obs_mask  # [L, B]
#             enc_mask = torch.cat([prompt_mask.transpose(0, 1), obs_mask_avg], dim=0)  # [2L, B]
#             enc_mask = enc_mask.transpose(0, 1).contiguous()  # [B, 2L]
#         else:
#             enc_mask = None

#         # 2) 先跑 encoder
#         encoder_outputs = self.t5_model.encoder(
#             inputs_embeds=encoder_inputs,
#             attention_mask=enc_mask,
#             return_dict=True
#         )

#         # 如果用户直接给出 decoder_inputs（张量），走张量路径：
#         if decoder_inputs is not None:
#             # 期望 decoder_inputs 为 [B, T, E]，投影到 T5 hidden
#             if decoder_inputs.dim() != 3:
#                 raise ValueError("decoder_inputs must be [B, T, E]")
#             Bx, Tx, Ex = decoder_inputs.shape
#             assert Bx == B and Ex == E, "decoder_inputs shape must match [B, T, E]"

#             tgt_proj = self.input_projection(decoder_inputs.transpose(0, 1))  # [T, B, Ht5]
#             tgt_proj = tgt_proj.transpose(0, 1).contiguous()  # [B, T, Ht5]

#             # 以起始 token 的嵌入做 shift-right
#             cfg = self.t5_model.config
#             start_id = cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else cfg.pad_token_id
#             start_embed_vec = self.t5_model.shared.weight[start_id].unsqueeze(0).unsqueeze(0).expand(B, 1, -1).contiguous()
#             decoder_inputs_embeds = torch.cat([start_embed_vec.to(tgt_proj.device), tgt_proj[:, :-1, :]], dim=1)  # [B, T, Ht5]

#             enc_hidden = encoder_outputs.last_hidden_state
#             dec_out = self.tensor_decoder(
#                 encoder_hidden_states=enc_hidden,
#                 encoder_attention_mask=enc_mask,
#                 decoder_inputs_embeds=decoder_inputs_embeds,
#                 return_dict=True,
#                 output_hidden_states=True
#             )
#             # 取 decoder 最后一层隐状态
#             dec_last = dec_out.decoder_hidden_states[-1]  # [B, T, Ht5]
#             pred_embeds = self.output_projection(dec_last)  # [B, T, E]
#             lengths = torch.full((B,), Tx, dtype=torch.long, device=device)
#             return pred_embeds, lengths

#         # 3) 自回归解码（仅为推进解码需要 ids，但不对外暴露）
#         cfg = self.t5_model.config
#         eos_id = eos_token_id if eos_token_id is not None else cfg.eos_token_id
#         pad_id = pad_token_id if pad_token_id is not None else cfg.pad_token_id
#         start_id = cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else pad_id

#         decoder_input_ids = torch.full((B, 1), start_id, dtype=torch.long, device=device)
#         past_key_values = None
#         finished = torch.zeros(B, dtype=torch.bool, device=device)
#         lengths = torch.zeros(B, dtype=torch.long, device=device)
#         collected_embeddings = []  # list of [B, E]

#         # 允许直接以 tensor 形式传递 encoder_hidden_states 给解码器
#         enc_hidden = encoder_outputs.last_hidden_state
#         for _ in range(max_new_tokens):
#             outputs = self.tensor_decoder(
#                 encoder_hidden_states=enc_hidden,
#                 encoder_attention_mask=enc_mask,
#                 decoder_input_ids=decoder_input_ids,
#                 past_key_values=past_key_values,
#                 use_cache=True,
#                 return_dict=True,
#                 output_hidden_states=True
#             )

#             # 取最后一步的 decoder 隐状态并投影到外部 E 维
#             last_hidden = outputs.decoder_hidden_states[-1][:, -1, :]  # [B, Ht5]
#             step_embed = self.output_projection(last_hidden)  # [B, E]
#             collected_embeddings.append(step_embed)

#             logits = outputs.logits[:, -1, :]  # [B, V]

#             if do_sample:
#                 # 温度、核采样
#                 logits = logits / max(temperature, 1e-6)
#                 if top_p < 1.0:
#                     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#                     cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
#                     cutoff = cumulative_probs > top_p
#                     cutoff[..., 1:] = cutoff[..., :-1].clone()
#                     cutoff[..., 0] = False
#                     sorted_logits[cutoff] = float('-inf')
#                     filtered_logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_indices, dim=-1))
#                     next_token = torch.distributions.Categorical(logits=filtered_logits).sample()
#                 else:
#                     next_token = torch.distributions.Categorical(logits=logits).sample()
#             else:
#                 next_token = torch.argmax(logits, dim=-1)

#             # 处理 eos：记录第一次到达的位置；已完成样本强制 pad
#             newly_finished = (~finished) & (next_token == eos_id)
#             lengths = torch.where(newly_finished, lengths + 1, lengths)
#             finished = finished | newly_finished
#             next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)

#             # 推进解码序列；保留过去缓存以 O(1) 追加
#             decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)
#             past_key_values = outputs.past_key_values

#             # 对于仍未结束的样本，步数 +1（用于那些未遇到 eos 的样本）
#             lengths = torch.where(~finished, lengths + 1, lengths)

#             if torch.all(finished):
#                 break

#         # 堆叠并右侧 pad 到批内最大生成长度
#         if len(collected_embeddings) == 0:
#             gen_len = 0
#             generated_embeddings = torch.zeros(B, 0, E, device=device)
#         else:
#             gen_len = len(collected_embeddings)
#             generated_embeddings = torch.stack(collected_embeddings, dim=1)  # [B, T, E]

#         return generated_embeddings, lengths

#     # ====== Training utilities ======
#     @staticmethod
#     def _shift_right(labels, start_id, pad_id):
#         """Shift decoder targets to build decoder inputs for teacher forcing.
#         - labels: [B, T] with -100 as ignore index for CE
#         """
#         input_ids = labels.clone()
#         input_ids[input_ids == -100] = pad_id
#         input_ids = torch.roll(input_ids, shifts=1, dims=1)
#         input_ids[:, 0] = start_id
#         return input_ids

#     def forward_training(
#         self,
#         prompt_tokens,
#         obs_token,
#         prompt_mask=None,
#         obs_mask=None,
#         labels=None,
#         label_mask=None,
#         target_embeddings=None,
#         loss_type="ce",
#         return_step_embeddings=False
#     ):
#         """
#         训练友好的前向：支持
#         - CE: 传入 token labels（[B, T]），内部用 teacher forcing 计算交叉熵
#         - Embedding: 传入 target_embeddings（[B, T, E]），回归 MSE 或 Cosine

#         Returns:
#             loss, outputs (dict)
#         """
#         # 0) Build encoder side from embeddings (same as forward/generate)
#         device = prompt_tokens.device
#         L, B, E = prompt_tokens.shape

#         prompt_projected = self.input_projection(prompt_tokens)  # [L, B, Ht5]
#         if obs_token.dim() == 4:
#             obs_avg = obs_token.mean(dim=2)
#         else:
#             obs_avg = obs_token
#         obs_projected = self.input_projection(obs_avg)  # [L, B, Ht5]

#         encoder_inputs = torch.cat([prompt_projected, obs_projected], dim=0).transpose(0, 1).contiguous()  # [B, 2L, Ht5]
#         if prompt_mask is not None and obs_mask is not None:
#             obs_mask_avg = obs_mask.any(dim=-1) if obs_mask.dim() == 3 else obs_mask  # [L, B]
#             enc_mask = torch.cat([prompt_mask.transpose(0, 1), obs_mask_avg], dim=0).transpose(0, 1).contiguous()  # [B, 2L]
#         else:
#             enc_mask = None

#         encoder_outputs = self.t5_model.encoder(
#             inputs_embeds=encoder_inputs,
#             attention_mask=enc_mask,
#             return_dict=True
#         )

#         cfg = self.t5_model.config
#         pad_id = cfg.pad_token_id
#         start_id = cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else pad_id

#         outputs = {}

#         if labels is not None and loss_type == "ce":
#             # 1) Cross-entropy with teacher forcing
#             # mask -> labels: set masked to -100
#             if label_mask is not None:
#                 labels = labels.masked_fill(label_mask == 0, -100)
#             # build decoder inputs
#             decoder_input_ids = self._shift_right(labels, start_id=start_id, pad_id=pad_id)
#             ce_out = self.t5_model(
#                 encoder_outputs=encoder_outputs,
#                 attention_mask=enc_mask,
#                 decoder_input_ids=decoder_input_ids,
#                 labels=labels,
#                 return_dict=True,
#                 output_hidden_states=return_step_embeddings
#             )
#             loss = ce_out.loss
#             outputs.update({"logits": ce_out.logits})
#             if return_step_embeddings:
#                 # Project decoder last hidden state at each step to external E
#                 dec_last = ce_out.last_hidden_state  # [B, T, Ht5]
#                 step_embeddings = self.output_projection(dec_last)  # [B, T, E]
#                 outputs.update({"step_embeddings": step_embeddings})
#             return loss, outputs

#         if target_embeddings is not None and loss_type in ("mse", "cosine"):
#             # 2) Embedding regression with decoder_inputs_embeds teacher forcing
#             Bx, Tx, Ex = target_embeddings.shape
#             assert Bx == B and Ex == E, "target_embeddings shape must be [B, T, E]"

#             # Project target embeddings to T5 hidden and shift-right by start token embedding
#             tgt_proj = self.input_projection(target_embeddings.transpose(0, 1))  # [T, B, Ht5]
#             tgt_proj = tgt_proj.transpose(0, 1).contiguous()  # [B, T, Ht5]

#             # start token embedding from shared embedding table
#             start_embed_vec = self.t5_model.shared.weight[start_id]  # [Vdim]
#             # Map vocab dim to model hidden if they differ (for T5 they match model_dim)
#             start_embed_vec = start_embed_vec.unsqueeze(0).unsqueeze(0).expand(B, 1, -1).contiguous()  # [B,1,Ht5]
#             decoder_inputs_embeds = torch.cat([start_embed_vec.to(tgt_proj.device), tgt_proj[:, :-1, :]], dim=1)  # [B, T, Ht5]

#             dec_out = self.t5_model(
#                 encoder_outputs=encoder_outputs,
#                 attention_mask=enc_mask,
#                 decoder_inputs_embeds=decoder_inputs_embeds,
#                 return_dict=True,
#                 output_hidden_states=True
#             )

#             # Take last hidden states for each step and project to external E
#             dec_last = dec_out.last_hidden_state  # [B, T, Ht5]
#             pred_embeds = self.output_projection(dec_last)  # [B, T, E]

#             if label_mask is not None:
#                 mask = label_mask.float().unsqueeze(-1)  # [B, T, 1]
#             else:
#                 mask = torch.ones(B, pred_embeds.size(1), 1, device=device)

#             if loss_type == "mse":
#                 diff = (pred_embeds - target_embeddings) ** 2
#                 diff = diff * mask
#                 denom = mask.sum().clamp_min(1.0)
#                 loss = diff.sum() / denom
#             else:  # cosine
#                 cos = F.cosine_similarity(pred_embeds, target_embeddings, dim=-1)  # [B, T]
#                 cos = cos * mask.squeeze(-1)
#                 denom = mask.squeeze(-1).sum().clamp_min(1.0)
#                 loss = (1.0 - cos).sum() / denom

#             outputs.update({"pred_embeddings": pred_embeds})
#             if return_step_embeddings:
#                 outputs.update({"step_embeddings": pred_embeds})
#             return loss, outputs

#         raise ValueError("forward_training requires (labels with loss_type='ce') or (target_embeddings with loss_type in {'mse','cosine'})")
    
#     def forward_simple(self, prompt_tokens, obs_token):
#         """
#         简化版本：只处理 prompt_tokens，不融合 obs
#         """
#         L, B, E = prompt_tokens.shape
        
#         # 投影到 T5 维度
#         prompt_projected = self.input_projection(prompt_tokens)
        
#         # 通过 T5 encoder
#         t5_outputs = self.t5_encoder(
#             inputs_embeds=prompt_projected,
#             return_dict=True
#         )
        
#         # 投影回原始维度
#         enhanced_prompt = self.output_projection(t5_outputs.last_hidden_state)
        
#         return enhanced_prompt


# class PromptDecoderWithCrossAttention(nn.Module):
    # """
    # 使用交叉注意力的版本，更好地融合 prompt 和 obs 信息
    # """
    # def __init__(self, embed_dim=512, t5_model_name="t5-small", freeze_t5=True):
    #     super().__init__()
        
    #     try:
    #         from transformers import T5EncoderModel
    #         self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)
    #     except ImportError:
    #         raise ImportError("请安装 transformers 库: pip install transformers")
        
    #     if freeze_t5:
    #         for param in self.t5_encoder.parameters():
    #             param.requires_grad = False
        
    #     self.input_projection = nn.Linear(embed_dim, self.t5_encoder.config.hidden_size)
    #     self.output_projection = nn.Linear(self.t5_encoder.config.hidden_size, embed_dim)
        
    #     # 交叉注意力层
    #     self.cross_attention = nn.MultiheadAttention(
    #         embed_dim=embed_dim,
    #         num_heads=8,
    #         batch_first=False
    #     )
        
    #     # 融合层
    #     self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        
    # def forward(self, prompt_tokens, obs_token, prompt_mask=None, obs_mask=None):
    #     """
    #     使用交叉注意力融合 prompt 和 obs
    #     """
    #     L, B, E = prompt_tokens.shape
        
    #     # 处理 obs_token
    #     if obs_token.dim() == 4:  # [L, B, Q, E]
    #         obs_avg = obs_token.mean(dim=2)  # [L, B, E]
    #     else:
    #         obs_avg = obs_token  # [L, B, E]
        
    #     # 交叉注意力：prompt 作为 query，obs 作为 key 和 value
    #     enhanced_prompt, _ = self.cross_attention(
    #         query=prompt_tokens,
    #         key=obs_avg,
    #         value=obs_avg,
    #         key_padding_mask=obs_mask.transpose(0, 1) if obs_mask is not None else None
    #     )
        
    #     # 融合原始 prompt 和增强后的 prompt
    #     final_prompt = self.fusion_layer(
    #         torch.cat([prompt_tokens, enhanced_prompt], dim=-1)
    #     )
        
    #     return final_prompt


# # 使用示例和集成代码
# def create_prompt_decoder_example():
#     """
#     创建 PromptDecoder 的示例
#     """
#     # 创建 decoder（使用 t5-small，冻结预训练权重）
#     prompt_decoder = PromptDecoder(
#         embed_dim=512,  # 与 VIMA 的 embed_dim 保持一致
#         t5_model_name="t5-small",
#         freeze_t5=True  # 冻结 T5 权重，只训练投影层
#     )
    
#     return prompt_decoder


# def integrate_with_training_loop():
#     """
#     展示如何在训练循环中集成 PromptDecoder
#     """
#     # 1. 初始化 PromptDecoder
#     prompt_decoder = PromptDecoder(embed_dim=512, t5_model_name="t5-small")
#     prompt_decoder.to('cuda')  # 移动到 GPU
    
#     # 2. 在训练循环中使用
#     # 假设你已经有了 prompt_tokens 和 obs_token
    
#     # 示例数据
#     batch_size = 2
#     seq_len = 56
#     embed_dim = 512
#     num_objects = 6
    
#     # 模拟已经编码的 prompt_tokens 和 obs_token
#     prompt_tokens = torch.randn(seq_len, batch_size, embed_dim).to('cuda')
#     obs_token = torch.randn(seq_len, batch_size, num_objects, embed_dim).to('cuda')
#     prompt_mask = torch.ones(batch_size, seq_len).to('cuda')
#     obs_mask = torch.ones(seq_len, batch_size, num_objects).to('cuda')
    
#     # 通过 PromptDecoder 处理
#     enhanced_prompt = prompt_decoder(
#         prompt_tokens=prompt_tokens,
#         obs_token=obs_token,
#         prompt_mask=prompt_mask,
#         obs_mask=obs_mask
#     )
    
#     print(f"原始 prompt_tokens 形状: {prompt_tokens.shape}")
#     print(f"增强后 prompt 形状: {enhanced_prompt.shape}")
    
#     return enhanced_prompt

