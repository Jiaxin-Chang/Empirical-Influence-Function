import torch
from torch import Tensor, nn
from collections.abc import Callable, Iterable

def compute_loss_per_sample(model, batch, device, ignored_token_ids):
    """
    核心 Loss 计算 (优化版)：
    直接修改 labels 为 -100 来屏蔽 loss。
    """
    # 确保 ignored_token_ids 是 Tensor 且在正确的设备上
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits.float()

    # 1. 进行错位
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous().clone()  # clone 一份，避免修改原始数据

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        mask_to_ignore = torch.isin(shift_labels, ignored_token_ids)
        shift_labels[mask_to_ignore] = -100

    # 4. 计算 Loss
    # reduction='none' 确保返回的是每个 token 的 loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    # 计算出来的 token_losses 在被忽略的位置上已经是 0 了
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

    # 5. 计算有效 token 数量 (即 label 不为 -100 的位置)
    valid_mask = shift_labels.ne(-100).float()
    num_valid  = valid_mask.sum(dim=1)

    # 6. 计算 sum 和 mean loss
    sum_loss = token_losses.sum(dim=1)

    # 避免除以 0
    mean_loss = sum_loss / (num_valid + 1e-9)

    return mean_loss, token_losses


def compute_answer_only_union_topk_loss(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    device: torch.device,
    target_idx: Tensor,
    top_k: int = 10,
    ignored_token_ids: Iterable[int] | Tensor | None = None,
    *,
    enable_grad: bool = False,
    renormalize: bool = True,
) -> tuple[Tensor, Tensor]:
    '''
    Deprecated. Compute loss on a union of top-k correlated tokens of answer tokens.
    '''
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
    labels = inputs["labels"].clone()
    start = int(target_idx[0].item())
    labels[..., :start] = -100

    with torch.set_grad_enabled(enable_grad):
        outputs = model(
            **inputs,
            return_dict=True,
            save_last_attention=True,
            use_cache=False,
        )

    logits = outputs.logits
    attn = outputs.attentions[-1].detach()
    del outputs

    bsz, n_heads, q_len, k_len = attn.shape
    k = min(top_k, k_len)
    attn_avg = attn.mean(dim=1)  # [B, Q, K]

    q_from = max(start, 0)
    if q_from >= q_len:
        raise ValueError("target_idx is beyond sequence length.")

    topk_indices = torch.topk(attn_avg[:, q_from:, :], k=k, dim=-1).indices
    union_mask = torch.zeros((bsz, k_len), device=attn.device, dtype=torch.bool)
    union_mask.scatter_(1, topk_indices.reshape(bsz, -1), True)

    masked_attn = attn_avg * union_mask[:, None, :]
    if renormalize:
        masked_attn = masked_attn / (masked_attn.sum(dim=-1, keepdim=True) + 1e-9)

    token_weights = torch.zeros_like(attn_avg)
    token_weights[:, q_from:, :] = masked_attn[:, q_from:, :].detach()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        shift_labels[torch.isin(shift_labels, ignored_token_ids)] = -100

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    weights = token_weights[..., :-1].contiguous()
    mask = shift_labels.ne(-100)
    weights = weights * mask

    weighted_token_losses = token_losses * weights
    denom = weights.sum(dim=1).clamp_min(1e-9)
    mean_loss = weighted_token_losses.sum(dim=1) / denom
    return mean_loss, weighted_token_losses


def compute_answer_only_saliency_masked_loss(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    device: torch.device,
    target_idx: Tensor,
    top_k: int = 10,
    ignored_token_ids: Iterable[int] | Tensor | None = None,
    *,
    enable_grad: bool = False,
) -> tuple[Tensor, Tensor, list[list[dict[str, object]]]]:
    '''
    Compute loss on answer part (>= `target_idx`), with gradient,
    attention-masked by `top_k` most relative tokens.

    Returns a tuple:
    - `mean_loss`:                of shape `(batch,)`.
    - `weighted_token_losses`:    of shape `(batch, token)`.
    - `saliency_list`:            list[list[dict]], of shape `(batch, target_token, previous_token)`.
    '''
    if ignored_token_ids is not None and not isinstance(ignored_token_ids, torch.Tensor):
        ignored_token_ids = torch.tensor(ignored_token_ids, device=device)
    elif ignored_token_ids is not None:
        ignored_token_ids = ignored_token_ids.to(device)

    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
    labels = inputs["labels"].clone()
    start = int(target_idx[0].item())
    labels[..., :start] = -100

    with torch.set_grad_enabled(enable_grad):
        outputs = model(
            **inputs,
            return_dict=True,
            save_last_attention=True,
            use_cache=False,
        )

    logits = outputs.logits
    attn = outputs.attentions[-1].detach()
    del outputs

    bsz, n_heads, q_len, k_len = attn.shape
    token_weights = torch.zeros((bsz, q_len), device=device, dtype=attn.dtype)

    saliency_list = []
    for i in range(bsz):
        saliency_list.append([])

    if not isinstance(model.get_input_embeddings, Callable):
        raise ValueError("Expect model.get_input_embeddings to be torch.nn.Module")

    for t in range(max(start, 1), q_len):
        curr_input_ids = inputs["input_ids"][:, :t]
        target_vocab_id = inputs["input_ids"][:, t]

        embeddings = model.get_input_embeddings()(curr_input_ids).detach()
        embeddings.requires_grad_(True)

        with torch.enable_grad():
            step_outputs = model(inputs_embeds=embeddings)
            target_logits = step_outputs.logits[:, -1, :]

            target_vocab_id_on_logits_device = target_vocab_id[:, None].to(target_logits.device)
            picked = target_logits.gather(1, target_vocab_id_on_logits_device).sum()
            # gradients from target logits to input embeddings
            grads = torch.autograd.grad(picked, embeddings, retain_graph=False, create_graph=False)[0]

        saliency = (embeddings * grads).abs().sum(dim=-1)   # [batch, token], l1 norm
        k = min(top_k, saliency.size(-1))

        # get top-k saliency token indices
        topk_indices = torch.topk(saliency, k=k, dim=-1).indices

        # ensure same device as mask (model may be split across GPUs with device_map="auto")
        topk_indices = topk_indices.to(device)
        
        # build masks
        mask = torch.zeros((bsz, k_len), device=device, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        # apply attention mask to token t-1
        attn_slice = attn[:, :, t - 1, :]
        masked_attn = attn_slice * mask[:, None, :].to(attn_slice.device)
        token_weights[:, t - 1] = masked_attn.sum(dim=-1).mean(dim=1).detach().to(device)
        # token_weights[:, t - 1] = torch.ones_like(masked_attn.sum(dim=-1).mean(dim=1).detach())

        for i in range(bsz):
            saliency_list[i].append({
                "index": t,
                "saliency": saliency[i].tolist()
            })
        del embeddings, grads, step_outputs

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        shift_labels[torch.isin(shift_labels, ignored_token_ids)] = -100

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    weights = token_weights[..., : token_losses.size(-1)]
    mask = shift_labels.ne(-100)
    weights = weights * mask

    weighted_token_losses = token_losses * weights
    denom = weights.sum(dim=1).clamp_min(1e-9)
    mean_loss = weighted_token_losses.sum(dim=1) / denom
    return mean_loss, weighted_token_losses, saliency_list


@torch.no_grad()
def compute_loss_in_minibatches(model, collator, samples_list, ignored_token_ids, batch_size=2):
    all_samples_loss_list = []  # 存储每个样本的 1D Tensor
    for i in range(0, len(samples_list), batch_size):
        batch_samples = samples_list[i: i + batch_size]
        batch = collator(batch_samples)
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)} # 移到 GPU

        with torch.no_grad():
            _, token_loss = compute_loss_per_sample(model, batch, model.device, ignored_token_ids)

        # 立即上 CPU
        token_loss_cpu = token_loss.detach().cpu()
        all_samples_loss_list.extend(token_loss_cpu.unbind(0))
        del batch

    return all_samples_loss_list

def compute_gradients(
        model,
        batch,
        param_filter_fn,
        device,
        ignored_token_ids
):
    model.eval()
    model.zero_grad(set_to_none=True)

    # Explicitly re-enable requires_grad for filtered params.
    # compute_correlation_second_order_gradient may have frozen all non-Q/K params,
    # so we need to ensure the target params are trainable before the forward pass.
    params = []
    for name, param in model.named_parameters():
        if param_filter_fn is None or param_filter_fn(name, param):
            param.requires_grad_(True)
            params.append(param)

    if not params:
        raise RuntimeError(
            "compute_gradients: no parameters matched param_filter_fn. "
            "Check that the filter is correct and the model has matching layers."
        )

    # Use torch.enable_grad() rather than torch.set_grad_enabled(True):
    # enable_grad() works even when called from inside a torch.no_grad() scope,
    # guaranteeing the forward pass builds a computation graph.
    with torch.enable_grad():
        mean_loss, _ = compute_loss_per_sample(model, batch, device, ignored_token_ids)
        loss = mean_loss.mean()
        if loss.numel() > 1:
            loss = loss.mean()

        # allow_unused=True: if a param doesn't appear in the graph (e.g. a Q/K
        # whose layer is never reached), autograd returns None instead of raising.
        grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
    return list(grads)


def compute_gradients_selected_attention(
    model,
    batch,
    param_filter_fn,
    device,
    ignored_token_ids,
    *,
    target_idx
):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Disable torch.inference_mode() before calling this function.")

    model.eval()
    model.zero_grad(set_to_none=True)

    params = [p for n, p in model.named_parameters()
              if (param_filter_fn is None or param_filter_fn(n, p))]
    if not params:
        raise RuntimeError("No parameters selected by param_filter_fn.")

    orig_flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)

    with torch.enable_grad():
        mean_loss, _, saliency = compute_answer_only_saliency_masked_loss(
            model,
            batch,
            device,
            target_idx,
            ignored_token_ids=ignored_token_ids,
            enable_grad=True,
        )
        loss = mean_loss.mean()

        if not loss.requires_grad:
            raise RuntimeError("Loss is detached. Check outer contexts and model freezing.")

        grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=False)

    for p, flag in zip(params, orig_flags):
        p.requires_grad_(flag)

    return grads, saliency

def compute_token_specific_update(
        model,
        batch,
        param_filter_fn,
        device,
        ignored_token_ids,
        target_sequence_idx: int,
        lr: float
):
    """
    针对 query_batch 中特定序列索引的 token 计算梯度，并应用一次更新。
    """
    model.zero_grad(set_to_none=True)

    # 1. 前向传播
    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits.float()

    # 2. 错位和屏蔽 (与 compute_loss_per_sample 逻辑相似)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous().clone()

    if ignored_token_ids is not None and len(ignored_token_ids) > 0:
        mask_to_ignore = torch.isin(shift_labels, ignored_token_ids.cpu())  # 确保在 CPU 上比较
        shift_labels[mask_to_ignore] = -100

    # 3. 提取单 Token Loss
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    token_losses_flat = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses_flat.view(shift_labels.size())

    # 4. 选取目标 Token 的损失并进行 Backward
    # 确保索引在范围内
    loss_index_in_shifted = target_sequence_idx - 1

    if loss_index_in_shifted >= shift_logits.shape[1] or loss_index_in_shifted < 0:
        raise IndexError(f"Warning: Token index {loss_index_in_shifted} out of bounds.")

    # 仅对该 Token 的损失进行反向传播
    single_token_loss = token_losses[0, loss_index_in_shifted]  # 假设 batch_size=1

    # 仅在损失有效时才反向传播（避免对 -100 的位置求导）
    if single_token_loss.item() != 0 or shift_labels[0, target_sequence_idx].item() != -100:
        single_token_loss.backward()

        # 5. 收集梯度并应用更新
        params = [p for n, p in model.named_parameters() if
                  p.requires_grad and (param_filter_fn is None or param_filter_fn(n, p))]
        grads = [p.grad for p in params]  # 直接使用 .grad

        return grads

    raise IndexError(f"Warning: Token index has label mask as -100.")


@torch.inference_mode()
def get_first_response_token(
        batch,
        ignored_token_ids,
):
    # 找到第一个未被忽略（即需要计算损失）的 Token 索引
    labels_shifted = batch["labels"][0, 1:].cpu()

    effective_ignored_ids = ignored_token_ids.cpu() if ignored_token_ids is not None and ignored_token_ids.numel() > 0 else torch.tensor([])
    is_valid = labels_shifted.ne(-100)
    if effective_ignored_ids.numel() > 0:
        is_valid = is_valid & ~torch.isin(labels_shifted, effective_ignored_ids)

    # 找到第一个为 True 的索引
    valid_indices = torch.where(is_valid)[0]
    if valid_indices.numel() == 0:
        print(f"Could not find any valid response token. Skipping report generation.")
        return None, None

    # query_response_start_idx_in_shifted_labels 是响应在 shift_labels 中的起始索引
    response_start_idx_in_shifted_labels = valid_indices[0].item()

    # 序列总长度 (shift_labels 的长度)
    total_shifted_len = labels_shifted.shape[0]

    # 响应的有效长度
    query_response_len = total_shifted_len - response_start_idx_in_shifted_labels
    return response_start_idx_in_shifted_labels, query_response_len


def compute_correlation_second_order_gradient(
    model, 
    batch, 
    target_idx_in_seq: int,
    source_idx_in_seq: int,
    param_filter_fn
):
    """
    计算二阶导特征: 抽取 source Token 导致 target Token 产生的 Saliency 背后的参数梯度特征
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Disable torch.inference_mode() before calling this function.")

    model.eval()
    model.zero_grad(set_to_none=True)
    
    # 1. 过滤我们需要微调的参数 (比如 qk_last_quarter)
    target_params = []
    for name, param in model.named_parameters():
        if param_filter_fn(name, param):
            param.requires_grad = True
            target_params.append(param)
        else:
            param.requires_grad = False

    device = model.device
    input_ids = batch["input_ids"].to(device)
    target_vocab_id = input_ids[0, target_idx_in_seq]
    
    # 2. 截断输入，只计算到 target 送入前的那一刀
    curr_input_ids = input_ids[:, :target_idx_in_seq]
    
    # 获取 Embeddings，并使其成为一阶导的“叶子结点”
    get_embeds_fn = getattr(model, "get_input_embeddings", lambda: model.model.embed_tokens)
    embeddings = get_embeds_fn()(curr_input_ids).detach()
    embeddings.requires_grad_(True)
    
    with torch.enable_grad():
        # 为了解决 PyTorch "Trying to backward a second time" 问题
        # 我们需要：
        # 1. 临时强制关掉某些可能释放中间激活值的内存优化 (如 gradient checkpointing / flash attention 内部机制)
        # 2. 如果模型在之前的代码中(如 outside)调用过 forward 并发生了 backward，那些图可能残破。
        # 我们用干净的 forward。
        
        # 3. 第一次前向传播（获取 Logit）
        outputs = model(inputs_embeds=embeddings, use_cache=False)
        target_logits = outputs.logits[0, -1, target_vocab_id] 
        
        # 4. 第一次反向传播
        # 注意 retain_graph=True 和 create_graph=True
        # 对 embeddings 取偏导数
        grad_embeds = torch.autograd.grad(
            target_logits, 
            embeddings, 
            retain_graph=True,
            create_graph=True,
            allow_unused=False
        )[0]
        
        # 5. 计算特定的 Correlation Saliency
        saliency_scores = (embeddings * grad_embeds).abs().sum(dim=-1)
        # 如果 source_idx_in_seq 这个值依赖计算图，它提取的元素标量也继续附带计算图
        target_saliency = saliency_scores[0, source_idx_in_seq]
        
        # Saliency 越大越好，等效于 Saliency_Loss (负的 Saliency) 越小越好
        saliency_loss = - target_saliency
        
        # 6. 第二次反向传播
        # 这时求 saliency_loss 关于我们想要提取特征的 target_params 的导数。
        # 因为我们上面使用了 retain_graph=True，计算 target_logits 经历的从 params -> logits 的整条图都被保留了
        final_grads = torch.autograd.grad(
            saliency_loss, 
            target_params, 
            retain_graph=False,   # 最后一次求导了，把图释放掉
            create_graph=False,
            allow_unused=True
        )
        
    # 7. 铺平并组装特征向量
    # 注意：device_map="auto" 时参数分布在多 GPU 上，各张量设备不同。
    # 统一搬到 CPU 再 cat，避免 "Expected all tensors on same device" 报错。
    flat_grad = torch.cat([
        g.reshape(-1).cpu() if g is not None else torch.zeros(p.numel(), dtype=p.dtype)
        for g, p in zip(final_grads, target_params)
    ])
    
    # 恢复 param
    for param in target_params:
         param.requires_grad = False
            
    return flat_grad.detach()


# ──────────────────────────────────────────────────────────────────────────────
# Causal Intervention Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_full_saliency_vector(
    model,
    batch,
    target_idx_in_seq: int,
) -> list[float]:
    """
    用一次前向 + 一次反向计算 [source_0 … source_{target_idx-1}] 对
    target_idx 处 token 的完整 saliency 向量。

    比 compute_answer_only_saliency_masked_loss 快得多：
      - 只需一次 forward，不迭代所有 response token
      - 不需要对 model params 求导（纯 embeddings 梯度）

    Returns:
        list[float], 长度为 target_idx_in_seq。
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Disable torch.inference_mode() before calling this function.")

    model.eval()
    # 确保 model params 不参与梯度图（只对 embeddings 求导）
    for p in model.parameters():
        p.requires_grad_(False)

    device = model.device
    input_ids = batch["input_ids"].to(device)
    target_vocab_id = input_ids[0, target_idx_in_seq]
    curr_input_ids = input_ids[:, :target_idx_in_seq]

    get_embeds_fn = getattr(model, "get_input_embeddings", lambda: model.model.embed_tokens)
    embeddings = get_embeds_fn()(curr_input_ids).detach()
    embeddings.requires_grad_(True)

    with torch.enable_grad():
        outputs = model(inputs_embeds=embeddings, use_cache=False)
        target_logit = outputs.logits[0, -1, target_vocab_id]
        grad_embeds = torch.autograd.grad(target_logit, embeddings)[0]  # [1, seq, dim]

    saliency = (embeddings.detach() * grad_embeds.detach()).abs().sum(dim=-1)  # [1, seq]
    result = saliency[0].tolist()

    del outputs, embeddings, grad_embeds, saliency
    torch.cuda.empty_cache()

    return result


def compute_saliency_score_only(
    model,
    batch,
    target_idx_in_seq: int,
    source_idx_in_seq: int,
) -> float:
    """
    compute_full_saliency_vector 的单值版本。
    仅返回 source_idx → target_idx 的 saliency 标量，用于干预前后的快速测量。
    """
    vec = compute_full_saliency_vector(model, batch, target_idx_in_seq)
    return vec[source_idx_in_seq]


def do_saliency_loss_step(
    model,
    batch,
    target_idx_in_seq: int,
    source_idx_in_seq: int,
    param_filter_fn,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    对 saliency(source_idx → target_idx) 做一步最大化梯度更新。

    Loss = -saliency(source_idx → target_idx)
    使用二阶导路径（retain_graph + create_graph），与
    compute_correlation_second_order_gradient 的前向计算完全一致。

    调用者负责：
      1. 在调用前通过 optimizer 绑定好 filtered params（requires_grad=True）
      2. 在所有步骤完成后恢复权重快照

    Returns:
        float  saliency_loss 的值（负的 saliency score）
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Disable torch.inference_mode() before calling this function.")

    model.eval()
    model.zero_grad(set_to_none=True)

    # 开放 filtered params 的梯度，屏蔽其余参数
    for name, param in model.named_parameters():
        param.requires_grad_(param_filter_fn(name, param))

    device = model.device
    input_ids = batch["input_ids"].to(device)
    target_vocab_id = input_ids[0, target_idx_in_seq]
    curr_input_ids = input_ids[:, :target_idx_in_seq]

    get_embeds_fn = getattr(model, "get_input_embeddings", lambda: model.model.embed_tokens)
    embeddings = get_embeds_fn()(curr_input_ids).detach()
    embeddings.requires_grad_(True)

    with torch.enable_grad():
        # 第一次前向
        outputs = model(inputs_embeds=embeddings, use_cache=False)
        target_logit = outputs.logits[0, -1, target_vocab_id]

        # 第一次反向（对 embeddings；retain_graph + create_graph 保留计算图）
        grad_embeds = torch.autograd.grad(
            target_logit, embeddings,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]

        # saliency score（标量，仍挂载计算图）
        saliency_scores = (embeddings * grad_embeds).abs().sum(dim=-1)
        target_saliency = saliency_scores[0, source_idx_in_seq]
        saliency_loss = -target_saliency  # 最大化 saliency ⇔ 最小化 -saliency

        # 第二次反向（对 model params）
        saliency_loss.backward()

    loss_val = saliency_loss.item()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # 恢复所有 param 的 requires_grad = False（保持 model 的干净状态）
    for param in model.parameters():
        param.requires_grad_(False)

    del outputs, embeddings, grad_embeds, saliency_scores, target_saliency, saliency_loss
    torch.cuda.empty_cache()

    return loss_val
