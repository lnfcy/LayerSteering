import os
import time
import torch

from collections.abc import Sequence, Mapping
from typing import Callable, TYPE_CHECKING, overload
from collections import defaultdict

if TYPE_CHECKING:
    from src.frame import DecodeRecord


class Timer:
    """
    Basic timer context manager for measuring CPU or GPU time.
    """

    cumulative = dict()

    def __init__(self, name: str | None = None):
        self.name = name

    def __enter__(self):
        if torch.cuda.is_available():
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.start_time.record()  # type: ignore
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available():
            assert isinstance(self.start_time, torch.cuda.Event)
            self.end_time.record()  # type: ignore
            torch.cuda.synchronize()
            self.elapsed_time_s = self.start_time.elapsed_time(self.end_time) / 1000
            self.elapsed_time_ms = self.elapsed_time_s * 1000
        else:
            assert isinstance(self.start_time, float)
            self.end_time = time.time()
            self.elapsed_time_s = self.end_time - self.start_time
            self.elapsed_time_ms = self.elapsed_time_s * 1000

        if self.name is not None:
            Timer.cumulative[self.name] = (
                Timer.cumulative.get(self.name, 0.0) + self.elapsed_time_s
            )

    @classmethod
    def get_cumulative_s(cls, name: str) -> float:
        return cls.cumulative.get(name, 0.0)

    @classmethod
    def get_cumulative_ms(cls, name: str) -> float:
        return cls.get_cumulative_s(name) * 1000

    @property
    def cumulative_s(self) -> float:
        if self.name is None:
            return 0.0
        return self.get_cumulative_s(self.name)

    @property
    def cumulative_ms(self) -> float:
        return self.cumulative_s * 1000

    def token_per_second(self, record: "DecodeRecord", until_eos: bool = True) -> float:
        """
        Calculate the number of tokens processed per second.
        """
        if not hasattr(self, "elapsed_time_s"):
            raise RuntimeError("Timer has not been started or stopped.")
        eos_token_id = int(
            os.environ.get("EOS_TOKEN_ID", "126081")
        )
        if until_eos:
            final_token_seqs = record[-1].generated_tokens
            num_tokens = (final_token_seqs == eos_token_id).int().argmax()
            num_tokens = torch.where(
                num_tokens > 0, num_tokens, final_token_seqs.size(-1)
            )
            return torch.sum(num_tokens).item() / self.elapsed_time_s
        else:
            total_tokens = [
                (
                    torch.cat(delta.transfer_index).numel()
                    if isinstance(delta.transfer_index, tuple)
                    else delta.transfer_index.numel()
                )
                for delta in record.deltas
            ]
            return sum(total_tokens) / self.elapsed_time_s

    def token_per_step(self, record: "DecodeRecord", until_eos: bool = True) -> float:
        """
        Calculate the number of tokens processed per step.
        """
        if until_eos:
            eos_token_id = int(
                os.environ["EOS_TOKEN_ID"]
            )
            final_token_seqs = record[-1].generated_tokens
            num_tokens = (final_token_seqs == eos_token_id).int().argmax(keepdim=True)
            num_tokens = torch.where(
                num_tokens > 0, num_tokens, final_token_seqs.size(-1)
            )
            total_steps = 0
            for batch_idx, eos_idx in enumerate(num_tokens):
                # add 1, since steps are 0-indexed
                total_steps += record[-1].steps[batch_idx, :eos_idx].max().item() + 1
            return torch.sum(num_tokens).item() / total_steps
        else:
            total_tokens = 0
            for delta in record.deltas:
                new_tokens = delta.transferred_tokens
                if isinstance(new_tokens, tuple):
                    total_tokens += sum(t.numel() for t in new_tokens)
                elif isinstance(new_tokens, torch.Tensor):
                    total_tokens += new_tokens.numel()
            return total_tokens / record.num_steps


class LoggerFilter:
    def __init__(self):
        self.msg_history = defaultdict(set)

    def __call__(self, record):
        if "once" in record["extra"]:
            level = record["level"].no
            message = record["message"]
            if message in self.msg_history[level]:
                return False
            self.msg_history[level].add(message)
        elif record["extra"].get("rank_zero_only", False):
            if os.environ.get("LOCAL_RANK", "0") != "0":
                return False
        return True


def apply_fn(
    obj: Mapping | Sequence,
    fn: Callable,
    check_cycles: bool = False,
):
    """
    Recursively traverse a mapping or sequence, and apply `fn` to each non-built-in element.

    Args:
        obj (Mapping | Sequence):
            The object to process. Can be a dict, list, tuple, set, etc.
        fn (Callable):
            The function to apply to each element. It should take a single argument and return a value to replace it.
        check_cycles (bool, optional):
            Whether to check for cycles during the traversal to prevent infinite recursion. Default is False.
    """

    def traverse(obj, visited, depth):
        if check_cycles:
            if id(obj) in visited:
                return obj
            visited.add(id(obj))

        if isinstance(obj, Mapping):
            return {
                key: traverse(value, visited, depth + 1) for key, value in obj.items()
            }

        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            return type(obj)(traverse(item, visited, depth + 1) for item in obj)  # type: ignore[call-arg]

        return fn(obj)

    return traverse(obj, set() if check_cycles else None, 0)


@overload
def tensor_insert(
    dest_tensor: torch.Tensor,
    insert_index: torch.Tensor,
    src: int | float,
) -> torch.Tensor: ...


@overload
def tensor_insert(
    dest_tensor: torch.Tensor,
    insert_index: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor: ...


def tensor_insert(dest_tensor, insert_index, src) -> torch.Tensor:
    """
    Inserts a source tensor or value into a destination tensor along dimension 1.

    This function is specialized for sequence data with shapes like (B, L, ...),
    where B is the batch size and L is the sequence length. `Frame.apply_delta()`
    normalizes single-sequence `Frame` and `FrameDelta` objects into this batched
    layout before calling this helper.

    Args:
        dest_tensor (torch.Tensor): The destination tensor, e.g., of shape (B, L, D).
        insert_index (torch.Tensor): A LongTensor of shape (B, K) specifying insertion
            points along dimension 1.
        src (torch.Tensor | Number): The source to insert.
            - If a Tensor, its shape must match `dest_tensor` but with size K at dim 1,
              e.g., (B, K, D).
            - If a Number, it will be broadcasted to fill the new entries.

    Returns:
        torch.Tensor: The resulting tensor with insertions, shape (B, L+K, ...).
    """

    B, L = dest_tensor.shape[:2]
    K = insert_index.shape[1]

    if K == 0:
        return dest_tensor.clone()

    # reshape inputs to a 3D view: (B, L, rest)
    trailing_dims = dest_tensor.shape[2:]
    T = int(torch.prod(torch.tensor(trailing_dims)).item()) if trailing_dims else 1
    dest_view = dest_tensor.contiguous().view(B, L, T)

    L_new = L + K
    sorted_indices, sort_perm = torch.sort(insert_index, dim=1)

    dest_indices_new = sorted_indices + torch.arange(
        K, device=dest_tensor.device
    ).unsqueeze(0)
    old_indices = torch.arange(L, device=dest_tensor.device).expand(B, -1)
    shifts = (sorted_indices.unsqueeze(2) <= old_indices.unsqueeze(1)).sum(dim=1)
    dest_indices_old = old_indices + shifts

    result_view = torch.empty(
        (B, L_new, T), dtype=dest_tensor.dtype, device=dest_tensor.device
    )

    dest_indices_old = dest_indices_old.unsqueeze(2).expand(B, L, T)
    dest_indices_new = dest_indices_new.unsqueeze(2).expand(B, K, T)

    result_view.scatter_(1, dest_indices_old, dest_view)

    if isinstance(src, (int, float)):
        result_view.scatter_(1, dest_indices_new, src)
    elif isinstance(src, torch.Tensor):
        src_view = src.contiguous().view(B, K, T)
        inv_sort_perm = torch.argsort(sort_perm, dim=1).unsqueeze(2).expand(B, K, T)
        gathered_src = torch.gather(src_view, 1, inv_sort_perm)
        result_view.scatter_(1, dest_indices_new, gathered_src)
    else:
        raise TypeError(f"src must be a Tensor or a number, but got {type(src)}")

    return result_view.view((B, L_new) + trailing_dims)


def tensor_delete(src_tensor: torch.Tensor, delete_index: torch.Tensor) -> torch.Tensor:
    """
    Deletes elements from each row of a (B, L) tensor.

    `Frame.apply_delta()` converts single-sequence `FrameDelta.delete_index` values of
    shape `(K,)` into this batched `(B, K)` layout before calling this helper.

    Args:
        src_tensor (torch.Tensor): The tensor to delete from, shape (B, L).
        delete_index (torch.Tensor): The indices to delete, shape (B, K).

    Returns:
        torch.Tensor: The resulting tensor of shape (B, L - K).
    """
    B, L_current = src_tensor.shape
    K_del = delete_index.shape[1]

    if K_del == 0:
        return src_tensor.clone()

    L_new = L_current - K_del
    device = src_tensor.device

    keep_mask = torch.ones(B, L_current, dtype=torch.bool, device=device)

    row_indices = torch.arange(B, device=device).repeat_interleave(K_del)
    col_indices = delete_index.flatten()

    keep_mask[row_indices, col_indices] = False

    return src_tensor[keep_mask].view(B, L_new)
