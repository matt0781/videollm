import torch, os
from functools import partial
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

_DEBUG_COLLATOR_LOGGED = False  # print only once per process

def _log_input_gt_pairs(batch_text, batch_labels, batch_input_ids, tokenizer):
    """Print input text and the ground-truth (learned) tokens for the first batch."""
    global _DEBUG_COLLATOR_LOGGED
    if _DEBUG_COLLATOR_LOGGED:
        return
    _DEBUG_COLLATOR_LOGGED = True

    print("\n" + "=" * 80)
    print("DEBUG: Training input – ground truth pairs (first batch)")
    print("=" * 80)
    for i, (text, labels, input_ids) in enumerate(zip(batch_text, batch_labels, batch_input_ids)):
        # Collect only the tokens that the model must predict (non-ignore positions)
        gt_token_ids = input_ids[labels != LabelSmoother.ignore_index]
        gt_text = tokenizer.decode(gt_token_ids, skip_special_tokens=False)
        print(f"\n--- Sample {i} ---")
        print(f"[INPUT TEXT]\n{text}")
        print(f"\n[GROUND TRUTH (tokens to predict)]\n{gt_text}")
    print("=" * 80 + "\n")

def data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    batch_text, batch_frames, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs = batch
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        batch_text, batch_labels, batch.input_ids, batch.offset_mapping, batch_learn_ranges
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            labels[start-1:stop-1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
    if os.environ.get("DEBUG_TRAINING_DATA", "1") != "0":
        _log_input_gt_pairs(batch_text, batch_labels, batch.input_ids, tokenizer)
    batch['labels'] = batch_labels
    batch.pop('offset_mapping')
    batch['frames'] = torch.cat(batch_frames)
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    if batch_evaluation_kwargs[0]:
        batch['evaluation_kwargs'] = batch_evaluation_kwargs[0] # evaluation only supports bs = 1, so its okay
    for k, v in batch.items():
        print(f"  batch[{k!r}]: {v.shape if isinstance(v, torch.Tensor) else v}")
    return batch

def get_data_collator(**kwargs):
    return partial(data_collator, **kwargs)
