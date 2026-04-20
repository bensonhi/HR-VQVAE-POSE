#!/usr/bin/env python
"""
Prepare training data for LoRA fine-tuning of gesture planner.

Converts BEAT2 (TextGrid + .sem) pairs into chat-format JSONL for SFTTrainer.
Each line = one conversation: system prompt + user transcript + assistant JSON plan.

Usage:
    conda run -n qwen python finetune_planner/prepare_data.py
    conda run -n qwen python finetune_planner/prepare_data.py --split train --split val
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_gestures import (
    parse_textgrid, format_transcript_for_prompt, load_gt_sem,
    find_textgrid, merge_adjacent, SYSTEM_PROMPT,
)


def gt_to_json_str(gt_segments):
    """Convert GT segments to the JSON string format expected as model output."""
    out = []
    for seg in gt_segments:
        out.append({
            "start_sec": round(seg['start_sec'], 3),
            "end_sec": round(seg['end_sec'], 3),
            "gesture_type": seg['gesture_type'],
            "text": seg.get('text', ''),
        })
    return json.dumps(out)


def build_conversation(transcript_str, gt_json_str):
    """Build a single conversation in chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Plan gesture types for this transcript:\n\n{transcript_str}"},
            {"role": "assistant", "content": gt_json_str},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beat2-path', default='BEAT2')
    parser.add_argument('--language', default='english')
    parser.add_argument('--output-dir', default='finetune_planner/data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lang_folders = {
        'english': 'beat_english_v2.0.0',
        'chinese': 'beat_chinese_v2.0.0',
    }
    lang_folder = lang_folders.get(args.language, f'beat_{args.language}_v2.0.0')
    split_csv = os.path.join(args.beat2_path, lang_folder, 'train_test_split.csv')

    # Read recording IDs per split from BEAT2's own train_test_split.csv
    split_recordings = {'train': [], 'val': []}
    with open(split_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] in split_recordings:
                split_recordings[row['type']].append(row['id'])

    for split, recs in split_recordings.items():
        print(f"Found {len(recs)} recordings in {split} split")

    # Process each split
    from tqdm import tqdm

    all_conversations = []
    for split, recordings in split_recordings.items():
        conversations = []
        n_skip = 0

        for basename in tqdm(recordings, desc=f"Processing {split}"):
            try:
                tg_path = find_textgrid(basename, args.beat2_path, args.language)
            except FileNotFoundError:
                n_skip += 1
                continue

            words = parse_textgrid(tg_path)
            if not words:
                n_skip += 1
                continue

            gt = load_gt_sem(basename, args.beat2_path, args.language)
            if gt is None or len(gt) == 0:
                n_skip += 1
                continue

            transcript_str = format_transcript_for_prompt(words)
            gt_json_str = gt_to_json_str(gt)
            conv = build_conversation(transcript_str, gt_json_str)
            conv['id'] = basename
            conversations.append(conv)

        # Write JSONL
        out_path = os.path.join(args.output_dir, f'{split}.jsonl')
        with open(out_path, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        print(f"Wrote {len(conversations)} examples to {out_path} ({n_skip} skipped)")
        all_conversations.extend(conversations)

    # Token length stats
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
    lengths = []
    for conv in all_conversations[:100]:
        text = tokenizer.apply_chat_template(conv['messages'], tokenize=False)
        toks = len(tokenizer.encode(text))
        lengths.append(toks)

    print(f"\nToken length stats (sampled {len(lengths)}):")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f}")
    print(f"  Min:  {min(lengths)}")
    print(f"  Max:  {max(lengths)}")


if __name__ == '__main__':
    main()
