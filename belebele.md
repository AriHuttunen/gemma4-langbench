# Belebele Benchmark

[Belebele](https://github.com/facebookresearch/belebele) is a massively multilingual reading comprehension benchmark by Meta, covering 122 language variants with 900 questions each.

Each question provides a passage (from FLORES-200), a question, and four multiple-choice answers. The model must select the correct one. The metric is simple **accuracy** (correct / total).

## Data

Downloaded via `download_belebele.py` into `data/belebele/` as JSONL files. Current languages: English, Finnish, Swedish, Estonian.

Each record contains:

| Field               | Description                                    |
|---------------------|------------------------------------------------|
| `flores_passage`    | Reading passage                                |
| `question`          | Comprehension question                         |
| `mc_answer1`–`4`    | Four answer choices                            |
| `correct_answer_num`| Correct answer (1-indexed)                     |
| `dialect`           | Language code (e.g. `fin_Latn`)                |

## Evaluation Methods

### Generative (chat-based)

Prompt the model with the passage, question, and choices. Ask it to output a letter (A/B/C/D). Parse the response and compare to the correct answer.

- Works with any chat API (Ollama, LM Studio, etc.)
- Simple to implement
- Fragile: model may not follow instructions, especially in low-resource languages

### Log-likelihood

Feed the prompt + each possible answer (A, B, C, D) separately and pick whichever letter the model assigns the highest probability.

- More robust (no parsing issues)
- Requires the API to expose token log-probabilities
- Ollama supports this; basic chat UIs do not

### lm-evaluation-harness

[EleutherAI's framework](https://github.com/EleutherAI/lm-evaluation-harness) has Belebele built in using log-likelihood scoring (following MMLU methodology). Run with:

```
lm_eval --model <backend> --tasks belebele_eng_Latn,belebele_fin_Latn,... --num_fewshot 0
```

## Prompt Template

The official few-shot template from the Belebele paper:

```
P: {passage}
Q: {question}
A: {mc_answer1}
B: {mc_answer2}
C: {mc_answer3}
D: {mc_answer4}
Answer:
```

For zero-shot with instruction-tuned models, prepend an instruction:

```
Read the passage and answer the question by writing only the letter (A, B, C, or D).

Passage: {passage}

Question: {question}

A: {mc_answer1}
B: {mc_answer2}
C: {mc_answer3}
D: {mc_answer4}

Answer:
```

## Few-shot Settings

- **0-shot**: No examples, just the question. Tests raw comprehension.
- **5-shot** (paper default): 5 English examples before the test question.

## Testing a Single Question Manually

Paste into a chat with the model:

```
Read the passage and answer the question by writing only the letter (A, B, C, or D).

Passage: Make sure your hand is as relaxed as possible while still hitting all
the notes correctly - also try not to make much extraneous motion with your
fingers. This way, you will tire yourself out as little as possible. Remember
there's no need to hit the keys with a lot of force for extra volume like on the
piano. On the accordion, to get extra volume, you use the bellows with more
pressure or speed.

Question: According to the passage, what would not be considered an accurate tip
for successfully playing the accordion?

A: For additional volume, increase the force with which you hit the keys
B: Keep unnecessary movement to a minimum in order to preserve your stamina
C: Be mindful of hitting the notes while maintaining a relaxed hand
D: Increase the speed with which you operate the bellows to achieve extra volume

Answer:
```

Expected answer: **A**

## References

- [Belebele paper](https://arxiv.org/abs/2308.16884) (Bandarkar et al., 2024, ACL)
- [HuggingFace dataset](https://huggingface.co/datasets/facebook/belebele)
- [GitHub repo](https://github.com/facebookresearch/belebele)
- [lm-evaluation-harness Belebele task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/belebele/README.md)
