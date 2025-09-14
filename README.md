# Numeral-Aware Headline Generation
---

## Overview
Large Language Models (LLMs) demonstrate strong text generation abilities but often struggle with **numerical reasoning** and **numeral-aware text generation**. This limitation is particularly evident in tasks like generating **news headlines containing numerical values**, where both semantic fidelity and numerical accuracy are required.  

As part of the **SemEval 2024 NumEval shared task**, we investigate approaches for **numeral-aware headline generation (English)**. Our work systematically evaluates zero-shot prompting, few-shot prompting, and fine-tuning methods, providing insights into the current capabilities and limitations of LLMs in handling numerically intensive generation tasks.

📄 Refer to the PDF for more details: [Numeval_Report.pdf](report/Numeval_Report.pdf)

---

## Approaches
We explore multiple paradigms for numeral-aware headline generation:

- **Zero-Shot Prompting**  
  - Applied to **Llama 2–7B**.  
  - Task-specific prompts guide the model to generate concise, number-aware headlines.

- **Few-Shot Prompting**  
  - Uses in-context examples to guide headline generation.  
  - Two-shot prompting improves contextual and numerical accuracy.

- **Fine-Tuning**  
  - **T5-large** and **T5-3B** are fine-tuned on the headline generation dataset.  
  - Prefix `"summarize:"` prompts models to generate short, precise headlines.  

We also experiment with **numerical reasoning tasks** using **XLM-R** and masked fine-tuning setups.

---

## Code Usage

### Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

### Model Running

### T5
Use `run_t5.py` to run the T5-large model.
Please adjust the `data` and `model` save path

Similarly, `run_t5-3b.py` for running T5-3B model.

### Llama 2 - 7B
Use `zero_shot_llama2.py` to run the zero-shot and few-shot performance for llama2.

Please adjust the `data` and `predictions` save path

### Evaluating the Predictions
Use `numhg_eval.py` for evaluating the predictions
```
python numhg_eval.py --tgt_path "path_to_labels.txt" --pre_path "path_to_predictions.txt" --num_gt_path "path_to_numerical_gt.txt"
```

### Notebooks

#### numerical_generation_mlm_fine_tune.ipynb
This implements the proposed approach of performing Masked fine-tuning for numerical value generation

#### numerical_generation_zero_shot.ipynb
This notebook contains zero-shot application of xlm-roberta for numerical generation.
<br>
**Note**: Notebooks are independent. Please update the data directories accordingly

---

## 📊 Results

- **Fine-tuning (T5-3B)** achieved the best performance, surpassing the BRIO baseline in headline generation.  
- **Zero-shot and few-shot Llama 2** produced reasonable headlines but lagged behind fine-tuned models.  
- **Numerical reasoning** remains a challenge, with RoBERTa and XLM-R showing promise in masked fine-tuning.  

| Model               | Headline Gen (Rouge-L) | Numerical Reasoning (Accuracy) |
|---------------------|-------------------------|--------------------------------|
| BRIO (baseline)     | 44.12                  | 66.56                          |
| Llama2–7B (zero-shot) | 30.63                  | 40.13                          |
| Llama2–7B (few-shot)  | 32.78                  | 41.08                          |
| T5-large            | 41.64                  | 62.18                          |
| T5-3B               | **42.90**              | **63.65**                      |


