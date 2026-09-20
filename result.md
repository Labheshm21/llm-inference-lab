# Inference Baseline Results

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt: `What does a GPU do?`
- Generation: `max_new_tokens=40`, `do_sample=False`
- CPU: GitHub Codespaces, PyTorch float32, called through the FastAPI `/chat` endpoint
- GPU: Google Colab Tesla T4, PyTorch float16, called directly in a notebook

## Model-call latency

All times are in milliseconds.

| Environment | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| Codespaces CPU | 4,765.8 | 4,588.3 | 4,572.1 | **4,588.3** |
| Colab T4 GPU | 1,329.6 | 1,718.1 | 1,488.2 | **1,488.2** |

In these runs, the GPU setup had approximately **3.1× lower median model-call latency** than the CPU setup.

## Interpretation and limits

This is a preliminary comparison, not a controlled hardware benchmark. The CPU used float32 while the GPU used float16. The CPU call ran through FastAPI, while the GPU call ran directly in Colab. Only three requests were measured, and generated token counts were not recorded.

The measurements include prompt processing and answer generation. They exclude model download and loading time.

## Next step

Serve the model on a GPU through an HTTP endpoint, connect the FastAPI gateway to it, and benchmark multiple concurrent requests.