# Step-Audio-EditX
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

<div align="center">
    <a href="https://stepaudiollm.github.io/step-audio-editx/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2511.03601"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Step-Audio-EditX&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Space%20Playground&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 News!!
* Nov 06, 2025: 👋 We release the technical report of [Step-Audio-EditX](https://arxiv.org/abs/2511.03601).

## Introduction
We are open-sourcing Step-Audio-EditX, a powerful LLM-based audio model specialized in expressive and iterative audio editing. It excels at editing emotion, speaking style, and paralinguistics, and also features robust zero-shot text-to-speech (TTS) capabilities. 

## 📑 Open-source Plan
- [x] Inference Code
- [x] Online demo (Gradio)
- [ ] Step-Audio-Edit-Benchmark
- [x] Model Checkpoints
  - [x] Step-Audio-Tokenizer
  - [ ] Step-Audio-EditX
  - [ ] Step-Audio-EditX-Int8
- [ ] Training Code
  - [ ] SFT training
  - [ ] PPO training

## Features
- **Zero-Shot TTS**
  - Excellent zero-shot TTS cloning for Mandarin, English, Sichuanese, and Cantonese.
  - To use a dialect, just add a **[Cantonese]** or **[Sichuanese]** tag before your text.
 
    
- **Emotion and Speaking Style Editing**
  - Remarkably effective iterative control over emotions and styles, supporting **dozens** of options for editing.
    - Emotion Editing : [ *Angry*, *Happy*, *Sad*, *Excited*, *Fearful*, *Surprised*, *Disgusted*, etc. ]
    - Speaking Style Editing: [ *Act_coy*, *Older*, *Child*, *Whisper*, *Serious*, *Generous*, *Exaggerated*, etc.]
    - Editing with more emotion and more speaking styles is on the way. **Get Ready!** 🚀 
    

- **Paralinguistic Editing**:
  -  Precise control over 10 types of paralinguistic features for more natural, human-like, and expressive synthetic audio.
  - Supporting Tags:
    - [ *Breathing*, *Laughter*, *Suprise-oh*, *Confirmation-en*, *Uhm*, *Suprise-ah*, *Suprise-wa*, *Sigh*, *Question-ei*, *Dissatisfaction-hnn* ]

For more examples, see [demo page](https://stepaudiollm.github.io/step-audio-editx/).

## Model Download

| Models   | 🤗 Hugging Face | ModelScope |
|-------|-------|-------|
| Step-Audio-EditX | [stepfun-ai/Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | [stepfun-ai/Step-Audio-EditX](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX) |


## Model Usage
### 📜 Requirements
The following table shows the requirements for running Step-Audio model (batch size = 1):

|     Model    |  Setting<br/>(sample frequency) | GPU Minimum Memory  |
|------------|--------------------------------|----------------|
| Step-Audio-EditX   |        41.6Hz          |       8GB        |

* An NVIDIA GPU with CUDA support is required.
  * The model is tested on a four A800 80G GPU.
  * **Recommended**: We recommend using 4xA800/H800 GPU with 80GB memory for better generation quality.
* Tested operating system: Linux

### 🔧 Dependencies and Installation
- Python >= 3.10.0 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.3-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
git clone https://github.com/stepfun-ai/Step-Audio-EditX.git
conda create -n stepaudioedit python=3.10
conda activate stepaudioedit

cd Step-Audio
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX

```

After downloading the models, where_you_download_dir should have the following structure:
```
where_you_download_dir
├── Step-Audio-Tokenizer
├── Step-Audio-EditX
```

#### Run with Docker

You can set up the environment required for running Step-Audio using the provided Dockerfile.

```bash
# build docker
docker build . -t step-audio-editx

# run docker
docker run --rm --gpus all \
    -v /your/code/path:/app \
    -v /your/model/path:/model \
    -p 7860:7860 \
    step-audio-editx
```


#### Launch Web Demo
Start a local server for online inference.
Assume you have 4 GPUs available and have already downloaded all the models.

```bash
# Step-Audio-EditX demo
python app.py --model-path where_you_download_dir --model-source local 
```

## Technical Details
<img src="assets/architechture.png" width=900>
Step-Audio-EditX comprises three primary components: 

- A dual-codebook audio tokenizer, which converts reference or input audio into discrete tokens.
- An audio LLM that generates dual-codebook token sequences.
- An audio decoder, which converts the dual-codebook token sequences predicted by the audio LLM back into audio waveforms using a flow matching approach.

Audio-Edit enables iterative control over emotion and speaking style across all voices, leveraging large-margin data during SFT and PPO training.

## Evaluation

### Comparison between Step-Audio-EditX and Closed-Source models.

- Step-Audio-EditX demonstrates superior performance over Minimax and Doubao in both zero-shot cloning and emotion control.
- Emotion editing of Step-Audio-EditX significantly improves the emotion-controlled audio outputs of all three models after just one iteration. With further iterations, their overall performance continues to improve.


<img src="assets/emotion-eval.png" width=800 >


## Acknowledgements

Part of the code for this project comes from:
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

Thank you to all the open-source projects for their contributions to this project!

## License Agreement
+ The code in this open-source repository is licensed under the [Apache 2.0](LICENSE) License.

## Citation

```
@misc{yan2025stepaudioeditxtechnicalreport,
      title={Step-Audio-EditX Technical Report}, 
      author={Chao Yan and Boyong Wu and Peng Yang and Pengfei Tan and Guoqiang Hu and Yuxin Zhang and Xiangyu and Zhang and Fei Tian and Xuerui Yang and Xiangyu Zhang and Daxin Jiang and Gang Yu},
      year={2025},
      eprint={2511.03601},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.03601}, 
}

```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=stepfun-ai/Step-Audio-EditX&type=Date)](https://star-history.com/#stepfun-ai/Step-Audio-EditX&Date)
