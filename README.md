# Step-Audio-EditX
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

<div align="center">
    <a href="https://stepaudiollm.github.io/step-audio-editx/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2511.03601"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Step-Audio-EditX&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Step-Audio-EditX&message=ModelScope&color=blue"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Space%20Playground&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 News!!
* Nov 19, 2025: ⚙️ We release a **new version** of our model, which **supports polyphonic pronunciation control** and improves the performance of emotion, speaking style, and paralinguistic editing.
* Nov 12, 2025: 📦 We release the **optimized inference code** and **model weights** of **Step-Audio-EditX** ([HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-EditX);  [ModelScope](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX)) and **Step-Audio-Tokenizer**([HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer);  [ModelScope](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer))
* Nov 07, 2025: ✨ [Demo Page](https://stepaudiollm.github.io/step-audio-editx/) ; 🎮  [HF Space Playground](https://huggingface.co/spaces/stepfun-ai/Step-Audio-EditX)
* Nov 06, 2025: 👋 We release the technical report of [Step-Audio-EditX](https://arxiv.org/abs/2511.03601).

## Introduction
We are open-sourcing Step-Audio-EditX, a powerful **3B-parameter** LLM-based **Reinforcement Learning** audio model specialized in expressive and iterative audio editing. It excels at editing emotion, speaking style, and paralinguistics, and also features robust zero-shot text-to-speech (TTS) capabilities. 

## 📑 Open-source Plan
- [x] Inference Code
- [x] Online demo (Gradio)
- [ ] Step-Audio-Edit-Benchmark
- [x] Model Checkpoints
  - [x] Step-Audio-Tokenizer
  - [x] Step-Audio-EditX
  - [ ] Step-Audio-EditX-Int4
- [ ] Training Code
  - [ ] SFT training
  - [ ] PPO training
- [ ] ⏳ Feature Support Plan
  - [ ] Editing
    - [x] Polyphone pronunciation control
    - [ ] More paralinguistic tags ([Cough, Crying, Stress, etc.])
    - [ ] Filler word removal
  - [ ] Other Languages
    - [ ] Japanese, Korean, Arabic, French, Russian, Spanish, etc.
  
## Features
- **Zero-Shot TTS**
  - Excellent zero-shot TTS cloning for Mandarin, English, Sichuanese, and Cantonese.
  - To use a dialect, just add a **[Sichuanese]** or **[Cantonese]** tag before your text.
  - 🔥 Polyphone pronunciation control, all you need to do is replace the polyphonic characters with pinyin.
    - **[我也想过过过儿过过的生活]** -> **[我也想guo4guo4guo1儿guo4guo4的生活]**
 
    
- **Emotion and Speaking Style Editing**
  - Remarkably effective iterative control over emotions and styles, supporting **dozens** of options for editing.
    - Emotion Editing : [ *Angry*, *Happy*, *Sad*, *Excited*, *Fearful*, *Surprised*, *Disgusted*, etc. ]
    - Speaking Style Editing: [ *Act_coy*, *Older*, *Child*, *Whisper*, *Serious*, *Generous*, *Exaggerated*, etc.]
    - Editing with more emotion and more speaking styles is on the way. **Get Ready!** 🚀
    

- **Paralinguistic Editing**
  -  Precise control over 10 types of paralinguistic features for more natural, human-like, and expressive synthetic audio.
  - Supporting Tags:
    - [ *Breathing*, *Laughter*, *Suprise-oh*, *Confirmation-en*, *Uhm*, *Suprise-ah*, *Suprise-wa*, *Sigh*, *Question-ei*, *Dissatisfaction-hnn* ]

- **Available Tags**
<table>
  <tr>
    <td rowspan="8" style="vertical-align: middle; text-align:center;" align="center">emotion</td>
    <td align="center"><b>happy</b></td>
    <td align="center">Expressing happiness</td>
    <td align="center"><b>angry</b></td>
    <td align="center">Expressing anger</td>
  </tr>
  <tr>
    <td align="center"><b>sad</b></td>
    <td align="center">Expressing sadness</td>
    <td align="center"><b>fear</b></td>
    <td align="center">Expressing fear</td>
  </tr>
  <tr>
    <td align="center"><b>surprised</b></td>
    <td align="center">Expressing surprise</td>
    <td align="center"><b>confusion</b></td>
    <td align="center">Expressing confusion</td>
  </tr>
  <tr>
    <td align="center"><b>empathy</b></td>
    <td align="center">Expressing empathy and understanding</td>
    <td align="center"><b>embarrass</b></td>
    <td align="center">Expressing embarrassment</td>
  </tr>
  <tr>
    <td align="center"><b>excited</b></td>
    <td align="center">Expressing excitement and enthusiasm</td>
    <td align="center"><b>depressed</b></td>
    <td align="center">Expressing a depressed or discouraged mood</td>
  </tr>
  <tr>
    <td align="center"><b>admiration</b></td>
    <td align="center">Expressing admiration or respect</td>
    <td align="center"><b>coldness</b></td>
    <td align="center">Expressing coldness and indifference</td>
  </tr>
  <tr>
    <td align="center"><b>disgusted</b></td>
    <td align="center">Expressing disgust or aversion</td>
    <td align="center"><b>humour</b></td>
    <td align="center">Expressing humor or playfulness</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td rowspan="17" style="vertical-align: middle; text-align:center;" align="center">speaking style</td>
    <td align="center"><b>serious</b></td>
    <td align="center">Speaking in a serious or solemn manner</td>
    <td align="center"><b>arrogant</b></td>
    <td align="center">Speaking in an arrogant manner</td>
  </tr>
  <tr>
    <td align="center"><b>child</b></td>
    <td align="center">Speaking in a childlike manner</td>
    <td align="center"><b>older</b></td>
    <td align="center">Speaking in an elderly-sounding manner</td>
  </tr>
  <tr>
    <td align="center"><b>girl</b></td>
    <td align="center">Speaking in a light, youthful feminine manner</td>
    <td align="center"><b>pure</b></td>
    <td align="center">Speaking in a pure, innocent manner</td>
  </tr>
  <tr>
    <td align="center"><b>sister</b></td>
    <td align="center">Speaking in a mature, confident feminine manner</td>
    <td align="center"><b>sweet</b></td>
    <td align="center">Speaking in a sweet, lovely manner</td>
  </tr>
  <tr>
    <td align="center"><b>exaggerated</b></td>
    <td align="center">Speaking in an exaggerated, dramatic manner</td>
    <td align="center"><b>ethereal</b></td>
    <td align="center">Speaking in a soft, airy, dreamy manner</td>
  </tr>
  <tr>
    <td align="center"><b>whisper</b></td>
    <td align="center">Speaking in a whispering, very soft manner</td>
    <td align="center"><b>generous</b></td>
    <td align="center">Speaking in a hearty, outgoing, and straight-talking manner</td>
  </tr>
  <tr>
    <td align="center"><b>recite</b></td>
    <td align="center">Speaking in a clear, well-paced, poetry-reading manner</td>
    <td align="center"><b>act_coy</b></td>
    <td align="center">Speaking in a sweet, playful, and endearing manner</td>
  </tr>
  <tr>
    <td align="center"><b>warm</b></td>
    <td align="center">Speaking in a warm, friendly manner</td>
    <td align="center"><b>shy</b></td>
    <td align="center">Speaking in a shy, timid manner</td>
  </tr>
  <tr>
    <td align="center"><b>comfort</b></td>
    <td align="center">Speaking in a comforting, reassuring manner</td>
    <td align="center"><b>authority</b></td>
    <td align="center">Speaking in an authoritative, commanding manner</td>
  </tr>
  <tr>
    <td align="center"><b>chat</b></td>
    <td align="center">Speaking in a casual, conversational manner</td>
    <td align="center"><b>radio</b></td>
    <td align="center">Speaking in a radio-broadcast manner</td>
  </tr>
  <tr>
    <td align="center"><b>soulful</b></td>
    <td align="center">Speaking in a heartfelt, deeply emotional manner</td>
    <td align="center"><b>gentle</b></td>
    <td align="center">Speaking in a gentle, soft manner</td>
  </tr>
  <tr>
    <td align="center"><b>story</b></td>
    <td align="center">Speaking in a narrative, audiobook-style manner</td>
    <td align="center"><b>vivid</b></td>
    <td align="center">Speaking in a lively, expressive manner</td>
  </tr>
  <tr>
    <td align="center"><b>program</b></td>
    <td align="center">Speaking in a show-host/presenter manner</td>
    <td align="center"><b>news</b></td>
    <td align="center">Speaking in a news broadcasting manner</td>
  </tr>
  <tr>
    <td align="center"><b>advertising</b></td>
    <td align="center">Speaking in a polished, high-end commercial voiceover manner</td>
    <td align="center"><b>roar</b></td>
    <td align="center">Speaking in a loud, deep, roaring manner</td>
  </tr>
  <tr>
    <td align="center"><b>murmur</b></td>
    <td align="center">Speaking in a quiet, low manner</td>
    <td align="center"><b>shout</b></td>
    <td align="center">Speaking in a loud, sharp, shouting manner</td>
  </tr>
  <tr>
    <td align="center"><b>deeply</b></td>
    <td align="center">Speaking in a deep and low-pitched tone</td>
    <td align="center"><b>loudly</b></td>
    <td align="center">Speaking in a loud and high-pitched tone</td>
  </tr>
  <tr>
  </tr>
  <tr>
  </tr>
  <tr>
    <td rowspan="5" style="vertical-align: middle; text-align:center;" align="center">paralinguistic</td>
    <td align="center"><b>Breathing</b></td>
    <td align="center">Breathing sound</td>
    <td align="center"><b>Laughter</b></td>
    <td align="center">Laughter or laughing sound</td>
  </tr>
  <tr>
    <td align="center"><b>Uhm</b></td>
    <td align="center">Hesitation sound: "Uhm"</td>
    <td align="center"><b>Sigh</b></td>
    <td align="center">Sighing sound</td>
  </tr>
  <tr>
    <td align="center"><b>Surprise-oh</b></td>
    <td align="center">Expressing surprise: "Oh"</td>
    <td align="center"><b>Surprise-ah</b></td>
    <td align="center">Expressing surprise: "Ah"</td>
  </tr>
  <tr>
    <td align="center"><b>Surprise-wa</b></td>
    <td align="center">Expressing surprise: "Wa"</td>
    <td align="center"><b>Confirmation-en</b></td>
    <td align="center">Confirming: "En"</td>
  </tr>
  <tr>
    <td align="center"><b>Question-ei</b></td>
    <td align="center">Questioning: "Ei"</td>
    <td align="center"><b>Dissatisfaction-hnn</b></td>
    <td align="center">Dissatisfied sound: "Hnn"</td>
  </tr>
</table>
 
## Feature Requests & Wishlist
💡 We welcome all ideas for new features! If you'd like to see a feature added to the project, please start a discussion in our [Discussions](https://github.com/stepfun-ai/Step-Audio-EditX/discussions) section.

We'll be collecting community feedback here and will incorporate popular suggestions into our future development plans. Thank you for your contribution!

## Demos

<table>
  <tr>
    <th style="vertical-align : middle;text-align: center">Task</th>
    <th style="vertical-align : middle;text-align: center">Text</th>
    <th style="vertical-align : middle;text-align: center">Source</th>
    <th style="vertical-align : middle;text-align: center">Edited</th>
  </tr>

  <tr>
    <td align="center"> Emotion-Fear</td>
    <td align="center"> 我总觉得，有人在跟着我，我能听到奇怪的脚步声。</td>
    <td align="center">

  [fear_zh_female_prompt.webm](https://github.com/user-attachments/assets/a088c059-032c-423f-81d6-3816ba347ff5) 
  </td>
    <td align="center">
      
  [fear_zh_female_output.webm](https://github.com/user-attachments/assets/917494ac-5913-4949-8022-46cf55ca05dd)
  </td>
  </tr>


  <tr>
    <td align="center"> Style-Whisper</td>
    <td align="center"> 比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力。</td>
    <td align="center">
      
  [whisper_prompt.webm](https://github.com/user-attachments/assets/ed9e22f1-1bac-417b-913a-5f1db31f35c9)
  </td>
    <td align="center">
      
  [whisper_output.webm](https://github.com/user-attachments/assets/e0501050-40db-4d45-b380-8bcc309f0b5f)
  </td>
  </tr>

  <tr>
    <td align="center"> Style-Act_coy</td>
    <td align="center"> 我今天想喝奶茶，可是不知道喝什么口味，你帮我选一下嘛，你选的都好喝～</td>
    <td align="center">

  [act_coy_prompt.webm](https://github.com/user-attachments/assets/74d60625-5b3c-4f45-becb-0d3fe7cc4b3f)
  </td>
    <td align="center"> 

  [act_coy_output.webm](https://github.com/user-attachments/assets/b2f74577-56c2-4997-afd6-6bf47d15ea51)
  </td>
  </tr>


  <tr>
    <td align="center"> Paralinguistics</td>
    <td align="center"> 你这次又忘记带钥匙了 [Dissatisfaction-hnn]，真是拿你没办法。</td>
    <td align="center">
      
  [paralingustic_prompt.webm](https://github.com/user-attachments/assets/21e831a3-8110-4c64-a157-60e0cf6735f0)
  </td>
    <td align="center">
      
  [paralingustic_output.webm](https://github.com/user-attachments/assets/a82f5a40-c6a3-409b-bbe6-271180b20d7b)
  </td>
  </tr>


  <tr>
    <td align="center"> Denoising</td>
    <td align="center"> Such legislation was clarified and extended from time to time thereafter. No, the man was not drunk, he wondered how we got tied up with this stranger. Suddenly, my reflexes had gone. It's healthier to cook without sugar.</td>
    <td align="center">
      
  [denoising_prompt.webm](https://github.com/user-attachments/assets/70464bf4-ebde-44a3-b2a6-8c292333319b)
  </td>
    <td align="center">
      
  [denoising_output.webm](https://github.com/user-attachments/assets/7cd0ae8d-1bf0-40fc-9bcd-f419bd4b2d21)
  </td>
  </tr>

  <tr>
    <td align="center"> Speed-Faster</td>
    <td align="center"> 上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。</td>
    <td align="center">
      
  [speed_faster_prompt.webm](https://github.com/user-attachments/assets/db46609e-1b98-48d8-99c8-e166cfdfc6e3)
  </td>
    <td align="center">
      
  [speed_faster_output.webm](https://github.com/user-attachments/assets/0fbc14ca-dd4a-4362-aadc-afe0629f4c9f)
  </td>
  </tr>
  
</table>


For more examples, see [demo page](https://stepaudiollm.github.io/step-audio-editx/).

## Model Download

| Models   | 🤗 Hugging Face | ModelScope |
|-------|-------|-------|
| Step-Audio-EditX | [stepfun-ai/Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | [stepfun-ai/Step-Audio-EditX](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX) |
| Step-Audio-Tokenizer | [stepfun-ai/Step-Audio-Tokenizer](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer) | [stepfun-ai/Step-Audio-Tokenizer](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer) |


## Model Usage
### 📜 Requirements
The following table shows the requirements for running Step-Audio-EditX model (batch size = 1):

|     Model    | Parameters |  Setting<br/>(sample frequency) | GPU Optimal Memory  |
|------------|------------|--------------------------------|----------------|
| Step-Audio-EditX   | 3B|         41.6Hz          |       12 GB        |

* An NVIDIA GPU with CUDA support is required.
  * The model is tested on a single L40S GPU.
  * 12GB is just a critical value, and 16GB GPU memory shoule be safer. 
* Tested operating system: Linux

### 🔧 Dependencies and Installation
- Python >= 3.10.0 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.4.1-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Sox
```bash
git clone https://github.com/stepfun-ai/Step-Audio-EditX.git
conda create -n stepaudioedit python=3.10
conda activate stepaudioedit

cd Step-Audio-EditX
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

You can set up the environment required for running Step-Audio-EditX using the provided Dockerfile.

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
#### Local Inference Demo
> [!TIP]
> For optimal performance, keep audio under 30 seconds per inference.

```bash
# zero-shot cloning
# The path of the generated audio file is output/fear_zh_female_prompt_cloned.wav
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-text "我总觉得，有人在跟着我，我能听到奇怪的脚步声。"\
    --prompt-audio "examples/fear_zh_female_prompt.wav"\
    --generated-text "可惜没有如果，已经发生的事情终究是发生了。" \
    --edit-type "clone" \
    --output-dir ./output 

python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-text "His political stance was conservative, and he was particularly close to margaret thatcher."\
    --prompt-audio "examples/zero_shot_en_prompt.wav"\
    --generated-text "Underneath the courtyard is a large underground exhibition room which connects the two buildings.	" \
    --edit-type "clone" \
    --output-dir ./output 

# edit
# There will be one or multiple wave files corresponding to each edit iteration, for example: output/fear_zh_female_prompt_edited_iter1.wav, output/fear_zh_female_prompt_edited_iter2.wav, ...
# emotion; fear
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-text "我总觉得，有人在跟着我，我能听到奇怪的脚步声。" \
    --prompt-audio "examples/fear_zh_female_prompt.wav"\
    --edit-type "emotion" \
    --edit-info "fear" \
    --n-edit-iter 2 \
    --output-dir ./output 

# emotion; happy
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-text "You know, I just finished that big project and feel so relieved. Everything seems easier and more colorful, what a wonderful feeling!" \
    --prompt-audio "examples/en_happy_prompt.wav"\
    --edit-type "emotion" \
    --edit-info "happy" \
    --n-edit-iter 2 \
    --output-dir ./output 

# style; whisper
# for style whisper, the edit iteration num should be set bigger than 1 to get better results.
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-text "比如在工作间隙，做一些简单的伸展运动，放松一下身体，这样，会让你更有精力." \
    --prompt-audio "examples/whisper_prompt.wav" \
    --edit-type "style" \
    --edit-info "whisper" \
    --n-edit-iter 2 \
    --output-dir ./output 

# paraliguistic 
# supported tags, Breathing, Laughter, Suprise-oh, Confirmation-en, Uhm, Suprise-ah, Suprise-wa, Sigh, Question-ei, Dissatisfaction-hnn
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-text "我觉得这个计划大概是可行的，不过还需要再仔细考虑一下。" \
    --prompt-audio "examples/paralingustic_prompt.wav" \
    --generated-text "我觉得这个计划大概是可行的，[Uhm]不过还需要再仔细考虑一下。" \
    --edit-type "paralinguistic" \
    --output-dir ./output 

# denoise
# Prompt text is not needed.
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-audio "examples/denoise_prompt.wav"\
    --edit-type "denoise" \
    --output-dir ./output 

# vad 
# Prompt text is not needed.
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-audio "examples/vad_prompt.wav" \
    --edit-type "vad" \
    --output-dir ./output 

# speed
# supported edit-info: faster, slower, more faster, more slower
python3 tts_infer.py \
    --model-path where_you_download_dir \
    --prompt-text "上次你说鞋子有点磨脚，我给你买了一双软软的鞋垫。" \
    --prompt-audio "examples/speed_prompt.wav" \
    --edit-type "speed" \
    --edit-info "faster" \
    --output-dir ./output 

```



#### Launch Web Demo
Start a local server for online inference.
Assume you have one GPU with at least 12GB memory available and have already downloaded all the models.

```bash
# Step-Audio-EditX demo
python app.py --model-path where_you_download_dir --model-source local

# Memory-efficient options with runtime quantization
# For systems with limited GPU memory, you can use quantization to reduce memory usage:

# INT8 quantization
python app.py --model-path where_you_download_dir --model-source local --quantization int8

# INT4 quantization
python app.py --model-path where_you_download_dir --model-source local --quantization int4

# Using pre-quantized AWQ models
python app.py --model-path path/to/quantized/model --model-source local --quantization awq-4bit

# Example with custom settings:
python app.py --model-path where_you_download_dir --model-source local --torch-dtype float16 --enable-auto-transcribe
```

### 🔄 Model Quantization (Optional)

For users with limited GPU memory, you can create quantized versions of the model to reduce memory requirements:

```bash
# Create an AWQ 4-bit quantized model
python quantization/awq_quantize.py --model_path path/to/Step-Audio-EditX

# Advanced quantization options
python quantization/awq_quantize.py
```

For detailed quantization options and parameters, see [quantization/README.md](quantization/README.md).


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

<div align="center">
<img src="assets/emotion-eval.png" width=800 >
</div>

### Generalization on Closed-Source Models.
- For emotion and speaking style editing, the built-in voices of leading closed-source systems possess considerable in-context capabilities, allowing them to partially convey the emotions in the text. After a single editing round with Step-Audio-EditX, the emotion and style accuracy across all voice models exhibited significant improvement. Further enhancement was observed over the next two iterations, robustly demonstrating our model's strong generalization.

- For paralinguistic editing, after editing with Step-Audio-EditX, the performance of paralinguistic reproduction is comparable to that achieved by the built-in voices of closed-source models when synthesizing native paralinguistic content directly. (**sub** means replacement of paralinguistic tags with native words)


<div align="center">

  <table border="1" cellspacing="0" cellpadding="5" style="border-collapse: collapse; font-family: sans-serif; width: auto;">
    <caption><b>Table: Generalization of Emotion, Speaking Style, and Paralinguistic Editing on Closed-Source Models.</b></caption>
    <thead>
      <tr>
        <th rowspan="2" align="center" style="vertical-align: bottom;">Language</th>
        <th rowspan="2" align="center" style="vertical-align: bottom;">Model</th>
        <th colspan="4" style="border-bottom: 1px solid black;">Emotion &uarr;</th>
        <th colspan="4" style="border-bottom: 1px solid black;">Speaking Style &uarr;</th>
        <th colspan="3" style="border-bottom: 1px solid black; border-left: 1px solid black;">Paralinguistic &uarr;</th>
      </tr>
      <tr>
        <th>Iter<sub>0</sub></th>
        <th>Iter<sub>1</sub></th>
        <th>Iter<sub>2</sub></th>
        <th>Iter<sub>3</sub></th>
        <th style="border-left: 1px solid #ccc;">Iter<sub>0</sub></th>
        <th>Iter<sub>1</sub></th>
        <th>Iter<sub>2</sub></th>
        <th>Iter<sub>3</sub></th>
        <th style="border-left: 1px solid black;">Iter<sub>0</sub></th>
        <th>sub</th>
        <th>Iter<sub>1</sub></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="4" align="center" style="font-weight: bold; vertical-align: middle;">Chinese</td>
        <td align="left">MiniMax-2.6-hd</td>
        <td align="center">71.6</td>
        <td align="center">78.6</td>
        <td align="center">81.2</td>
        <td align="center"><b>83.4</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">36.7</td>
        <td align="center">58.8</td>
        <td align="center">63.1</td>
        <td align="center"><b>67.3</b></td>
        <td align="center" style="border-left: 1px solid black;">1.73</td>
        <td align="center">2.80</td>
        <td align="center">2.90</td>
      </tr>
      <tr>
        <td align="left">Doubao-Seed-TTS-2.0</td>
        <td align="center">67.4</td>
        <td align="center">77.8</td>
        <td align="center">80.6</td>
        <td align="center"><b>82.8</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">38.2</td>
        <td align="center">60.2</td>
        <td align="center"><b>65.0</b></td>
        <td align="center">64.9</td>
        <td align="center" style="border-left: 1px solid black;">1.67</td>
        <td align="center">2.81</td>
        <td align="center">2.90</td>
      </tr>
      <tr>
        <td align="left">GPT-4o-mini-TTS</td>
        <td align="center">62.6</td>
        <td align="center">76.0</td>
        <td align="center">77.0</td>
        <td align="center"><b>81.8</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">45.9</td>
        <td align="center">64.0</td>
        <td align="center">65.7</td>
        <td align="center"><b>69.7</b></td>
        <td align="center" style="border-left: 1px solid black;">1.71</td>
        <td align="center">2.88</td>
        <td align="center">2.93</td>
      </tr>
      <tr style="border-bottom: 1px solid black;">
        <td align="left">ElevenLabs-v2</td>
        <td align="center">60.4</td>
        <td align="center">74.6</td>
        <td align="center">77.4</td>
        <td align="center"><b>79.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">43.8</td>
        <td align="center">63.3</td>
        <td align="center">69.7</td>
        <td align="center"><b>70.8</b></td>
        <td align="center" style="border-left: 1px solid black;">1.70</td>
        <td align="center">2.71</td>
        <td align="center">2.92</td>
      </tr>
      <tr>
        <td rowspan="4" align="center" style="font-weight: bold; vertical-align: middle;">English</td>
        <td align="left">MiniMax-2.6-hd</td>
        <td align="center">55.0</td>
        <td align="center">64.0</td>
        <td align="center">64.2</td>
        <td align="center"><b>66.4</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">51.9</td>
        <td align="center">60.3</td>
        <td align="center">62.3</td>
        <td align="center"><b>64.3</b></td>
        <td align="center" style="border-left: 1px solid black;">1.72</td>
        <td align="center">2.87</td>
        <td align="center">2.88</td>
      </tr>
      <tr>
        <td align="left">Doubao-Seed-TTS-2.0</td>
        <td align="center">53.8</td>
        <td align="center">65.8</td>
        <td align="center">65.8</td>
        <td align="center"><b>66.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">47.0</td>
        <td align="center">62.0</td>
        <td align="center"><b>62.7</b></td>
        <td align="center">62.3</td>
        <td align="center" style="border-left: 1px solid black;">1.72</td>
        <td align="center">2.75</td>
        <td align="center">2.92</td>
      </tr>
      <tr>
        <td align="left">GPT-4o-mini-TTS</td>
        <td align="center">56.8</td>
        <td align="center">61.4</td>
        <td align="center">64.8</td>
        <td align="center"><b>65.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">52.3</td>
        <td align="center">62.3</td>
        <td align="center">62.4</td>
        <td align="center"><b>63.4</b></td>
        <td align="center" style="border-left: 1px solid black;">1.90</td>
        <td align="center">2.90</td>
        <td align="center">2.88</td>
      </tr>
      <tr style="border-bottom: 1px solid black;">
        <td align="left">ElevenLabs-v2</td>
        <td align="center">51.0</td>
        <td align="center">61.2</td>
        <td align="center">64.0</td>
        <td align="center"><b>65.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">51.0</td>
        <td align="center">62.1</td>
        <td align="center">62.6</td>
        <td align="center"><b>64.0</b></td>
        <td align="center" style="border-left: 1px solid black;">1.93</td>
        <td align="center">2.87</td>
        <td align="center">2.88</td>
      </tr>
      <tr>
        <td rowspan="4" align="center" style="font-weight: bold; vertical-align: middle;">Average</td>
        <td align="left">MiniMax-2.6-hd</td>
        <td align="center">63.3</td>
        <td align="center">71.3</td>
        <td align="center">72.7</td>
        <td align="center"><b>74.9</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">44.2</td>
        <td align="center">59.6</td>
        <td align="center">62.7</td>
        <td align="center"><b>65.8</b></td>
        <td align="center" style="border-left: 1px solid black;">1.73</td>
        <td align="center">2.84</td>
        <td align="center">2.89</td>
      </tr>
      <tr>
        <td align="left">Doubao-Seed-TTS-2.0</td>
        <td align="center">60.6</td>
        <td align="center">71.8</td>
        <td align="center">73.2</td>
        <td align="center"><b>74.5</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">42.6</td>
        <td align="center">61.1</td>
        <td align="center"><b>63.9</b></td>
        <td align="center">63.6</td>
        <td align="center" style="border-left: 1px solid black;">1.70</td>
        <td align="center">2.78</td>
        <td align="center">2.91</td>
      </tr>
      <tr>
        <td align="left">GPT-4o-mini-TTS</td>
        <td align="center">59.7</td>
        <td align="center">68.7</td>
        <td align="center">70.9</td>
        <td align="center"><b>73.5</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">49.1</td>
        <td align="center">63.2</td>
        <td align="center">64.1</td>
        <td align="center"><b>66.6</b></td>
        <td align="center" style="border-left: 1px solid black;">1.81</td>
        <td align="center">2.89</td>
        <td align="center">2.90</td>
      </tr>
      <tr>
        <td align="left">ElevenLabs-v2</td>
        <td align="center">55.7</td>
        <td align="center">67.9</td>
        <td align="center">70.7</td>
        <td align="center"><b>72.2</b></td>
        <td align="center" style="border-left: 1px solid #ccc;">47.4</td>
        <td align="center">62.7</td>
        <td align="center">66.1</td>
        <td align="center"><b>67.4</b></td>
        <td align="center" style="border-left: 1px solid black;">1.82</td>
        <td align="center">2.79</td>
        <td align="center">2.90</td>
      </tr>
    </tbody>
  </table>

</div>


## Acknowledgements

Part of the code and data for this project comes from:
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)
* [NVSpeech](https://huggingface.co/datasets/amphion/Emilia-NV)

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


## ⚠️ Usage Disclaimer
- Do not use this model for any unauthorized activities, including but not limited to:
  - Voice cloning without permission
  - Identity impersonation
  - Fraud
  - Deepfakes or any other illegal purposes
- Ensure compliance with local laws and regulations, and adhere to ethical guidelines when using this model.
- The model developers are not responsible for any misuse or abuse of this technology.

We advocate for responsible generative AI research and urge the community to uphold safety and ethical standards in AI development and application. If you have any concerns regarding the use of this model, please feel free to contact us.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=stepfun-ai/Step-Audio-EditX&type=Date)](https://star-history.com/#stepfun-ai/Step-Audio-EditX&Date)
