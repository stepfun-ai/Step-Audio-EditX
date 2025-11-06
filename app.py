import gradio as gr
import os
import argparse
import torch
import logging
from datetime import datetime
import torchaudio
import librosa
import soundfile as sf

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Save audio to temporary directory
def save_audio(audio_type, audio_data, sr, tmp_dir):
    """Save audio data to a temporary file with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(tmp_dir, audio_type, f"{current_time}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(save_path, audio_data, sr)
        logger.debug(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


class EditxTab:
    """Audio editing and voice cloning interface tab"""

    def __init__(self, args):
        self.args = args
        self.edit_type_list = list(get_supported_edit_types().keys())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def history_messages_to_show(self, messages):
        """Convert message history to gradio chatbot format"""
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": gr.Audio(value=raw_audio_part, interactive=False)},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": gr.Audio(value=edit_audio_part, interactive=False)}
            ])
        return show_msgs

    def generate_clone(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate cloned audio"""
        self.logger.info("Starting voice cloning process")
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Use common_tts_engine for cloning
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed"
                self.logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state
        
    def generate_edit(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate edited audio"""
        self.logger.info("Starting audio editing process")

        # Input validation
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                self.logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                sample_rate, audio_numpy, previous_text = state["history_audio"][-1]
                temp_path = save_audio("temp", audio_numpy, sample_rate, self.args.tmp_dir)
                audio_to_edit = temp_path
                text_to_use = previous_text
                self.logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # For para-linguistic, use generated_text; otherwise use source text
            if edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, edit_type, edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                self.logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }

    def register_components(self):
        """Register gradio components - maintaining exact layout from original"""
        with gr.Tab("Editx"):
            with gr.Row():
                with gr.Column():
                    self.model_input = gr.Textbox(label="Model Name", value="Step-Audio-EditX", scale=1)
                    self.prompt_text_input = gr.Textbox(label="Prompt Text", value="", scale=1)
                    self.prompt_audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        format="wav",
                        type="filepath",
                        label="Input Audio",
                    )
                    self.generated_text = gr.Textbox(label="Target Text", lines=1, max_lines=200, max_length=100)
                    with gr.Row():
                        self.button_tts = gr.Button("CLONE")
                        self.button_edit = gr.Button("EDIT")

                with gr.Column():
                    with gr.Row():
                        self.edit_type = gr.Dropdown(label="Task", choices=self.edit_type_list, value="clone")
                        self.edit_info = gr.Dropdown(label="Sub-task", choices=[], value=None)
                    self.chat_box = gr.Chatbot(label="History", type="messages", height=480*1)
                    self.clean_history_submit = gr.Button("Clear History")

            gr.Markdown("---")
            gr.Markdown("""
                **Button Description:**
                - CLONE: Synthesizes audio based on uploaded audio and text, only used for clone mode, will clear history information when used.
                - EDIT: Edits based on uploaded audio, or continues to stack edit effects based on the previous round of generated audio.
                """)
            gr.Markdown("""
                **Operation Workflow:**
                - Upload the audio to be edited on the left side and fill in the corresponding text content of the audio;
                - If the task requires modifying text content (such as clone, para-linguistic), fill in the text to be synthesized in the "clone text" field. For all other tasks, keep the uploaded audio text content unchanged;
                - Select tasks and subtasks on the right side (some tasks have no subtasks, such as vad, etc.);
                - Click the "CLONE" or "EDIT" button on the left side, and audio will be generated in the dialog box on the right side.
                """)
            gr.Markdown("""
                **Para-linguistic Description:**
                - Supported tags include: [Breathing] [Laughter] [Surprise-oh] [Confirmation-en] [Uhm] [Surprise-ah] [Surprise-wa] [Sigh] [Question-ei] [Dissatisfaction-hnn]
                - Example:
                    - Fill in "clone text" field: "Great, the weather is so nice today." Click the "CLONE" button to get audio.
                    - Change "clone text" field to: "Great[Laughter], the weather is so nice today[Surprise-ah]." Click the "EDIT" button to get para-linguistic audio.
                """)

    def register_events(self):
        """Register event handlers"""
        # Create independent state for each session
        state = gr.State(self.init_state())

        self.button_tts.click(self.generate_clone,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])
        self.button_edit.click(self.generate_edit,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])

        self.clean_history_submit.click(self.clear_history, inputs=[state], outputs=[self.chat_box, state])
        self.edit_type.change(
            fn=self.update_edit_info,
            inputs=self.edit_type,
            outputs=self.edit_info,
        )

    def update_edit_info(self, category):
        """Update sub-task dropdown based on main task selection"""
        category_items = get_supported_edit_types()
        choices = category_items.get(category, [])
        value = None if len(choices) == 0 else choices[0]
        return gr.Dropdown(label="Sub-task", choices=choices, value=value)


def launch_demo(args, editx_tab):
    """Launch the gradio demo"""
    with gr.Blocks(theme=gr.themes.Soft(), title="🎙️ Step-Audio-EditX") as demo:
        gr.Markdown("## 🎙️ Step-Audio-EditX")
        gr.Markdown("Audio Editing and Zero-Shot Cloning using Step-Audio_EditX")

        # Register components
        editx_tab.register_components()

        # Register events
        editx_tab.register_events()

    # Launch demo
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share if hasattr(args, 'share') else False
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Step-Audio Edit Demo")
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument("--server-port", type=int, default=7860, help="Demo server port.")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/gradio", help="Save path.")
    parser.add_argument("--share", action="store_true", help="Share gradio app.")

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="Tokenizer model ID for online loading"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID for online loading (if different from model-path)"
    )

    args = parser.parse_args()

    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    model_source = source_mapping[args.model_source]

    logger.info(f"Loading models with source: {args.model_source}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer model ID: {args.tokenizer_model_id}")
    if args.tts_model_id:
        logger.info(f"TTS model ID: {args.tts_model_id}")

    # Initialize models
    try:
        # Load StepAudioTokenizer
        encoder = StepAudioTokenizer(
            os.path.join(args.model_path, "Step-Audio-Tokenizer"),
            model_source=model_source,
            funasr_model_id=args.tokenizer_model_id
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")

        # Initialize common TTS engine directly
        common_tts_engine = StepAudioTTS(
            os.path.join(args.model_path, "Step-Audio-EditX"),
            encoder,
            model_source=model_source,
            tts_model_id=args.tts_model_id
        )
        logger.info("✓ StepCommonAudioTTS loaded successfully")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        exit(1)

    # Create EditxTab instance
    editx_tab = EditxTab(args)

    # Launch demo
    launch_demo(args, editx_tab)