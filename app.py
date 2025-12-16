import gradio as gr
import torch
from pathlib import Path
import torchaudio
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
import tempfile
import os
import locale
import json
import datetime

# 检测系统语言
def get_system_language():
    try:
        lang, _ = locale.getdefaultlocale()
        if lang and 'zh' in lang:
            return 'zh'
        else:
            return 'en'
    except:
        return 'en'

# 设置默认语言
DEFAULT_LANG = get_system_language()

# 翻译字典
translations = {
    'zh': {
        'title': 'GLM-ASR语音识别',
        'description': '使用本地GLM-ASR-Nano-2512模型进行语音识别',
        'audio_label': '上传音频文件或录音',
        'device_label': '设备选择',
        'max_tokens_label': '最大生成token数',
        'transcribe_btn': '开始转录',
        'status_label': '模型加载状态',
        'status_placeholder': '模型加载状态将显示在这里',
        'transcript_label': '转录结果（文本）',
        'transcript_placeholder': '转录结果将显示在这里',
        'srt_label': 'SRT字幕',
        'srt_placeholder': 'SRT格式字幕将显示在这里',
        'json_label': 'JSON格式',
        'json_placeholder': 'JSON格式结果将显示在这里',
        'model_loaded': '模型已加载',
        'model_load_success': '模型加载成功',
        'model_load_fail': '模型加载失败: {error}',
        'model_not_loaded': '模型未加载，请先加载模型',
        'transcribe_fail': '转录失败: {error}',
        'empty_transcript': '[空转录结果]',
        'audio_empty': '音频内容为空或加载失败。'
    },
    'en': {
        'title': 'GLM-ASR Speech Recognition',
        'description': 'Use local GLM-ASR-Nano-2512 model for speech recognition',
        'audio_label': 'Upload audio file or record',
        'device_label': 'Device Selection',
        'max_tokens_label': 'Max New Tokens',
        'transcribe_btn': 'Start Transcription',
        'status_label': 'Model Load Status',
        'status_placeholder': 'Model load status will be displayed here',
        'transcript_label': 'Transcript (Text)',
        'transcript_placeholder': 'Transcript result will be displayed here',
        'srt_label': 'SRT Subtitle',
        'srt_placeholder': 'SRT format subtitle will be displayed here',
        'json_label': 'JSON Format',
        'json_placeholder': 'JSON format result will be displayed here',
        'model_loaded': 'Model already loaded',
        'model_load_success': 'Model loaded successfully',
        'model_load_fail': 'Model load failed: {error}',
        'model_not_loaded': 'Model not loaded, please load model first',
        'transcribe_fail': 'Transcription failed: {error}',
        'empty_transcript': '[Empty transcript result]',
        'audio_empty': 'Audio content is empty or failed to load.'
    }
}

# 获取翻译
def _(key, lang=DEFAULT_LANG, **kwargs):
    if key in translations[lang]:
        return translations[lang][key].format(**kwargs)
    return key

# 从inference.py复制必要的配置和函数
WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
    audio_token_num = min(audio_token_num, 1500 // merge_factor)
    return audio_token_num

def build_prompt(
    audio_path: Path,
    tokenizer,
    feature_extractor: WhisperFeatureExtractor,
    merge_factor: int,
    chunk_seconds: int = 30,
) -> tuple[dict, list]:
    audio_path = Path(audio_path)
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav[:1, :]
    if sr != feature_extractor.sampling_rate:
        wav = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)(wav)

    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")

    audios = []
    audio_offsets = []
    audio_length = []
    chunk_timestamps = []  # 存储每个chunk的开始和结束时间（秒）
    chunk_size = chunk_seconds * feature_extractor.sampling_rate
    
    for start in range(0, wav.shape[1], chunk_size):
        end = start + chunk_size
        chunk = wav[:, start:end]
        
        # 计算chunk的时间戳
        start_time = start / feature_extractor.sampling_rate
        end_time = end / feature_extractor.sampling_rate
        chunk_timestamps.append((start_time, end_time))
        
        mel = feature_extractor(
            chunk.numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]
        audios.append(mel)
        seconds = chunk.shape[1] / feature_extractor.sampling_rate
        num_tokens = get_audio_token_length(seconds, merge_factor)
        tokens += tokenizer.encode("<|begin_of_audio|>")
        audio_offsets.append(len(tokens))
        tokens += [0] * num_tokens
        tokens += tokenizer.encode("<|end_of_audio|>")
        audio_length.append(num_tokens)

    if not audios:
        raise ValueError(_('audio_empty'))

    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")

    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")

    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": torch.cat(audios, dim=0),
        "audio_offsets": [audio_offsets],
        "audio_length": [audio_length],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }
    return batch, chunk_timestamps

def prepare_inputs(batch: dict, device: torch.device) -> tuple[dict, int]:
    tokens = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audios = batch["audios"].to(device)
    model_inputs = {
        "inputs": tokens,
        "attention_mask": attention_mask,
        "audios": audios.to(torch.bfloat16),
        "audio_offsets": batch["audio_offsets"],
        "audio_length": batch["audio_length"],
    }
    return model_inputs, tokens.size(1)

# 辅助函数：将秒转换为SRT时间格式
def seconds_to_srt_time(seconds):
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

# 辅助函数：生成SRT格式字幕
def generate_srt(transcript_segments):
    srt_content = []
    for i, (start_time, end_time, text) in enumerate(transcript_segments, 1):
        start_srt = seconds_to_srt_time(start_time)
        end_srt = seconds_to_srt_time(end_time)
        srt_content.append(str(i))
        srt_content.append(f"{start_srt} --> {end_srt}")
        srt_content.append(text)
        srt_content.append("")
    return "\n".join(srt_content)

# 辅助函数：生成JSON格式结果
def generate_json(transcript, transcript_segments):
    result = {
        "transcript": transcript,
        "segments": [
            {
                "id": i + 1,
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "text": text
            }
            for i, (start_time, end_time, text) in enumerate(transcript_segments)
        ],
        "language": DEFAULT_LANG,
        "timestamp": datetime.datetime.now().isoformat()
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

# 模型加载器
class GLMASRModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.config = None
        self.device = None
    
    def load_model(self, checkpoint_dir, device):
        if self.model is not None and self.device == device:
            return _('model_loaded')
        
        try:
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self.feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
            
            self.config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
            self.model.eval()
            
            return _('model_load_success')
        except Exception as e:
            return _('model_load_fail', error=str(e))
    
    def transcribe(self, audio_path, max_new_tokens=128):
        if self.model is None:
            return _('model_not_loaded'), [], "", ""
        
        try:
            batch, chunk_timestamps = build_prompt(
                audio_path,
                self.tokenizer,
                self.feature_extractor,
                merge_factor=self.config.merge_factor,
            )
            
            model_inputs, prompt_len = prepare_inputs(batch, self.device)
            
            with torch.inference_mode():
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            transcript_ids = generated[0, prompt_len:].cpu().tolist()
            transcript = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
            
            # 处理分段转录结果
            # 注意：当前模型可能不返回分段结果，这里简化处理
            # 如果模型支持分段输出，需要根据实际情况解析
            if transcript:
                # 简单分割为句子（实际应用中可能需要更复杂的逻辑）
                sentences = [sentence.strip() for sentence in transcript.split('.') if sentence.strip()]
                if not sentences:
                    sentences = [transcript]
                
                # 平均分配时间戳（实际应用中需要模型支持时间戳输出）
                total_duration = chunk_timestamps[-1][1] if chunk_timestamps else 0
                avg_sentence_duration = total_duration / len(sentences)
                
                transcript_segments = []
                current_time = 0.0
                for sentence in sentences:
                    end_time = min(current_time + avg_sentence_duration, total_duration)
                    transcript_segments.append((current_time, end_time, sentence + '.'))
                    current_time = end_time
            else:
                transcript_segments = []
            
            # 生成SRT和JSON格式
            srt_content = generate_srt(transcript_segments)
            json_content = generate_json(transcript, transcript_segments)
            
            return transcript or _('empty_transcript'), transcript_segments, srt_content, json_content
        except Exception as e:
            return _('transcribe_fail', error=str(e)), [], "", ""

# 初始化模型实例
model_instance = GLMASRModel()

# 默认模型路径
DEFAULT_MODEL_PATH = str(Path(__file__).parent / "zai-org" / "GLM-ASR-Nano-2512")

# Gradio界面函数
def transcribe_audio(audio_path, device, max_new_tokens):
    # 加载模型
    load_status = model_instance.load_model(DEFAULT_MODEL_PATH, device)
    if "失败" in load_status:
        return load_status, "", "", ""
    
    # 转录音频
    transcript, segments, srt_content, json_content = model_instance.transcribe(audio_path, max_new_tokens)
    
    return load_status, transcript, srt_content, json_content

# 创建Gradio界面
with gr.Blocks(title=_('title')) as demo:
    gr.Markdown(f"# {_('title')}")
    gr.Markdown(_('description'))
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label=_('audio_label')
            )
            
            with gr.Row():
                device_dropdown = gr.Dropdown(
                    choices=["cpu", "cuda", "mps"],
                    value="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
                    label=_('device_label')
                )
                
                max_new_tokens_slider = gr.Slider(
                    minimum=32,
                    maximum=512,
                    value=128,
                    step=1,
                    label=_('max_tokens_label')
                )
            
            transcribe_btn = gr.Button(_('transcribe_btn'), variant="primary")
        
        with gr.Column(scale=2):
            load_status = gr.Textbox(
                label=_('status_label'),
                interactive=False,
                placeholder=_('status_placeholder')
            )
            
            with gr.Tabs():
                with gr.TabItem(_('transcript_label')):
                    transcript_output = gr.Textbox(
                        label=_('transcript_label'),
                        interactive=False,
                        placeholder=_('transcript_placeholder'),
                        lines=10
                    )
                
                with gr.TabItem(_('srt_label')):
                    srt_output = gr.Textbox(
                        label=_('srt_label'),
                        interactive=False,
                        placeholder=_('srt_placeholder'),
                        lines=10
                    )
                
                with gr.TabItem(_('json_label')):
                    json_output = gr.Textbox(
                        label=_('json_label'),
                        interactive=False,
                        placeholder=_('json_placeholder'),
                        lines=10
                    )
    
    # 示例音频
    gr.Examples(
        examples=[
            [str(Path(__file__).parent / "examples" / "example_zh.wav")],
            [str(Path(__file__).parent / "examples" / "example_en.wav")]
        ],
        inputs=[audio_input],
        outputs=[load_status, transcript_output, srt_output, json_output],
        fn=transcribe_audio,
        cache_examples=False
    )
    
    # 按钮点击事件
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, device_dropdown, max_new_tokens_slider],
        outputs=[load_status, transcript_output, srt_output, json_output]
    )

# 启动界面
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
