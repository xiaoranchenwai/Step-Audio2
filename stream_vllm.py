"""
FastRTC + Gradio 实时流式对话系统 for Step-Audio-2 (支持 VAD 和打断)

使用 fastrtc.WebRTC 实现持续语音输入，无需手动停止录音:
- VAD 自动检测语音活动
- 支持打断 AI 回复
- 实时流式文本和音频输出
- 持续对话无需点击

运行方式:
    pip install fastrtc librosa webrtcvad
    
    python step_audio2_streaming.py \
        --ssl-certfile cert.pem --ssl-keyfile key.pem --ssl-no-verify
    
    # 或使用 share
    python step_audio2_streaming.py --share
"""

import argparse
import gradio as gr
import fastrtc
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime
import threading
import time
import wave
from typing import Generator, override
from queue import Queue, Empty

from stepaudio2vllm import StepAudio2
from token2wav import Token2wav

# VAD 配置
try:
    import webrtcvad
    import librosa
    VAD_AVAILABLE = True
except ImportError as e:
    VAD_AVAILABLE = False
    print(f"警告: VAD依赖未安装 ({e})")
    print("安装: pip install webrtcvad librosa")

CHUNK_SIZE = 25


class StepAudio2Service:
    """Step-Audio-2 服务封装"""
    
    def __init__(self, api_url: str, model_name: str, token2wav_path: str, prompt_wav_path: str):
        self.model = StepAudio2(api_url, model_name)
        self.token2wav = Token2wav(token2wav_path)
        self.prompt_wav = prompt_wav_path
        self.tools = []
        self.generation_lock = threading.Lock()
        self.token2wav.set_stream_cache(self.prompt_wav)
        
    def save_audio_temp(self, sr: int, audio_data: np.ndarray) -> str:
        """保存音频到临时文件"""
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav_path = temp_wav.name
        temp_wav.close()
        
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            # 确保是 int16 格式
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        
        return wav_path
    
    def generate_response(
        self, 
        audio_input: tuple[int, np.ndarray],
        history: list,
        system_prompt: str = None
    ) -> Generator:
        """流式生成响应"""
        
        sr, audio_data = audio_input
        
        # 更新历史记录
        user_msg = "🎤 [语音消息]"
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ""})
        yield fastrtc.AdditionalOutputs(history)

        try:
            # 保存音频到临时文件
            temp_audio_path = self.save_audio_temp(sr, audio_data)
            print(f"[Step-Audio-2] Processing audio from {temp_audio_path}")

            # 构建系统提示
            if system_prompt is None:
                system_prompt = (
                    f"你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。"
                    f"今天是{datetime.now().strftime('%Y年%m月%d日')}。"
                    f"请用默认女声与用户交流，回复要简洁友好。"
                )
            
            # 构建对话历史
            step_history = [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": [{"type": "audio", "audio": temp_audio_path}]},
                {"role": "assistant", "content": "<tts_start>", "eot": False}
            ]

            # 初始化 token2wav 缓存
            #self.token2wav.set_stream_cache(self.prompt_wav)
            
            # 用于累积结果
            full_text = ""
            audio_chunks = []
            pcm_buffer = []
            
            with self.generation_lock:
                # 流式生成
                for line, text, audio in self.model.stream(
                    step_history,
                    tools=self.tools,
                    max_tokens=4096,
                    repetition_penalty=1.05,
                    top_p=0.9,
                    temperature=0.7
                ):
                    # 处理文本流
                    if text:
                        full_text += text
                        history[-1]["content"] = full_text
                        yield fastrtc.AdditionalOutputs(history)
                    
                    # 处理音频流
                    if audio:
                        pcm_buffer += audio
                        
                        # 当缓冲区足够大时，生成音频
                        if len(pcm_buffer) >= CHUNK_SIZE + self.token2wav.flow.pre_lookahead_len:
                            chunk_to_decode = pcm_buffer[:CHUNK_SIZE + self.token2wav.flow.pre_lookahead_len]
                            wav_chunk = self.token2wav.stream(
                                chunk_to_decode,
                                prompt_wav=self.prompt_wav,
                                last_chunk=False
                            )
                            
                            # 将 PCM bytes 转换为 numpy array (int16)
                            audio_np = np.frombuffer(wav_chunk, dtype=np.int16)
                            audio_chunks.append(audio_np)
                            pcm_buffer = pcm_buffer[CHUNK_SIZE:]
                            
                            # 输出累积的音频 (int16 格式，24kHz)
                            full_audio = np.concatenate(audio_chunks)
                            yield (24000, full_audio)
                
                # 处理剩余的音频缓冲
                if pcm_buffer:
                    wav_chunk = self.token2wav.stream(
                        pcm_buffer,
                        prompt_wav=self.prompt_wav,
                        last_chunk=True
                    )
                    audio_np = np.frombuffer(wav_chunk, dtype=np.int16)
                    audio_chunks.append(audio_np)
                    full_audio = np.concatenate(audio_chunks)
                    yield (24000, full_audio)

            # 最终输出
            yield fastrtc.AdditionalOutputs(history)

            # 清理临时文件
            try:
                Path(temp_audio_path).unlink()
            except:
                pass

        except GeneratorExit:
            print("[Step-Audio-2] Generation interrupted by VAD")
            raise
        except Exception as e:
            print(f"[Step-Audio-2] Error: {e}")
            import traceback
            traceback.print_exc()
            history[-1]["content"] += f"\n[错误: {e}]"
            yield fastrtc.AdditionalOutputs(history)


class RealTimeVAD:
    """实时 VAD 处理器 - 使用新的实现方式"""
    
    def __init__(self, src_sr=24000, vad_sr=16000, frame_duration_ms=30, mode=3):
        self.src_sr = src_sr
        self.vad_sr = vad_sr
        self.frame_duration_ms = frame_duration_ms
        
        if not VAD_AVAILABLE:
            raise ImportError("webrtcvad 和 librosa 必须安装")
        
        self.vad = webrtcvad.Vad(mode)
        
        # 计算每帧样本数
        self.samples_per_frame = int(vad_sr * frame_duration_ms / 1000)
        
        # 重采样后音频的积累缓冲区
        self.vad_buffer = np.array([], dtype=np.int16)
        
        # 状态机相关
        self.audio_buffer = []  # 存储原始采样率的音频
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.silence_threshold = 5
        self.speech_threshold = 30
        self.is_ai_speaking = False
        self.frame_count = 0
    
    class VADEvent:
        def __init__(self):
            self.interrupt_signal = False
            self.full_audio: tuple[int, np.ndarray] | None = None
    
    def process_chunk(self, audio_chunk: bytes):
        """
        处理一段实时音频 chunk，返回这一段中所有帧的 VAD 结果
        """
        if not audio_chunk:
            return []
        
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # 转换为 float32
        float_audio = audio_data.astype(np.float32) / 32768.0
        
        # 重采样
        if self.src_sr != self.vad_sr:
            resampled = librosa.resample(float_audio, orig_sr=self.src_sr, target_sr=self.vad_sr)
        else:
            resampled = float_audio
        
        # 转回 int16
        resampled_int16 = np.clip(resampled * 32767.0, -32768, 32767).round().astype(np.int16)
        
        # 添加到缓冲区
        self.vad_buffer = np.concatenate((self.vad_buffer, resampled_int16))
        
        results_this_chunk = []
        while len(self.vad_buffer) >= self.samples_per_frame:
            frame = self.vad_buffer[:self.samples_per_frame]
            frame_bytes = frame.tobytes()
            
            try:
                is_speech = self.vad.is_speech(frame_bytes, self.vad_sr)
            except Exception as e:
                print(f"[VAD] 检测错误: {e}")
                is_speech = False
            
            results_this_chunk.append(is_speech)
            self.vad_buffer = self.vad_buffer[self.samples_per_frame:]
        
        return results_this_chunk
    
    def process(self, audio_data: np.ndarray):
        """处理音频帧并产生事件"""
        self.frame_count += 1
        event = self.VADEvent()
        
        # 将 numpy array 转换为 bytes
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # 使用新的 VAD 处理
        vad_results = self.process_chunk(audio_bytes)
        
        # 如果没有完整的帧，直接返回
        if not vad_results:
            yield event
            return
        
        # 使用最后一帧的结果作为当前状态（或者可以使用多数投票）
        is_speech = vad_results[-1]  # 或者: is_speech = sum(vad_results) > len(vad_results) / 2
        
        # 详细的状态日志
        if self.frame_count % 20 == 0:
            print(f"[VAD] 帧{self.frame_count} | is_speaking:{self.is_speaking} | speech_frames:{self.speech_frames} | silence_frames:{self.silence_frames} | buffer_size:{len(self.audio_buffer)} | is_speech:{is_speech}")
        
        # 如果正在 AI 说话时检测到人声，触发打断
        if self.is_ai_speaking and is_speech:
            print("[VAD] ⚠️ 检测到打断信号！")
            event.interrupt_signal = True
            self.is_ai_speaking = False
            self.audio_buffer = []
            self.is_speaking = False
            self.speech_frames = 0
            self.silence_frames = 0
            yield event
            return
        
        # VAD 状态机
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.is_speaking and self.speech_frames >= self.speech_threshold:
                print(f"[VAD] ✅✅✅ 开始说话！(语音帧: {self.speech_frames})")
                self.is_speaking = True
                self.audio_buffer = []
            
            if self.is_speaking:
                self.audio_buffer.append(audio_data)
                if len(self.audio_buffer) % 50 == 0:
                    duration = len(self.audio_buffer) * len(audio_data) / self.src_sr
                    print(f"[VAD] 📝 录音中... 缓冲区: {len(self.audio_buffer)} 帧 ({duration:.2f}秒)")
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            if self.is_speaking:
                self.audio_buffer.append(audio_data)
                
                print(f"[VAD] 🔇 静音中: {self.silence_frames}/{self.silence_threshold} (缓冲区: {len(self.audio_buffer)} 帧)")
                
                if self.silence_frames >= self.silence_threshold:
                    print(f"[VAD] ✅✅✅ 说话结束！触发生成！")
                    print(f"[VAD] 最终缓冲区: {len(self.audio_buffer)} 帧")
                    self.is_speaking = False
                    
                    # 合并音频
                    full_audio = np.concatenate(self.audio_buffer)
                    audio_duration = len(full_audio) / self.src_sr
                    print(f"[VAD] 🎵 音频时长: {audio_duration:.2f} 秒, 采样点: {len(full_audio)}")
                    
                    event.full_audio = (self.src_sr, full_audio)
                    self.audio_buffer = []
                    self.silence_frames = 0
                    self.speech_frames = 0
                    
                    # 标记 AI 开始说话
                    self.is_ai_speaking = True
                    
                    print(f"[VAD] 🚀 准备返回 full_audio 事件")
                    yield event
                    return
        
        yield event


type StreamerGenerator = Generator[fastrtc.tracks.EmitType, None, None]


class VADStreamHandler(fastrtc.StreamHandler):
    """FastRTC Stream Handler 带 VAD 支持"""
    
    def __init__(
        self,
        step_service: StepAudio2Service,
        input_sample_rate: int = 24000,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            output_frame_size=None,
            input_sample_rate=input_sample_rate,
        )
        self.step_service = step_service
        self.realtime_vad = RealTimeVAD(src_sr=input_sample_rate)
        self.generator: StreamerGenerator | None = None
        self.latest_history = []
        self.close_requested = threading.Event()

    @override
    def emit(self) -> fastrtc.tracks.EmitType:
        """发送数据到前端"""
        if self.close_requested.is_set():
            if self.generator:
                print("[Handler] 关闭生成器（打断）")
                self.generator.close()
                self.generator = None
            self.close_requested.clear()
            return None
        
        if self.generator is None:
            return None

        try:
            return next(self.generator)
        except StopIteration:
            print("[Handler] 生成完成")
            self.generator = None
            # 通知 VAD AI 已停止说话
            self.realtime_vad.is_ai_speaking = False
            return None
        except Exception as e:
            print(f"[Handler] 生成器错误: {e}")
            import traceback
            traceback.print_exc()
            self.generator = None
            self.realtime_vad.is_ai_speaking = False
            return None

    @override
    def receive(self, frame: tuple[int, np.ndarray]):
        """接收来自前端的音频帧"""
        sr, audio_data = frame
        
        # 调试：打印音频数据信息
        if hasattr(self, '_frame_count'):
            self._frame_count += 1
        else:
            self._frame_count = 0
            print("[Handler] ========== 初始化 ==========")
            
        if self._frame_count % 100 == 0:
            print(f"\n[Handler] === 帧 {self._frame_count} ===")
            print(f"  采样率: {sr}, 形状: {audio_data.shape}, 类型: {audio_data.dtype}")
            print(f"  范围: [{audio_data.min():.6f}, {audio_data.max():.6f}]")
            print(f"  能量: {np.abs(audio_data).mean():.6f}")
        
        event_count = 0
        for event in self.realtime_vad.process(audio_data):
            event_count += 1
            
            if event.interrupt_signal:
                print("[Handler] ⚠️⚠️⚠️ >>> 检测到打断信号 <<<")
                self.close_requested.set()
                self.clear_queue()

            if event.full_audio is not None:
                print("\n" + "="*70)
                print("[Handler] 🎉🎉🎉 >>> 接收到完整音频！<<<")
                print("="*70)
                
                sr_full, audio_full = event.full_audio
                print(f"[Handler] 完整音频信息:")
                print(f"  - 采样率: {sr_full} Hz")
                print(f"  - 采样点数: {len(audio_full)}")
                print(f"  - 时长: {len(audio_full) / sr_full:.2f} 秒")
                print(f"  - 数据类型: {audio_full.dtype}")
                
                if self.close_requested.is_set():
                    print("[Handler] 清除打断标志")
                    self.close_requested.clear()
                
                # 同步获取最新的历史记录
                print("[Handler] 等待参数同步...")
                self.wait_for_args()
                if len(self.latest_args) > 0:
                    self.latest_history = self.latest_args[-1]
                    print(f"[Handler] ✅ 对话历史长度: {len(self.latest_history)}")
                else:
                    print("[Handler] ⚠️ 没有历史记录，使用空列表")
                    self.latest_history = []
                
                # 使用 Step-Audio-2 服务生成响应
                print("[Handler] 🚀 正在调用 generate_response...")
                try:
                    self.generator = self.step_service.generate_response(
                        event.full_audio,
                        self.latest_history
                    )
                    print("[Handler] ✅ generate_response 已启动，generator:", self.generator)
                except Exception as e:
                    print(f"[Handler] ❌ generate_response 启动失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                print("="*70 + "\n")
        
        if self._frame_count % 100 == 0 and event_count > 0:
            print(f"[Handler] 处理了 {event_count} 个事件")

    @override
    def copy(self):
        """创建新的处理器副本"""
        return VADStreamHandler(
            step_service=self.step_service,
            input_sample_rate=self.input_sample_rate,
        )


def build_interface(service: StepAudio2Service) -> gr.Blocks:
    """构建 Gradio 界面"""
    
    with gr.Blocks(title="Step-Audio-2 实时对话系统") as demo:
        gr.Markdown("### 🎙️ Step-Audio-2 实时语音对话 (支持 VAD 和打断)")
        gr.Markdown(
            "**说明**: 开始说话会自动触发 VAD。在 AI 回复过程中说话，会触发打断信号并停止当前回复。\n\n"
            "✨ **无需点击** - 直接说话即可，系统会自动检测语音活动"
        )

        with gr.Row():
            with gr.Column(scale=1):
                webrtc = fastrtc.WebRTC(
                    label="🎙️ 语音通话",
                    mode="send-receive",
                    modality="audio",
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清除历史", size="sm")
                    manual_trigger_btn = gr.Button("⚡ 手动触发（测试用）", size="sm", variant="secondary")
                
                gr.Markdown("""
                **调试模式**: 如果 VAD 自动触发有问题，可以：
                1. 说话后点击"手动触发"按钮强制结束录音
                2. 查看控制台日志了解 VAD 状态
                """)
            
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 对话记录",
                    height=500,
                    type="messages"
                )

        chat_history = gr.State([])

        # 初始化 Handler
        handler = VADStreamHandler(
            step_service=service,
            input_sample_rate=24000
        )

        # 绑定 Stream
        webrtc.stream(
            handler,
            inputs=[webrtc, chat_history],
            outputs=[webrtc],
            time_limit=3600  # 1小时
        )

        # 监听额外输出（对话历史更新）
        webrtc.on_additional_outputs(
            lambda h: h,
            outputs=[chatbot],
            queue=False,
            show_progress="hidden"
        )

        # 清除历史
        clear_btn.click(
            lambda: ([], []),
            outputs=[chat_history, chatbot]
        )
        
        # 手动触发（测试用）
        def manual_trigger():
            """手动触发录音结束"""
            print("\n" + "="*70)
            print("[手动触发] 用户点击了手动触发按钮")
            print("="*70)
            
            if handler.realtime_vad.is_speaking and len(handler.realtime_vad.audio_buffer) > 0:
                print(f"[手动触发] 当前正在说话，缓冲区有 {len(handler.realtime_vad.audio_buffer)} 帧")
                print("[手动触发] 强制设置静音帧数以触发结束")
                handler.realtime_vad.silence_frames = handler.realtime_vad.silence_threshold
                return "✅ 已手动触发，请等待处理..."
            else:
                print(f"[手动触发] 当前未在说话状态 (is_speaking={handler.realtime_vad.is_speaking})")
                print(f"[手动触发] 缓冲区大小: {len(handler.realtime_vad.audio_buffer)}")
                return "⚠️ 未检测到录音数据"
        
        manual_trigger_btn.click(
            fn=manual_trigger,
            outputs=[]
        )
        
        gr.Markdown("""
        ---
        ### 💡 使用说明
        1. **点击"开始"** 启动语音通话
        2. **直接说话** - 系统自动检测语音开始和结束
        3. **等待回复** - AI 会实时生成文本和语音
        4. **随时打断** - 在 AI 说话时开始说话，会自动打断
        5. **继续对话** - 打断后可以继续说话或等待
        
        ### 🔧 技术特性
        - ✅ VAD 自动语音检测（基于 webrtcvad + librosa）
        - ✅ 持续通话无需点击
        - ✅ 支持打断功能
        - ✅ 流式文本生成
        - ✅ 流式语音合成
        - ✅ 低延迟实时交互
        
        ### ⚙️ VAD 参数
        - 源采样率: 24kHz
        - VAD 采样率: 16kHz
        - 激进度: 3 (0-3)
        - 语音阈值: 3 帧
        - 静音阈值: 15 帧
        - 帧时长: 30ms
        
        ### 🐛 调试提示
        - 查看控制台日志，了解 VAD 检测状态
        - 如果一直不触发，可能是麦克风音量太小
        - 建议靠近麦克风，清晰说话
        """)
    
    return demo


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Step-Audio-2 实时流式对话系统")
    
    # 模型配置
    parser.add_argument(
        "--api-url",
        default="http://10.250.2.26:8005/v1/chat/completions",
        help="Step-Audio-2 API 地址"
    )
    parser.add_argument(
        "--model-name",
        default="step-audio-2-mini",
        help="模型名称"
    )
    parser.add_argument(
        "--token2wav-path",
        default="/home/user/cx/new_data/Step-Audio2/token2wav",
        help="Token2Wav 模型路径"
    )
    parser.add_argument(
        "--prompt-wav-path",
        default="/home/user/cx/new_data/Step-Audio2/assets/default_female.wav",
        help="提示音频文件路径"
    )
    
    # 服务器配置
    parser.add_argument(
        "--server-name",
        default="0.0.0.0",
        help="服务器地址"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7862,
        help="服务器端口"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建 Gradio 公共分享链接"
    )
    
    # SSL 配置
    parser.add_argument(
        "--ssl-certfile",
        default="/home/user/cx/new_data/MiMo-Audio/cert.pem",
        help="SSL 证书文件路径"
    )
    parser.add_argument(
        "--ssl-keyfile",
        default="/home/user/cx/new_data/MiMo-Audio/key.pem",
        help="SSL 私钥文件路径"
    )
    parser.add_argument(
        "--ssl-keyfile-password",
        default=None,
        help="SSL 私钥密码"
    )
    parser.add_argument(
        "--ssl-no-verify",
        default=True,
        help="禁用 SSL 证书验证"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("Step-Audio-2 实时流式对话系统 (FastRTC + VAD)")
    print("=" * 70)
    
    print(f"\n📡 模型配置:")
    print(f"  - API 地址: {args.api_url}")
    print(f"  - 模型名称: {args.model_name}")
    print(f"  - Token2Wav: {args.token2wav_path}")
    print(f"  - 提示音频: {args.prompt_wav_path}")
    
    print(f"\n🌐 服务器配置:")
    print(f"  - 地址: {args.server_name}")
    print(f"  - 端口: {args.server_port}")
    print(f"  - 分享链接: {'启用' if args.share else '禁用'}")
    
    if not VAD_AVAILABLE:
        print("\n⚠️  警告: webrtcvad 未安装，将使用简单的能量检测")
        print("   推荐安装: pip install webrtcvad")
    
    # SSL 配置检查
    ssl_enabled = False
    if args.ssl_certfile and args.ssl_keyfile:
        print(f"\n🔒 SSL/HTTPS 配置:")
        print(f"  - 证书文件: {args.ssl_certfile}")
        print(f"  - 私钥文件: {args.ssl_keyfile}")
        print(f"  - 证书验证: {'禁用' if args.ssl_no_verify else '启用'}")
        ssl_enabled = True
    
    # 初始化服务
    print("\n🔧 初始化服务...")
    try:
        service = StepAudio2Service(
            api_url=args.api_url,
            model_name=args.model_name,
            token2wav_path=args.token2wav_path,
            prompt_wav_path=args.prompt_wav_path,
        )
        print("✅ 服务初始化成功")
    except Exception as e:
        print(f"❌ 服务初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 构建界面
    print("🎨 构建界面...")
    demo = build_interface(service)
    
    # 启动服务
    print("\n🚀 启动服务...")
    
    protocol = "https" if ssl_enabled else "http"
    
    if args.server_name == "0.0.0.0":
        print(f"   访问地址: {protocol}://localhost:{args.server_port}")
        print(f"   局域网访问: {protocol}://<your-ip>:{args.server_port}")
    else:
        print(f"   访问地址: {protocol}://{args.server_name}:{args.server_port}")
    
    if args.share:
        print("   Gradio 分享链接: 启动后自动生成...")
    
    print("\n⚠️  重要提示:")
    print("   - 浏览器需要麦克风权限")
    print("   - WebRTC 需要安全上下文（HTTPS 或 localhost）")
    print("   - 建议使用 Chrome/Edge 浏览器")
    print("   - 点击'开始'后直接说话，无需其他操作")
    print("\n📦 依赖检查:")
    print("   - pip install fastrtc")
    print("   - pip install webrtcvad (可选，提供更好的 VAD)")
    
    print("\n" + "=" * 70)
    
    # 启动 Gradio
    try:
        launch_kwargs = {
            "server_name": args.server_name,
            "server_port": args.server_port,
            "share": args.share,
            "debug": True,
        }
        
        if ssl_enabled:
            launch_kwargs.update({
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
                "ssl_keyfile_password": args.ssl_keyfile_password,
                "ssl_verify": not args.ssl_no_verify,
            })
        
        demo.launch(**launch_kwargs)
        
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()