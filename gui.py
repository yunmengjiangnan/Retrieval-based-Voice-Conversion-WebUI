# 导入标准库并指定导入顺序（标准库→第三方库→自定义库）
import os
import sys
import re
import json
import time
import traceback
from multiprocessing import Queue, cpu_count, shared_memory, Process
import shutil

# 导入第三方库
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
import librosa
import sounddevice as sd
import FreeSimpleGUI as sg
from dotenv import load_dotenv

# 导入自定义库
from configs.config_manager import ConfigManager
from configs import Config
from infer.lib.audio import AudioIoProcess
from infer.lib.rtrvc import RVC as RTRVC
from infer.lib.rvcmd import check_all_assets, download_all_assets
from infer.modules.gui import TorchGate
from i18n.i18n import I18nAuto


# -------------------------- 全局配置与初始化 --------------------------
def init_global_env():
    """初始化全局环境：加载环境变量、设置线程数、初始化国际化等"""
    # 加载环境变量（.env和sha256.env）
    load_dotenv()
    load_dotenv("sha256.env")

    # 设置OMP线程数，避免CPU过度占用
    os.environ["OMP_NUM_THREADS"] = "4"

    # macOS特殊配置：启用MPS回退（当MPS不支持某些操作时使用CPU）
    if sys.platform == "darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 设置工作目录并添加到系统路径
    current_work_dir = os.getcwd()
    sys.path.append(current_work_dir)

    # 初始化国际化工具
    global i18n
    i18n = I18nAuto()

    # 初始化全局标志：控制语音转换状态与推理时间显示
    global flag_vc
    flag_vc = False

    # 初始化全局进程池配置（用于Harvest基频提取）
    global max_harvest_cpu
    max_harvest_cpu = min(cpu_count(), 8)  # 限制最大CPU进程数为8，平衡性能与资源占用

    return current_work_dir


# -------------------------- 工具函数 --------------------------
def print_with_format(format_str: str, *args):
    """格式化打印函数，兼容无参数场景"""
    if not args:
        print(format_str)
    else:
        print(format_str % args)


def phase_vocoder(
    prev_audio: torch.Tensor,
    curr_audio: torch.Tensor,
    fade_out_window: torch.Tensor,
    fade_in_window: torch.Tensor,
) -> torch.Tensor:
    """
    相位声码器：用于音频块拼接时保持相位连续性，提升拼接质量
    参数：
        prev_audio: 历史音频块（用于拼接的前序数据）
        curr_audio: 当前音频块（待拼接的新数据）
        fade_out_window: 淡出窗口（用于历史音频块的过渡）
        fade_in_window: 淡入窗口（用于当前音频块的过渡）
    返回：
        相位对齐后的拼接音频块
    """
    # 计算混合窗口（平衡淡出与淡入的过渡效果）
    mix_window = torch.sqrt(fade_out_window * fade_in_window)

    # 对输入音频块应用窗口并进行FFT（频域处理）
    prev_fft = torch.fft.rfft(prev_audio * mix_window)
    curr_fft = torch.fft.rfft(curr_audio * mix_window)

    # 计算幅度和相位：用于相位对齐
    amp_sum = torch.abs(prev_fft) + torch.abs(curr_fft)
    prev_phase = torch.angle(prev_fft)
    curr_phase = torch.angle(curr_fft)

    # 处理偶/奇数长度的FFT结果（修正幅度计算）
    audio_len = prev_audio.shape[0]
    if audio_len % 2 == 0:
        amp_sum[1:-1] *= 2  # 偶数长度：中间部分加倍
    else:
        amp_sum[1:] *= 2  # 奇数长度：非直流部分加倍

    # 计算相位差并归一化（避免相位缠绕）
    phase_diff = curr_phase - prev_phase
    phase_diff = phase_diff - 2 * np.pi * torch.floor(phase_diff / (2 * np.pi) + 0.5)

    # 生成频率轴与时间轴（用于频域到时域的转换）
    freq_axis = (
        2 * np.pi * torch.arange(audio_len // 2 + 1).to(prev_audio.device) + phase_diff
    )
    time_axis = torch.arange(audio_len).unsqueeze(-1).to(prev_audio.device) / audio_len

    # 时域重建：结合历史、当前音频块及相位对齐结果
    result = (
        prev_audio * (fade_out_window**2)  # 历史块淡出
        + curr_audio * (fade_in_window**2)  # 当前块淡入
        + torch.sum(amp_sum * torch.cos(freq_axis * time_axis + prev_phase), dim=-1)
        * mix_window
        / audio_len  # 相位对齐部分
    )

    return result


# -------------------------- Harvest基频提取进程类 --------------------------
class HarvestPitchProcess(Process):
    """
    基于pyworld.Harvest的基频提取进程类（多进程实现，提升处理速度）
    功能：从输入队列获取音频数据，计算基频后存入结果字典，完成后通知输出队列
    """

    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__(daemon=True)  # 设为守护进程，主进程退出时自动关闭
        self.input_queue = input_queue  # 输入队列：存储待处理的音频数据
        self.output_queue = output_queue  # 输出队列：通知处理完成的时间戳

    def run(self) -> None:
        """进程主逻辑：循环从队列获取任务并处理"""
        # 延迟导入pyworld（避免主进程初始化时加载）
        import pyworld

        while True:
            # 从输入队列获取任务：(索引, 音频数据, 结果字典, 总进程数, 时间戳)
            task_idx, audio_data, result_dict, total_processes, timestamp = (
                self.input_queue.get()
            )

            # 调用pyworld.Harvest计算基频（16kHz为固定采样率，匹配语音转换模型要求）
            f0, _ = pyworld.harvest(
                audio_data.astype(np.double),
                fs=16000,
                f0_ceil=1100,  # 最高基频（1100Hz，覆盖大多数人声范围）
                f0_floor=50,  # 最低基频（50Hz，覆盖大多数人声范围）
                frame_period=10,  # 帧周期（10ms，平衡精度与速度）
            )

            # 将结果存入字典（按索引标记，便于主进程汇总）
            result_dict[task_idx] = f0

            # 当所有进程都完成当前批次处理时，通知主进程
            if len(result_dict.keys()) >= total_processes:
                self.output_queue.put(timestamp)


# -------------------------- GUI配置类 --------------------------
class VoiceConversionGUIConfig:
    """
    语音转换GUI配置类：存储所有GUI相关的用户设置与参数
    职责：统一管理配置项，避免散落在主类中，提升可维护性
    """

    def __init__(self):
        # 模型与索引文件路径
        self.model_path: str = ""  # 模型文件（.pth）路径
        self.index_path: str = ""  # 特征索引文件（.index）路径

        # 音频处理参数
        self.pitch_shift: int = 0  # 音高偏移（半音，范围-24~24）
        self.formant_shift: float = 0.0  # 共振峰偏移（调整音色，范围-5~5）
        self.sr_mode: str = (
            "sr_model"  # 采样率模式（sr_model:模型采样率；sr_device:设备采样率）
        )
        self.block_duration: float = 0.25  # 音频块时长（秒，影响实时性，范围0.02~1.5）
        self.silence_threshold: int = -60  # 静音检测阈值（dB，低于此值判定为静音）
        self.crossfade_duration: float = 0.05  # 交叉fade时长（秒，避免拼接爆音）
        self.extra_infer_duration: float = 2.5  # 额外推理时长（秒，避免尾部截断）

        # 功能开关
        self.enable_input_denoise: bool = False  # 输入降噪开关
        self.enable_output_denoise: bool = False  # 输出降噪开关
        self.enable_phase_vocoder: bool = False  # 相位声码器开关（提升拼接质量）

        # 混合与匹配参数
        self.rms_mix_ratio: float = 0.0  # 音量匹配比例（0~1，0为完全匹配输入音量）
        self.index_search_ratio: float = 0.0  # 特征索引搜索比例（0~1，1为完全使用索引）

        # 硬件与性能参数
        self.harvest_cpu_count: int = min(
            max_harvest_cpu, 4
        )  # Harvest进程数（1~max_harvest_cpu）
        self.f0_extract_method: str = (
            "fcpe"  # 基频提取方法（pm/dio/harvest/crepe/rmvpe/fcpe）
        )

        # 音频设备配置
        self.hostapi_name: str = ""  # 音频主机API名称（如WASAPI/ASIO）
        self.enable_wasapi_exclusive: bool = False  # WASAPI独占模式开关
        self.input_device_name: str = ""  # 输入设备名称
        self.output_device_name: str = ""  # 输出设备名称
        self.target_samplerate: int = 48000  # 目标采样率（Hz）
        self.target_channels: int = 2  # 目标声道数（默认2声道，兼容大多数设备）


# -------------------------- 语音转换GUI主类 --------------------------
class VoiceConversionGUIMain:
    """
    语音转换GUI主类：整合界面渲染、设备管理、音频处理、模型推理全流程
    设计原则：职责分离（UI渲染/设备管理/音频处理拆分为独立方法）、可维护性（参数集中管理）
    """

    def __init__(self):
        # 初始化全局环境与工作目录
        self.work_dir = init_global_env()

        # 初始化配置对象
        self.gui_config = VoiceConversionGUIConfig()  # GUI配置（用户设置）
        self.app_config = Config()  # 应用核心配置（系统参数）

        # 初始化状态变量
        self.current_function = "vc"  # 当前功能模式（vc:语音转换；im:输入监听）
        self.processing_delay = 0  # 处理延迟（毫秒，用于UI显示）
        self.audio_stream = None  # 音频流对象（AudioIoProcess实例）
        self.rvc_model = None  # RVC模型实例（语音转换核心）

        # 初始化音频设备相关变量
        self.hostapi_list = None  # 音频主机API列表
        self.input_device_list = None  # 输入设备名称列表
        self.output_device_list = None  # 输出设备名称列表
        self.input_device_index_list = None  # 输入设备索引列表
        self.output_device_index_list = None  # 输出设备索引列表

        # 初始化共享内存与缓冲区（音频数据传输用）
        self.input_shared_mem = None  # 输入共享内存
        self.output_shared_mem = None  # 输出共享内存
        self.input_buffer = None  # 输入音频缓冲区（numpy数组）
        self.output_buffer = None  # 输出音频缓冲区（numpy数组）

        # 初始化指针与事件（线程/进程同步用）
        self.input_write_ptr = None  # 输入缓冲区写指针（共享内存指针）
        self.output_read_ptr = None  # 输出缓冲区读指针（共享内存指针）
        self.playback_ptr = None  # 播放位置指针（跟踪音频播放进度）
        self.input_event = None  # 输入数据就绪事件（通知音频处理线程）
        self.stop_event = None  # 停止事件（通知所有线程/进程退出）

        # 初始化音频处理相关缓冲区（torch张量，GPU加速用）
        self.init_audio_buffers()

        # 初始化Harvest基频提取进程池
        self.init_harvest_process_pool()

        # 更新音频设备列表（初始化时获取系统所有设备）
        self.update_audio_devices()

        # 启动GUI界面（渲染窗口并进入事件循环）
        self.launch_gui()

    def init_audio_buffers(self):
        """初始化音频处理所需的缓冲区（预定义空结构，避免运行时动态创建）"""
        # 核心音频缓冲区（后续根据采样率动态调整大小）
        self.input_audio_buf: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # 主输入缓冲区
        self.denoised_audio_buf: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # 降噪后输入缓冲区
        self.resampled_audio_buf: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # 16k重采样缓冲区（模型输入用）
        self.output_audio_buf: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # 输出缓冲区（降噪用）

        # 辅助缓冲区
        self.rms_detect_buf: np.ndarray = np.array(
            [], dtype=np.float32
        )  # RMS静音检测缓冲区
        self.sola_buffer: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # SOLA拼接缓冲区
        self.denoise_buffer: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # 降噪历史缓冲区

        # 窗口函数（后续根据SOLA缓冲区大小动态生成）
        self.fade_in_window: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # 淡入窗口
        self.fade_out_window: torch.Tensor = torch.tensor(
            [], dtype=torch.float32
        )  # 淡出窗口

        # 重采样器（后续根据采样率动态初始化）
        self.resampler_to_16k: tat.Resample = None  # 目标采样率→16k重采样器
        self.resampler_to_target: tat.Resample = None  # 16k→目标采样率重采样器

        # 降噪模型（后续根据配置动态初始化）
        self.torch_gate: TorchGate = None

        # 音频块参数（后续根据GUI配置动态计算）
        self.zc_step: int = 0  # 零交叉步长（采样率//100，用于帧对齐）
        self.block_frame: int = 0  # 单块音频帧数（基于block_duration计算）
        self.block_frame_16k: int = 0  # 16k采样率下的单块帧数
        self.crossfade_frame: int = 0  # 交叉fade帧数
        self.sola_buffer_frame: int = 0  # SOLA缓冲区帧数
        self.sola_search_frame: int = 0  # SOLA搜索范围帧数
        self.extra_frame: int = 0  # 额外推理帧数
        self.skip_head_frame: int = 0  # 模型推理时跳过的头部帧数
        self.return_length_frame: int = 0  # 模型推理期望输出帧数

    def init_harvest_process_pool(self):
        """初始化Harvest基频提取的多进程池（输入/输出队列+工作进程）"""
        self.harvest_input_queue = Queue()  # 任务输入队列（存储待处理音频数据）
        self.harvest_output_queue = Queue()  # 结果输出队列（通知处理完成）

        # 启动Harvest工作进程（数量由gui_config.harvest_cpu_count控制）
        for _ in range(self.gui_config.harvest_cpu_count):
            harvest_process = HarvestPitchProcess(
                input_queue=self.harvest_input_queue,
                output_queue=self.harvest_output_queue,
            )
            harvest_process.start()

    def check_necessary_assets(self):
        """检查并下载必要的资源文件（模型依赖、配置文件等）"""
        # 清理并重建临时目录
        temp_dir = os.path.join(self.work_dir, "TEMP")
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)

        # 检查资源完整性，缺失则下载
        if not check_all_assets(update=self.app_config.update):
            if self.app_config.update:
                print_with_format("[Asset Check] 缺失必要资源，开始下载...")
                download_all_assets(tmpdir=temp_dir)

                # 二次检查，确保下载成功
                if not check_all_assets(update=self.app_config.update):
                    print_with_format("[Asset Check] 资源下载失败，无法继续运行！")
                    sys.exit(1)
            else:
                print_with_format(
                    "[Asset Check] 缺失必要资源，请启用更新模式重新运行！"
                )
                sys.exit(1)
        print_with_format("[Asset Check] 所有必要资源已就绪")

    def load_saved_config(self) -> dict:
        """加载保存的配置文件（configs/inuse/config.json），无配置则生成默认值"""
        config_path = os.path.join(self.work_dir, "configs", "inuse", "config.json")
        default_config = self._get_default_config()

        # 确保配置目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        try:
            # 读取已保存的配置
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            # 验证配置中的音频设备是否有效（设备可能已变更）
            saved_config = self._validate_device_config(saved_config)

            # 补充配置中缺失的字段（兼容旧版本配置）
            for key, value in default_config.items():
                if key not in saved_config:
                    saved_config[key] = value

            return saved_config

        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            # 配置文件不存在或损坏，返回默认配置
            print_with_format(f"[Config Load] 配置文件读取失败：{str(e)}，使用默认配置")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config

    def _get_default_config(self) -> dict:
        """生成默认配置（基于当前系统设备与默认参数）"""
        # 获取默认音频设备
        default_input_dev_idx = sd.default.device[0]
        default_output_dev_idx = sd.default.device[1]
        default_input_dev_name = self.input_device_list[
            self.input_device_index_list.index(default_input_dev_idx)
        ]
        default_output_dev_name = self.output_device_list[
            self.output_device_index_list.index(default_output_dev_idx)
        ]

        # 默认配置字典
        return {
            "pth_path": "",
            "index_path": "",
            "sg_hostapi": self.hostapi_list[0] if self.hostapi_list else "",
            "sg_wasapi_exclusive": False,
            "sg_input_device": default_input_dev_name,
            "sg_output_device": default_output_dev_name,
            "sr_type": "sr_model",
            "threhold": -60,
            "pitch": 0,
            "formant": 0.0,
            "index_rate": 0.0,
            "rms_mix_rate": 0.0,
            "block_time": 0.25,
            "crossfade_length": 0.05,
            "extra_time": 2.5,
            "n_cpu": min(max_harvest_cpu, 4),
            "f0method": "rmvpe",
            "use_jit": False,
            "use_pv": False,
            "samplerate": 48000,
        }

    def _validate_device_config(self, config: dict) -> dict:
        """验证配置中的音频设备是否有效，无效则重置为默认设备"""
        # 验证主机API
        if config.get("sg_hostapi") not in self.hostapi_list:
            config["sg_hostapi"] = self.hostapi_list[0] if self.hostapi_list else ""
            print_with_format(
                f"[Config Validate] 主机API无效，重置为：{config['sg_hostapi']}"
            )

        # 只有当API确实发生变化时，才重新获取设备列表
        if config["sg_hostapi"] != self.gui_config.hostapi_name:
            self.update_audio_devices(hostapi_name=config["sg_hostapi"])

        # 验证输入设备
        if config.get("sg_input_device") not in self.input_device_list:
            default_input_idx = sd.default.device[0]
            config["sg_input_device"] = self.input_device_list[
                self.input_device_index_list.index(default_input_idx)
            ]
            print_with_format(
                f"[Config Validate] 输入设备无效，重置为：{config['sg_input_device']}"
            )

        # 验证输出设备
        if config.get("sg_output_device") not in self.output_device_list:
            default_output_idx = sd.default.device[1]
            config["sg_output_device"] = self.output_device_list[
                self.output_device_index_list.index(default_output_idx)
            ]
            print_with_format(
                f"[Config Validate] 输出设备无效，重置为：{config['sg_output_device']}"
            )

        return config

    def update_audio_devices(self, hostapi_name: str = None):
        """
        更新音频设备列表（基于指定的主机API）
        参数：
            hostapi_name: 主机API名称（如WASAPI），为None则使用当前配置的API
        """
        global flag_vc
        flag_vc = False  # 停止当前语音转换，避免设备冲突

        # 重启sounddevice，确保设备列表刷新
        sd._terminate()
        sd._initialize()

        # 获取所有主机API与设备
        all_hostapis = sd.query_hostapis()
        all_devices = sd.query_devices()

        # 为每个设备添加所属主机API名称（便于筛选）
        for api in all_hostapis:
            for dev_idx in api["devices"]:
                all_devices[dev_idx]["hostapi_name"] = api["name"]

        # 确定目标主机API（优先使用指定的API，无则用配置的API，再无则用第一个API）
        target_api = hostapi_name or self.gui_config.hostapi_name
        if not target_api or target_api not in [api["name"] for api in all_hostapis]:
            target_api = all_hostapis[0]["name"] if all_hostapis else ""

        # 更新主机API列表
        self.hostapi_list = [api["name"] for api in all_hostapis]

        # 筛选当前API下的输入/输出设备
        self.input_device_list = [
            dev["name"]
            for dev in all_devices
            if dev["max_input_channels"] > 0 and dev["hostapi_name"] == target_api
        ]
        self.output_device_list = [
            dev["name"]
            for dev in all_devices
            if dev["max_output_channels"] > 0 and dev["hostapi_name"] == target_api
        ]

        # 筛选设备对应的索引（用于sounddevice调用）
        self.input_device_index_list = [
            dev["index"]
            for dev in all_devices
            if dev["max_input_channels"] > 0 and dev["hostapi_name"] == target_api
        ]
        self.output_device_index_list = [
            dev["index"]
            for dev in all_devices
            if dev["max_output_channels"] > 0 and dev["hostapi_name"] == target_api
        ]

        # 更新GUI配置中的主机API
        self.gui_config.hostapi_name = target_api
        print_with_format(
            f"[Device Update] 已更新{target_api}下的设备列表：输入设备{len(self.input_device_list)}个，输出设备{len(self.output_device_list)}个"
        )

    def launch_gui(self):
        """启动GUI窗口：加载配置、构建布局、进入事件循环"""
        # 检查必要资源
        self.check_necessary_assets()

        # 加载保存的配置
        saved_config = self.load_saved_config()

        # 初始化GUI主题（LightBlue3，兼顾美观与可读性）
        sg.theme("LightBlue3")

        # 构建GUI布局（分区域组织，提升可读性）
        layout = self._build_gui_layout(saved_config)

        # 创建窗口（finalize=True确保窗口创建后可立即操作控件）
        self.window = sg.Window(
            title="RVC - Real-time Voice Conversion",
            layout=layout,
            finalize=True,
            resizable=True,  # 允许窗口调整大小
        )

        # 进入事件循环（处理用户交互）
        self.run_event_loop()

    def _build_gui_layout(self, saved_config: dict) -> list:
        """构建GUI布局（按功能模块拆分，便于维护）"""
        # 1. 模型加载区域
        model_load_frame = sg.Frame(
            title=i18n("Load Model"),
            layout=[
                [
                    sg.Input(
                        default_text=saved_config.get("pth_path", ""),
                        key="model_path",
                        expand_x=True,  # 输入框占满剩余宽度
                        tooltip=i18n("Path to the .pth model file"),
                    ),
                    sg.FileBrowse(
                        button_text=i18n("Select the .pth File"),
                        initial_folder=os.path.join(self.work_dir, "assets", "weights"),
                        file_types=[("Model Files", "*.pth")],
                        tooltip=i18n("Browse and select the model file"),
                    ),
                ],
                [
                    sg.Input(
                        default_text=saved_config.get("index_path", ""),
                        key="index_path",
                        expand_x=True,
                        tooltip=i18n("Path to the .index feature file"),
                    ),
                    sg.FileBrowse(
                        button_text=i18n("Select .index File"),
                        initial_folder=os.path.join(self.work_dir, "logs"),
                        file_types=[("Index Files", "*.index")],
                        tooltip=i18n("Browse and select the feature index file"),
                    ),
                ],
            ],
            expand_x=True,
        )

        # 2. 音频设备配置区域
        audio_device_frame = sg.Frame(
            title=i18n("Audio Device"),
            layout=[
                [
                    sg.Text(i18n("Host API")),
                    sg.Combo(
                        values=self.hostapi_list,
                        key="hostapi_name",
                        default_value=saved_config.get("sg_hostapi", ""),
                        enable_events=True,
                        size=(20, 1),
                        tooltip=i18n("Select the audio host API (e.g., WASAPI)"),
                    ),
                    sg.Checkbox(
                        text=i18n("WASAPI Exclusive Mode"),
                        key="wasapi_exclusive",
                        default=saved_config.get("sg_wasapi_exclusive", False),
                        enable_events=True,
                        tooltip=i18n("Enable WASAPI exclusive mode for lower latency"),
                    ),
                ],
                [
                    sg.Text(i18n("Input Device")),
                    sg.Combo(
                        values=self.input_device_list,
                        key="input_device",
                        default_value=saved_config.get("sg_input_device", ""),
                        enable_events=True,
                        size=(45, 1),
                        expand_x=True,
                        tooltip=i18n(
                            "Select the audio input device (e.g., microphone)"
                        ),
                    ),
                ],
                [
                    sg.Text(i18n("Output Device")),
                    sg.Combo(
                        values=self.output_device_list,
                        key="output_device",
                        default_value=saved_config.get("sg_output_device", ""),
                        enable_events=True,
                        size=(45, 1),
                        expand_x=True,
                        tooltip=i18n("Select the audio output device (e.g., speakers)"),
                    ),
                ],
                [
                    sg.Button(
                        button_text=i18n("Reload Device List"),
                        key="reload_devices",
                        tooltip=i18n("Refresh the list of available audio devices"),
                    ),
                    sg.Radio(
                        text=i18n("Use Model Sampling Rate"),
                        group_id="sr_mode",
                        key="sr_model",
                        default=saved_config.get("sr_type") == "sr_model",
                        enable_events=True,
                        tooltip=i18n("Use the sampling rate of the loaded model"),
                    ),
                    sg.Radio(
                        text=i18n("Use Device Sampling Rate"),
                        group_id="sr_mode",
                        key="sr_device",
                        default=saved_config.get("sr_type") == "sr_device",
                        enable_events=True,
                        tooltip=i18n("Use the sampling rate of the audio device"),
                    ),
                    sg.Text(i18n("Target Sampling Rate (Hz)")),
                    sg.Combo(
                        values=[44100, 48000, 32000, 16000],
                        key="target_samplerate",
                        default_value=saved_config.get("samplerate", 48000),
                        enable_events=True,
                        size=(8, 1),
                        tooltip=i18n(
                            "Select the target sampling rate for audio processing"
                        ),
                    ),
                    sg.Text(
                        key="current_sr_display",
                        tooltip=i18n("Currently active sampling rate"),
                    ),
                ],
            ],
            expand_x=True,
        )

        # 3. 通用设置区域
        general_settings_frame = sg.Frame(
            title=i18n("General Settings"),
            layout=[
                [
                    sg.Text(i18n("Silence Threshold (dB)")),
                    sg.Slider(
                        range=(-60, 0),
                        key="silence_threshold",
                        resolution=1,
                        orientation="h",
                        default_value=saved_config.get("threhold", -60),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n(
                            "Audio below this level is treated as silence (-60 to 0 dB)"
                        ),
                    ),
                ],
                [
                    sg.Text(i18n("Pitch Shift (semitones)")),
                    sg.Slider(
                        range=(-24, 24),
                        key="pitch_shift",
                        resolution=1,
                        orientation="h",
                        default_value=saved_config.get("pitch", 0),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n(
                            "Shift pitch by this number of semitones (-24 to 24)"
                        ),
                    ),
                ],
                [
                    sg.Text(i18n("Formant Shift")),
                    sg.Slider(
                        range=(-5, 5),
                        key="formant_shift",
                        resolution=0.01,
                        orientation="h",
                        default_value=saved_config.get("formant", 0.0),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n("Shift formants to change timbre (-5 to 5)"),
                    ),
                ],
                [
                    sg.Text(i18n("Feature Search Ratio")),
                    sg.Slider(
                        range=(0.0, 1.0),
                        key="index_ratio",
                        resolution=0.01,
                        orientation="h",
                        default_value=saved_config.get("index_rate", 0.0),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n("Ratio of feature index usage (0.0 to 1.0)"),
                    ),
                ],
                [
                    sg.Text(i18n("Loudness Matching")),
                    sg.Slider(
                        range=(0.0, 1.0),
                        key="rms_mix_ratio",
                        resolution=0.01,
                        orientation="h",
                        default_value=saved_config.get("rms_mix_rate", 0.0),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n(
                            "Match output loudness to input (0.0 = full match)"
                        ),
                    ),
                ],
                [
                    sg.Text(i18n("Pitch Detection Algorithm")),
                    sg.Combo(
                        values=["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"],
                        key="f0_method",
                        default_value=saved_config.get("f0method", "fcpe"),
                        enable_events=True,
                        size=(10, 1),
                        readonly=True,
                    ),
                ],
            ],
            size=(500, 300),
        )

        # 4. 性能设置区域
        performance_settings_frame = sg.Frame(
            title=i18n("Performance Settings"),
            layout=[
                [
                    sg.Text(i18n("Audio Block Duration (s)")),
                    sg.Slider(
                        range=(0.02, 1.5),
                        key="block_duration",
                        resolution=0.01,
                        orientation="h",
                        default_value=saved_config.get("block_time", 0.25),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n(
                            "Duration of each audio processing block (lower = more responsive)"
                        ),
                    ),
                ],
                [
                    sg.Text(i18n("Harvest CPU Processes")),
                    sg.Slider(
                        range=(1, max_harvest_cpu),
                        key="harvest_cpu_count",
                        resolution=1,
                        orientation="h",
                        default_value=saved_config.get(
                            "n_cpu", min(max_harvest_cpu, 4)
                        ),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n("Number of CPU cores for Harvest pitch detection"),
                    ),
                ],
                [
                    sg.Text(i18n("Crossfade Duration (s)")),
                    sg.Slider(
                        range=(0.01, 0.15),
                        key="crossfade_duration",
                        resolution=0.01,
                        orientation="h",
                        default_value=saved_config.get("crossfade_length", 0.05),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n("Duration of crossfade between audio blocks"),
                    ),
                ],
                [
                    sg.Text(i18n("Extra Inference Time (s)")),
                    sg.Slider(
                        range=(0.05, 5.00),
                        key="extra_infer_duration",
                        resolution=0.01,
                        orientation="h",
                        default_value=saved_config.get("extra_time", 2.5),
                        enable_events=True,
                        expand_x=True,
                        tooltip=i18n(
                            "Additional time for model inference to prevent clipping"
                        ),
                    ),
                ],
                [
                    sg.Checkbox(
                        i18n("Input Noise Reduction"),
                        key="enable_input_denoise",
                        default=False,
                        enable_events=True,
                        tooltip=i18n("Reduce noise from input audio"),
                    ),
                    sg.Checkbox(
                        i18n("Output Noise Reduction"),
                        key="enable_output_denoise",
                        default=False,
                        enable_events=True,
                        tooltip=i18n("Reduce noise from output audio"),
                    ),
                    sg.Checkbox(
                        i18n("Enable Phase Vocoder"),
                        key="enable_phase_vocoder",
                        default=saved_config.get("use_pv", False),
                        enable_events=True,
                        tooltip=i18n("Improve audio quality with phase alignment"),
                    ),
                ],
            ],
            size=(500, 300),
        )

        # 5. 控制按钮区域
        control_frame = [
            sg.Button(
                i18n("Start Conversion"),
                key="start_vc",
                size=(15, 1),
                button_color=("white", "green"),
                tooltip=i18n("Start real-time voice conversion"),
            ),
            sg.Button(
                i18n("Stop Conversion"),
                key="stop_vc",
                size=(15, 1),
                button_color=("white", "red"),
                tooltip=i18n("Stop real-time voice conversion"),
            ),
            sg.Radio(
                i18n("Monitor Input"),
                "function_mode",
                key="mode_input_monitor",
                default=False,
                enable_events=True,
                tooltip=i18n("Listen to original input audio"),
            ),
            sg.Radio(
                i18n("Convert Voice"),
                "function_mode",
                key="mode_voice_convert",
                default=True,
                enable_events=True,
                tooltip=i18n("Process and convert voice"),
            ),
            sg.Text(i18n("Processing Delay (ms):")),
            sg.Text("0", key="delay_time_display", size=(5, 1)),
            sg.Text(i18n("Inference Time (ms):")),
            sg.Text("0", key="infer_time_display", size=(5, 1)),
        ]

        # 组合所有区域为完整布局
        return [
            [model_load_frame],
            [audio_device_frame],
            [general_settings_frame, performance_settings_frame],
            control_frame,
        ]

    def run_event_loop(self):
        """运行GUI事件循环，处理用户交互"""
        global flag_vc
        while True:
            try:
                event, values = self.window.read()

                # 窗口关闭事件
                if event == sg.WINDOW_CLOSED:
                    self.stop_audio_stream()
                    break

                # 采样率选择事件
                elif event == "target_samplerate":
                    self.handle_samplerate_change(values)

                # 设备相关事件（刷新设备列表或切换主机API）
                elif event in ("reload_devices", "hostapi_name"):
                    self.handle_device_event(values)

                # 开始语音转换
                elif event == "start_vc" and not flag_vc:
                    self.handle_start_conversion(values)

                # 参数更新事件（滑块、复选框等）
                elif event in [
                    "silence_threshold",
                    "pitch_shift",
                    "formant_shift",
                    "index_ratio",
                    "rms_mix_ratio",
                    "f0_pm",
                    "f0_dio",
                    "f0_harvest",
                    "f0_crepe",
                    "f0_rmvpe",
                    "f0_fcpe",
                    "enable_input_denoise",
                    "enable_output_denoise",
                    "enable_phase_vocoder",
                    "mode_voice_convert",
                    "mode_input_monitor",
                ]:
                    self.handle_parameter_update(event, values)

                # 停止语音转换
                elif event == "stop_vc":
                    self.stop_audio_stream()

            except Exception as e:
                print_with_format(f"[GUI Error] Exception in event loop: {str(e)}")
                traceback.print_exc()

                # 发生错误时确保释放资源
                try:
                    self.stop_audio_stream()
                    # 清理其他可能的资源
                    if hasattr(self, "rvc_model"):
                        del self.rvc_model
                        torch.cuda.empty_cache()
                except Exception as cleanup_err:
                    print_with_format(f"[Cleanup Error] {str(cleanup_err)}")

    def handle_samplerate_change(self, values):
        """处理采样率变更事件"""
        try:
            self.gui_config.target_samplerate = int(values["target_samplerate"])
            print_with_format(
                f"[Samplerate] Changed to {self.gui_config.target_samplerate} Hz"
            )
        except Exception:
            print_with_format("[Samplerate] Invalid samplerate value")
            traceback.print_exc()

    def handle_device_event(self, values):
        """处理设备相关事件（刷新设备列表或切换主机API）"""
        try:
            # 更新主机API配置
            self.gui_config.hostapi_name = values["hostapi_name"]

            # 重新加载设备列表
            self.update_audio_devices(hostapi_name=values["hostapi_name"])

            # 更新GUI控件
            self.window["hostapi_name"].update(
                values=self.hostapi_list, value=self.gui_config.hostapi_name
            )

            # 重置输入设备（如果当前设备不在新列表中）
            if (
                self.gui_config.input_device_name not in self.input_device_list
                and self.input_device_list
            ):
                self.gui_config.input_device_name = self.input_device_list[0]
            self.window["input_device"].update(
                values=self.input_device_list, value=self.gui_config.input_device_name
            )

            # 重置输出设备（如果当前设备不在新列表中）
            if (
                self.gui_config.output_device_name not in self.output_device_list
                and self.output_device_list
            ):
                self.gui_config.output_device_name = self.output_device_list[0]
            self.window["output_device"].update(
                values=self.output_device_list, value=self.gui_config.output_device_name
            )

        except Exception:
            print_with_format("[Device Event] Error handling device event")
            traceback.print_exc()

    def handle_start_conversion(self, values):
        """处理开始语音转换事件"""
        try:
            # 验证配置并更新GUI配置对象
            if not self.validate_and_update_config(values):
                return

            # 保存当前配置
            self.save_current_config(values)

            # 启动语音转换
            self.start_voice_conversion()

            # 更新UI显示的采样率和延迟
            self.window["current_sr_display"].update(
                f"{self.gui_config.target_samplerate} Hz"
            )
            self.window["delay_time_display"].update(
                int(np.round(self.processing_delay * 1000))
            )

        except Exception:
            print_with_format("[Start Conversion] Error starting voice conversion")
            traceback.print_exc()

    def validate_and_update_config(self, values) -> bool:
        """验证配置有效性并更新GUI配置对象"""
        # 验证模型文件路径
        if not values["model_path"].strip():
            sg.popup(i18n("Please select a .pth model file"))
            return False

        # 验证索引文件路径
        if not values["index_path"].strip():
            sg.popup(i18n("Please select a .index file"))
            return False

        # 验证路径不含Unicode字符（避免后续处理错误）
        unicode_pattern = re.compile("[^\x00-\x7f]+")
        if unicode_pattern.findall(values["model_path"]):
            sg.popup(i18n("Model path cannot contain non-ASCII characters"))
            return False

        if unicode_pattern.findall(values["index_path"]):
            sg.popup(i18n("Index path cannot contain non-ASCII characters"))
            return False

        # 更新设备配置
        self.set_audio_devices(
            input_device_name=values["input_device"],
            output_device_name=values["output_device"],
        )

        # 更新核心配置
        self.app_config.use_jit = False  # 禁用JIT加速（暂不支持）

        # 更新GUI配置
        self.gui_config.model_path = values["model_path"]
        self.gui_config.index_path = values["index_path"]
        self.gui_config.sr_mode = "sr_model" if values["sr_model"] else "sr_device"
        self.gui_config.silence_threshold = values["silence_threshold"]
        self.gui_config.pitch_shift = values["pitch_shift"]
        self.gui_config.formant_shift = values["formant_shift"]
        self.gui_config.block_duration = values["block_duration"]
        self.gui_config.crossfade_duration = values["crossfade_duration"]
        self.gui_config.extra_infer_duration = values["extra_infer_duration"]
        self.gui_config.enable_input_denoise = values["enable_input_denoise"]
        self.gui_config.enable_output_denoise = values["enable_output_denoise"]
        self.gui_config.enable_phase_vocoder = values["enable_phase_vocoder"]
        self.gui_config.rms_mix_ratio = values["rms_mix_ratio"]
        self.gui_config.index_search_ratio = values["index_ratio"]
        self.gui_config.harvest_cpu_count = values["harvest_cpu_count"]
        self.gui_config.hostapi_name = values["hostapi_name"]
        self.gui_config.enable_wasapi_exclusive = values["wasapi_exclusive"]
        self.gui_config.input_device_name = values["input_device"]
        self.gui_config.output_device_name = values["output_device"]

        # 确定基频提取方法 - 使用下拉框选择的值
        if "f0_method" in values and values["f0_method"]:
            # 直接使用下拉框选择的值
            self.gui_config.f0_extract_method = values["f0_method"]

        return True

    def save_current_config(self, values):
        """保存当前配置到文件（configs/inuse/config.json）"""
        config_path = os.path.join(self.work_dir, "configs", "inuse", "config.json")

        # 构建配置字典
        config = {
            "pth_path": values["model_path"],
            "index_path": values["index_path"],
            "sg_hostapi": values["hostapi_name"],
            "sg_wasapi_exclusive": values["wasapi_exclusive"],
            "sg_input_device": values["input_device"],
            "sg_output_device": values["output_device"],
            "sr_type": "sr_model" if values["sr_model"] else "sr_device",
            "threhold": values["silence_threshold"],
            "pitch": values["pitch_shift"],
            "formant": values["formant_shift"],
            "index_rate": values["index_ratio"],
            "rms_mix_rate": values["rms_mix_ratio"],
            "block_time": values["block_duration"],
            "crossfade_length": values["crossfade_duration"],
            "extra_time": values["extra_infer_duration"],
            "n_cpu": values["harvest_cpu_count"],
            "use_jit": False,
            "use_pv": values["enable_phase_vocoder"],
            "f0method": self.gui_config.f0_extract_method,
            "samplerate": self.gui_config.target_samplerate,
        }

        # 保存配置
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print_with_format(f"[Config Save] Configuration saved to {config_path}")

    def set_audio_devices(self, input_device_name: str, output_device_name: str):
        """设置音频输入/输出设备，并验证采样率兼容性"""
        try:
            # 获取设备索引
            input_idx = self.input_device_index_list[
                self.input_device_list.index(input_device_name)
            ]
            output_idx = self.output_device_index_list[
                self.output_device_list.index(output_device_name)
            ]

            # 设置默认设备
            sd.default.device[0] = input_idx
            sd.default.device[1] = output_idx

            print_with_format(
                f"[Device Setup] Input: {input_device_name} (Index: {input_idx})"
            )
            print_with_format(
                f"[Device Setup] Output: {output_device_name} (Index: {output_idx})"
            )

            # 验证采样率兼容性
            self.verify_samplerate_compatibility()

        except Exception as e:
            print_with_format(f"[Device Setup] Error setting audio devices: {str(e)}")
            traceback.print_exc()
            sg.popup(i18n(f"Failed to set audio devices: {str(e)}"))

    def verify_samplerate_compatibility(self):
        """验证所选采样率是否被设备支持，不支持则回退到设备默认采样率"""
        target_sr = self.gui_config.target_samplerate
        input_dev_info = sd.query_devices(device=sd.default.device[0])

        try:
            # 检查输入设备是否支持目标采样率
            sd.check_input_settings(device=sd.default.device[0], samplerate=target_sr)
            self.gui_config.target_samplerate = target_sr
            print_with_format(f"[Samplerate] Device supports {target_sr} Hz")

        except Exception as e:
            # 回退到设备默认采样率
            fallback_sr = int(input_dev_info.get("default_samplerate", 44100))
            self.gui_config.target_samplerate = fallback_sr
            print_with_format(
                f"[Samplerate] Device does not support {target_sr} Hz. Fallback to {fallback_sr} Hz"
            )
            sg.popup(
                i18n(
                    f"Selected samplerate {target_sr} Hz not supported.\nFalling back to {fallback_sr} Hz"
                )
            )

    def start_voice_conversion(self):
        """启动语音转换：初始化模型、设置缓冲区、启动音频流"""
        # 清理GPU缓存
        torch.cuda.empty_cache()

        # 初始化RVC模型
        self.rvc_model = RTRVC(
            key=self.gui_config.pitch_shift,
            formant=self.gui_config.formant_shift,
            pth_path=self.gui_config.model_path,
            index_path=self.gui_config.index_path,
            index_rate=self.gui_config.index_search_ratio,
            n_cpu=self.gui_config.harvest_cpu_count,
            device=self.app_config.device,
            use_jit=self.app_config.use_jit,
            is_half=self.app_config.is_half,
            is_dml=self.app_config.dml,
        )

        # 确定目标采样率（模型采样率或设备采样率）
        if self.gui_config.sr_mode == "sr_model":
            self.gui_config.target_samplerate = self.rvc_model.tgt_sr
        else:
            self.gui_config.target_samplerate = self.get_device_samplerate()

        # 确定目标声道数（取输入输出设备的最小声道数，最大为2）
        self.gui_config.target_channels = self.get_device_channel_count()

        # 计算音频处理参数（基于采样率和配置的时长）
        self.calculate_audio_parameters()

        # 初始化音频缓冲区和窗口函数
        self.initialize_audio_buffers()

        # 启动音频流
        self.start_audio_stream()

    def calculate_audio_parameters(self):
        """计算音频处理所需的各种参数（帧数、步长等）"""
        self.zc_step = self.gui_config.target_samplerate // 100  # 零交叉步长（10ms）

        # 计算块大小（确保是zc_step的整数倍，保证帧对齐）
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_duration
                    * self.gui_config.target_samplerate
                    / self.zc_step
                )
            )
            * self.zc_step
        )

        # 16k采样率下的块大小（模型输入要求）
        self.block_frame_16k = 160 * self.block_frame // self.zc_step

        # 交叉fade帧数
        self.crossfade_frame = (
            int(
                np.round(
                    self.gui_config.crossfade_duration
                    * self.gui_config.target_samplerate
                    / self.zc_step
                )
            )
            * self.zc_step
        )

        # SOLA算法相关帧数
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc_step)
        self.sola_search_frame = self.zc_step

        # 额外推理帧数（避免尾部截断）
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_infer_duration
                    * self.gui_config.target_samplerate
                    / self.zc_step
                )
            )
            * self.zc_step
        )

        # 模型推理参数
        self.skip_head_frame = self.extra_frame // self.zc_step
        self.return_length_frame = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc_step

        # 计算处理延迟（用于UI显示）
        self.processing_delay = (
            self.gui_config.block_duration
            + self.gui_config.crossfade_duration
            + 0.01  # 基础处理时间
        )
        if self.gui_config.enable_input_denoise:
            self.processing_delay += min(self.gui_config.crossfade_duration, 0.04)

    def initialize_audio_buffers(self):
        """初始化音频处理所需的缓冲区和窗口函数"""
        # 主输入缓冲区（包含额外推理帧+交叉fade帧+SOLA搜索帧+块帧）
        buffer_size = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
        )
        self.input_audio_buf = torch.zeros(
            buffer_size, device=self.app_config.device, dtype=torch.float32
        )
        self.denoised_audio_buf = self.input_audio_buf.clone()

        # 16k重采样缓冲区（模型输入用）
        self.resampled_audio_buf = torch.zeros(
            160 * self.input_audio_buf.shape[0] // self.zc_step,
            device=self.app_config.device,
            dtype=torch.float32,
        )

        # 辅助缓冲区
        self.rms_detect_buf = np.zeros(4 * self.zc_step, dtype=np.float32)
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.app_config.device, dtype=torch.float32
        )
        self.denoise_buffer = self.sola_buffer.clone()
        self.output_audio_buf = self.input_audio_buf.clone()

        # 生成淡入淡出窗口（正弦平方窗口，平滑过渡）
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.app_config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        # 初始化重采样器
        self.resampler_to_16k = tat.Resample(
            orig_freq=self.gui_config.target_samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.app_config.device)

        # 如果模型采样率与目标采样率不同，初始化额外的重采样器
        if self.rvc_model.tgt_sr != self.gui_config.target_samplerate:
            self.resampler_to_target = tat.Resample(
                orig_freq=self.rvc_model.tgt_sr,
                new_freq=self.gui_config.target_samplerate,
                dtype=torch.float32,
            ).to(self.app_config.device)
        else:
            self.resampler_to_target = None

        # 初始化降噪模型
        self.torch_gate = TorchGate(
            sr=self.gui_config.target_samplerate,
            n_fft=4 * self.zc_step,
            prop_decrease=0.9,
        ).to(self.app_config.device)

    def start_audio_stream(self):
        """启动音频流和处理线程"""
        global flag_vc
        if not flag_vc:
            flag_vc = True

            # 检查是否启用WASAPI独占模式
            use_wasapi_exclusive = (
                "WASAPI" in self.gui_config.hostapi_name
                and self.gui_config.enable_wasapi_exclusive
            )

            # 初始化音频I/O进程
            self.audio_stream = AudioIoProcess(
                input_device=sd.default.device[0],
                output_device=sd.default.device[1],
                input_audio_block_size=self.block_frame,
                sample_rate=self.gui_config.target_samplerate,
                channel_num=self.gui_config.target_channels,
                is_input_wasapi_exclusive=use_wasapi_exclusive,
                is_output_wasapi_exclusive=use_wasapi_exclusive,
                is_device_combined=True,
            )

            # 连接共享内存
            self.input_shared_mem = shared_memory.SharedMemory(
                name=self.audio_stream.get_in_mem_name()
            )
            self.output_shared_mem = shared_memory.SharedMemory(
                name=self.audio_stream.get_out_mem_name()
            )

            # 映射缓冲区
            self.input_buffer = np.ndarray(
                self.audio_stream.get_np_shape(),
                dtype=self.audio_stream.get_np_dtype(),
                buffer=self.input_shared_mem.buf,
                order="C",
            )
            self.output_buffer = np.ndarray(
                self.audio_stream.get_np_shape(),
                dtype=self.audio_stream.get_np_dtype(),
                buffer=self.output_shared_mem.buf,
                order="C",
            )

            # 获取同步指针和事件
            (
                self.input_write_ptr,
                self.output_read_ptr,
                self.playback_ptr,
                self.input_event,
                self.stop_event,
            ) = self.audio_stream.get_ptrs_and_events()

            # 启动音频流
            self.audio_stream.start()

            # 启动音频处理线程
            import threading

            audio_thread = threading.Thread(
                target=self.audio_processing_loop, daemon=True
            )
            audio_thread.start()
            print_with_format("[Audio Stream] Started real-time processing")

    def stop_audio_stream(self):
        """停止音频流和处理线程"""
        global flag_vc
        if flag_vc:
            flag_vc = False
            if self.audio_stream is not None:
                print_with_format("[Audio Stream] Stopping...")
                # 发送停止信号
                self.stop_event.set()
                # 清理共享内存
                self.input_shared_mem.close()
                self.output_shared_mem.close()
                # 等待进程结束
                self.audio_stream.join()
                self.audio_stream = None
                print_with_format("[Audio Stream] Stopped")

    def audio_processing_loop(self):
        """音频实时处理主循环"""
        while flag_vc:
            self.process_audio_block(
                self.block_frame << 1
            )  # buf_size = 2 * block_frame

    def process_audio_block(self, buf_size: int):
        """
        处理单个音频块：从输入缓冲区读取数据，处理后写入输出缓冲区
        参数：
            buf_size: 缓冲区大小（用于计算读写位置）
        """
        try:
            # 等待新的音频数据
            self.input_event.wait()
            read_ptr = self.input_write_ptr.value
            self.input_event.clear()

            # 记录处理开始时间
            start_time = time.perf_counter()

            # 计算数据范围并读取输入音频
            end_ptr = read_ptr + self.block_frame
            input_data = np.copy(self.input_buffer[read_ptr:end_ptr])

            # 裁剪极端值，防止爆音
            input_data = np.clip(input_data, -1, 1)

            # 转换为单声道
            input_data = librosa.to_mono(input_data.T)

            # 静音检测与处理
            processed_data = self.apply_silence_detection(input_data)

            # 更新主输入缓冲区
            self.input_audio_buf[: -self.block_frame] = self.input_audio_buf[
                self.block_frame :
            ].clone()
            self.input_audio_buf[-processed_data.shape[0] :] = torch.from_numpy(
                processed_data
            ).to(self.app_config.device)

            # 处理输入（降噪和重采样）
            self.process_input_audio()

            # 语音转换或直通处理
            output_audio = self.process_voice_conversion()

            # 输出降噪处理
            if self.gui_config.enable_output_denoise and self.current_function == "vc":
                output_audio = self.apply_output_denoising(output_audio)

            # 音量匹配
            if self.gui_config.rms_mix_ratio < 1 and self.current_function == "vc":
                output_audio = self.match_volume(output_audio)

            # 音频块拼接（SOLA算法或相位声码器）
            output_audio = self.stitch_audio_blocks(output_audio)

            # 写入输出缓冲区
            self.write_to_output_buffer(output_audio, buf_size)

            # 更新推理时间显示
            total_time = time.perf_counter() - start_time
            if flag_vc:
                self.window["infer_time_display"].update(int(total_time * 1000))

        except Exception as e:
            print_with_format("[Audio Processing] Error processing audio block:")
            traceback.print_exc()

    def apply_silence_detection(self, audio_data: np.ndarray) -> np.ndarray:
        """应用基于RMS的静音检测，将静音部分归零"""
        if self.gui_config.silence_threshold > -60:
            # 拼接历史数据以保持检测连续性
            audio_data = np.append(self.rms_detect_buf, audio_data)

            # 计算RMS能量
            rms = librosa.feature.rms(
                y=audio_data, frame_length=4 * self.zc_step, hop_length=self.zc_step
            )[
                :, 2:
            ]  # 跳过前两帧避免边缘效应

            # 更新RMS缓冲区
            self.rms_detect_buf[:] = audio_data[-4 * self.zc_step :]

            # 裁剪数据以对齐帧
            audio_data = audio_data[2 * self.zc_step - self.zc_step // 2 :]

            # 检测静音帧并归零
            silence_mask = (
                librosa.amplitude_to_db(rms, ref=1.0)[0]
                < self.gui_config.silence_threshold
            )
            for i in range(silence_mask.shape[0]):
                if silence_mask[i]:
                    start = i * self.zc_step
                    end = start + self.zc_step
                    audio_data[start:end] = 0

            # 最终裁剪以完成帧对齐
            audio_data = audio_data[self.zc_step // 2 :]

        return audio_data

    def process_input_audio(self):
        """处理输入音频：降噪和重采样"""
        if self.gui_config.enable_input_denoise:
            # 更新降噪缓冲区
            self.denoised_audio_buf[: -self.block_frame] = self.denoised_audio_buf[
                self.block_frame :
            ].clone()

            # 提取需要降噪的音频片段
            audio_segment = self.input_audio_buf[
                -self.sola_buffer_frame - self.block_frame :
            ]

            # 应用降噪
            denoised_segment = self.torch_gate(
                audio_segment.unsqueeze(0), self.input_audio_buf.unsqueeze(0)
            ).squeeze(0)

            # 应用淡入窗口，与历史数据平滑过渡
            denoised_segment[: self.sola_buffer_frame] *= self.fade_in_window
            denoised_segment[: self.sola_buffer_frame] += (
                self.denoise_buffer * self.fade_out_window
            )

            # 更新缓冲区
            self.denoised_audio_buf[-self.block_frame :] = denoised_segment[
                : self.block_frame
            ]
            self.denoise_buffer[:] = denoised_segment[self.block_frame :]

            # 重采样到16k（模型输入要求）
            self.resampled_audio_buf[-self.block_frame_16k - 160 :] = (
                self.resampler_to_16k(
                    self.denoised_audio_buf[-self.block_frame - 2 * self.zc_step :]
                )[160:]
            )
        else:
            # 无降噪时直接重采样
            self.resampled_audio_buf[
                -160 * (self.block_frame // self.zc_step + 1) :
            ] = self.resampler_to_16k(
                self.input_audio_buf[-self.block_frame - 2 * self.zc_step :]
            )[
                160:
            ]

    def process_voice_conversion(self) -> torch.Tensor:
        """执行语音转换或直接返回输入（根据当前模式）"""
        if self.current_function == "vc":
            # 执行语音转换
            converted_audio = self.rvc_model.infer(
                self.resampled_audio_buf,
                self.block_frame_16k,
                self.skip_head_frame,
                self.return_length_frame,
                self.gui_config.f0_extract_method,
            )

            # 必要时重采样到目标采样率
            if self.resampler_to_target is not None:
                converted_audio = self.resampler_to_target(converted_audio)
            return converted_audio
        elif self.gui_config.enable_input_denoise:
            # 仅降噪模式
            return self.denoised_audio_buf[self.extra_frame :].clone()
        else:
            # 直通模式
            return self.input_audio_buf[self.extra_frame :].clone()

    def apply_output_denoising(self, audio: torch.Tensor) -> torch.Tensor:
        """对输出音频应用降噪处理"""
        self.output_audio_buf[: -self.block_frame] = self.output_audio_buf[
            self.block_frame :
        ].clone()
        self.output_audio_buf[-self.block_frame :] = audio[-self.block_frame :]

        return self.torch_gate(
            audio.unsqueeze(0), self.output_audio_buf.unsqueeze(0)
        ).squeeze(0)

    def match_volume(self, output_audio: torch.Tensor) -> torch.Tensor:
        """将输出音频的音量与输入音频匹配"""
        # 选择参考音频（降噪后或原始输入）
        if self.gui_config.enable_input_denoise:
            reference_audio = self.denoised_audio_buf[self.extra_frame :]
        else:
            reference_audio = self.input_audio_buf[self.extra_frame :]

        # 确保参考音频长度与输出一致
        reference_audio = reference_audio[: output_audio.shape[0]]

        # 计算参考音频的RMS
        ref_rms = librosa.feature.rms(
            y=reference_audio.cpu().numpy(),
            frame_length=4 * self.zc_step,
            hop_length=self.zc_step,
        )
        ref_rms = torch.from_numpy(ref_rms).to(self.app_config.device)
        ref_rms = F.interpolate(
            ref_rms.unsqueeze(0),
            size=output_audio.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]

        # 计算输出音频的RMS
        out_rms = librosa.feature.rms(
            y=output_audio.cpu().numpy(),
            frame_length=4 * self.zc_step,
            hop_length=self.zc_step,
        )
        out_rms = torch.from_numpy(out_rms).to(self.app_config.device)
        out_rms = F.interpolate(
            out_rms.unsqueeze(0),
            size=output_audio.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        out_rms = torch.max(
            out_rms, torch.tensor(1e-3, device=self.app_config.device)
        )  # 避免除零

        # 应用音量匹配
        return output_audio * torch.pow(
            ref_rms / out_rms, torch.tensor(1 - self.gui_config.rms_mix_ratio)
        )

    def stitch_audio_blocks(self, audio_block: torch.Tensor) -> torch.Tensor:
        """使用SOLA算法或相位声码器拼接音频块，确保平滑过渡"""
        # 计算与历史缓冲区的互相关，找到最佳对齐位置
        conv_input = audio_block[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.app_config.device),
            )
            + 1e-8
        )

        # 确定最佳偏移量（不同平台兼容处理）
        if sys.platform == "darwin":  # macOS
            _, offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            offset = offset.item()
        else:  # Windows/Linux
            offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

        # 根据偏移量裁剪当前块
        audio_block = audio_block[offset:]

        # 应用淡入淡出或相位声码器
        if (
            "privateuseone" in str(self.app_config.device)
            or not self.gui_config.enable_phase_vocoder
        ):
            # 普通淡入淡出
            audio_block[: self.sola_buffer_frame] *= self.fade_in_window
            audio_block[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
        else:
            # 相位声码器（更高质量）
            audio_block[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                audio_block[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )

        # 更新SOLA缓冲区
        self.sola_buffer[:] = audio_block[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]

        return audio_block

    def write_to_output_buffer(self, audio_data: torch.Tensor, buf_size: int):
        """将处理后的音频写入输出缓冲区，处理循环缓冲区逻辑"""
        # 准备输出数据（转换为目标声道数）
        output_data = (
            audio_data[: self.block_frame]
            .repeat(self.gui_config.target_channels, 1)
            .t()
            .cpu()
            .numpy()
        )

        # 计算写入位置（处理循环缓冲区）
        write_start = self.output_read_ptr.value
        play_pos = self.playback_ptr.value

        # 计算读写位置差距（考虑缓冲区循环）
        position_diff = (write_start - play_pos + buf_size) % buf_size

        if position_diff < self.block_frame:
            # 缓冲区不足，从播放位置开始写（可能导致短暂爆音）
            print_with_format("[Buffer Warning] Output underrun detected")
            write_pos = play_pos
        else:
            # 正常写入位置
            write_pos = (write_start + self.block_frame) % buf_size

        # 写入输出缓冲区
        write_end = (write_pos + self.block_frame) % buf_size
        if write_end > write_pos:
            self.output_buffer[write_pos:write_end] = output_data
        else:
            # 处理缓冲区边界
            first_part_len = buf_size - write_pos
            self.output_buffer[write_pos:] = output_data[:first_part_len]
            self.output_buffer[:write_end] = output_data[first_part_len:]

        # 更新写指针
        self.output_read_ptr.value = write_pos

        # 检查输入缓冲区溢出
        if self.input_event.is_set():
            print_with_format("[Buffer Warning] Input overrun detected")
            self.input_event.clear()

    def handle_parameter_update(self, event: str, values: dict):
        """处理参数更新事件（实时调整处理参数）"""
        try:
            if event == "silence_threshold":
                self.gui_config.silence_threshold = values["silence_threshold"]

            elif event == "pitch_shift":
                self.gui_config.pitch_shift = values["pitch_shift"]
                if hasattr(self, "rvc_model"):
                    self.rvc_model.set_key(values["pitch_shift"])

            elif event == "formant_shift":
                self.gui_config.formant_shift = values["formant_shift"]
                if hasattr(self, "rvc_model"):
                    self.rvc_model.set_formant(values["formant_shift"])

            elif event == "index_ratio":
                self.gui_config.index_search_ratio = values["index_ratio"]
                if hasattr(self, "rvc_model"):
                    self.rvc_model.set_index_rate(values["index_ratio"])

            elif event == "rms_mix_ratio":
                self.gui_config.rms_mix_ratio = values["rms_mix_ratio"]

            elif event == "f0_method":
                # 直接使用下拉框选择的值
                self.gui_config.f0_extract_method = values["f0_method"]

            elif event == "enable_input_denoise":
                self.gui_config.enable_input_denoise = values["enable_input_denoise"]
                # 调整延迟计算
                if self.audio_stream is not None:
                    delay_adjust = min(self.gui_config.crossfade_duration, 0.04)
                    self.processing_delay += (
                        delay_adjust
                        if values["enable_input_denoise"]
                        else -delay_adjust
                    )
                    self.window["delay_time_display"].update(
                        int(np.round(self.processing_delay * 1000))
                    )

            elif event == "enable_output_denoise":
                self.gui_config.enable_output_denoise = values["enable_output_denoise"]

            elif event == "enable_phase_vocoder":
                self.gui_config.enable_phase_vocoder = values["enable_phase_vocoder"]

            elif event in ["mode_voice_convert", "mode_input_monitor"]:
                self.current_function = "vc" if values["mode_voice_convert"] else "im"

        except Exception:
            print_with_format(f"[Parameter Update] Error updating parameter: {event}")
            traceback.print_exc()

    def get_device_samplerate(self) -> int:
        """获取设备的采样率（用户选择或默认）"""
        return self.gui_config.target_samplerate

    def get_device_channel_count(self) -> int:
        """获取设备支持的声道数（取输入输出设备的最小值，最大为2）"""
        input_channels = sd.query_devices(device=sd.default.device[0])[
            "max_input_channels"
        ]
        output_channels = sd.query_devices(device=sd.default.device[1])[
            "max_output_channels"
        ]
        return min(input_channels, output_channels, 2)


# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    # 启动语音转换GUI
    gui_app = VoiceConversionGUIMain()
