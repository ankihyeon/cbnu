#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 실시간 STT (ETRI) — Auto Sensitivity + AI Noise Suppression (통합본)
- 전송 모드: 고정 간격 / 침묵 감지(기본)
- 침묵 1.1초 초과 시 세그먼트 종료 → STT 호출 → 문장 단위 줄바꿈 출력
- 입력 감도 자동 보정(ON) / 재보정 버튼
- 소음 억제: off / RNNoise / Spectral Gate (미설치면 off만 노출 + 툴팁 안내)
- High-DPI 대응, 레이아웃/스타일 정리
- 안정성: 버퍼 락, RNNoise 480프레임 처리, 로그 append, 종료 대기 강화
- 성능: Spectral 단축(짧은 버퍼 통과, 완만 감쇠), 프리롤 300ms로 첫 음절 보존
"""

import os, io, time, json, base64, traceback, threading, re
from dataclasses import dataclass
from collections import deque
from typing import Any, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from PyQt5 import QtCore, QtWidgets

# ── 선택 라이브러리 (있으면 자동사용)
try:
    import webrtcvad
    USE_VAD = True
except Exception:
    USE_VAD = False

try:
    from rnnoise import RNNoise
    HAVE_RNNOISE = True
except Exception:
    HAVE_RNNOISE = False
    RNNoise = None

try:
    import noisereduce as nr
    HAVE_NR = True
except Exception:
    HAVE_NR = False

# ===================== 환경 =====================
load_dotenv()
ETRI_ACCESS_KEY = os.getenv("ETRI_ACCESS_KEY", "").strip()
ETRI_URL = "http://epretx.etri.re.kr:8000/api/WiseASR_Recognition"  # 서비스가 https를 지원하면 교체 권장
TARGET_SR = 16000

# ===================== 유틸 =====================
def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2 and x.shape[1] > 1:
        return x.mean(axis=1, keepdims=True).astype(np.float32, copy=False)
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)

def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    n = x.shape[0]
    new_n = int(round(n * dst_sr / max(1, src_sr)))
    if new_n <= 1 or n <= 1:
        return np.zeros((0, 1), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, n, endpoint=False)
    xq = np.linspace(0.0, 1.0, new_n, endpoint=False)
    y = np.interp(xq, xp, x[:, 0]).astype(np.float32)
    return y.reshape(-1, 1)

def wav_bytes_from_float(x16k_mono: np.ndarray, sr: int = TARGET_SR) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, x16k_mono, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()

def call_etri_asr(wav_bytes: bytes, language_code: str = "korean") -> dict:
    """간단 재시도 1회 포함"""
    out = {"ok": False, "text": "", "http": None, "elapsed_ms": None, "error": None}
    last_err = None
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            payload = {"argument": {"language_code": language_code, "audio": wav_b64}}
            headers = {
                "Content-Type": "application/json; charset=UTF-8",
                "Authorization": ETRI_ACCESS_KEY,
            }
            r = requests.post(ETRI_URL, headers=headers, data=json.dumps(payload), timeout=30)
            out["http"] = r.status_code
            if r.status_code == 200:
                data = r.json()
                obj = data.get("return_object") or {}
                text = obj.get("recognized") or obj.get("result") or obj.get("text") or ""
                if isinstance(text, list):
                    text = " ".join(t for t in text if isinstance(t, str))
                out["text"] = (text or "").strip()
                out["ok"] = bool(out["text"])
            else:
                out["error"] = (r.text or "")[:300]
            out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
            return out
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
            if attempt == 0:
                time.sleep(0.3)
                continue
    out["error"] = last_err
    return out

# 문장 분리(한/영 혼용)
_SENT_SPLIT_RE = re.compile(r'([.?!。？！]|…)+\s*')
def split_sentences_kor(text: str):
    parts, start = [], 0
    for m in _SENT_SPLIT_RE.finditer(text):
        end = m.end()
        seg = text[start:end].strip()
        if seg:
            parts.append(seg)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if len(p.replace(" ", "")) >= 2]

# ===================== 메시지 =====================
@dataclass
class WorkerMsg:
    kind: str   # "log" | "level" | "text" | "stat" | "error"
    data: Any

# ===================== 워커 =====================
class AudioWorker(QtCore.QThread):
    msg = QtCore.pyqtSignal(object)

    def __init__(
        self,
        device_index: int,
        language: str,
        mode: str = "interval",          # "interval" | "silence"
        chunk_sec: float = 3.0,
        vad_aggr: int = 2,
        sil_thr: float = 0.06,
        sil_min_sec: float = 1.20,
        auto_sensitivity: bool = True,
        noise_suppression: str = "off",  # "off" | "rnnoise" | "spectral"
    ):
        super().__init__()
        self.device_index = device_index
        self.language = language
        self.mode = mode
        self.chunk_sec = max(1.0, float(chunk_sec))
        self.vad_aggr = int(np.clip(vad_aggr, 0, 3))
        self.sil_thr = float(sil_thr)
        self.sil_min_sec = float(sil_min_sec)
        self.auto_sensitivity = bool(auto_sensitivity)
        self.ns_mode = noise_suppression

        # 내부 상태
        self._stop = False
        self._buf = np.zeros((0,1), np.float32)
        self._buf_lock = threading.Lock()
        self._device_sr = None
        self._device_ch = None

        # 인터벌 모드
        self._chunk_len = int(TARGET_SR * self.chunk_sec)
        self._max_buf = TARGET_SR * 12

        # 자동 감도/세그 규칙
        self._start_rms_gate = 0.04   # 발화 시작 최소 RMS (화장실 환경에서 약간 상향)
        self._min_seg_sec = 1.0       # 전송 최소 길이
        self._min_seg_rms = 0.02
        self._min_voice_ratio = 0.30  # VAD 유성 비율

        # VAD
        self._use_vad = USE_VAD if self.mode == "silence" else False
        if self._use_vad:
            self.vad = webrtcvad.Vad(self.vad_aggr)
            self._frame_20 = int(TARGET_SR * 0.02)
            self._speech_active = False
            self._nspeech = 0
            self._nsil = 0
            self._need_consec_speech = 3
            self._need_consec_sil = max(1, int(self.sil_min_sec / 0.02))
            self._seg_start_idx = 0
            self._speech_samples = 0
            self._max_speech_sec = 7.0
            self._max_speech_samples = int(TARGET_SR * self._max_speech_sec)
            self._vad_total_frames = 0
            self._vad_voiced_frames = 0
            self._sil_elapsed = 0.0

            # 프리롤(첫 음절 보존)
            self._preroll = int(TARGET_SR * 0.30)     # 300ms
            self._ring = np.zeros((0,1), np.float32)  # 최근 샘플 링버퍼

        # 소음 억제
        self._rnnoise = RNNoise() if (self.ns_mode == "rnnoise" and HAVE_RNNOISE) else None
        self._spectral_profile = None

        self._seg_queue = deque()
        self._seg_lock = threading.Lock()

    def stop(self):
        self._stop = True

    def _float_to_pcm16(self, x: np.ndarray) -> bytes:
        x = np.clip(x[:,0], -1.0, 1.0)
        return (x * 32767.0).astype(np.int16).tobytes()

    def _segment_ok(self, seg: np.ndarray, voiced_frames: int, total_frames: int) -> bool:
        dur = seg.shape[0] / TARGET_SR
        if dur < self._min_seg_sec:
            return False
        rms = float(np.sqrt(np.mean(seg**2)))
        if rms < self._min_seg_rms:
            return False
        if self._use_vad and total_frames > 0:
            ratio = voiced_frames / total_frames
            if ratio < self._min_voice_ratio:
                return False
        return True

    def _push_segment(self, seg: np.ndarray, reason: str):
        if seg.shape[0] < int(TARGET_SR * 0.35):
            return
        with self._seg_lock:
            self._seg_queue.append(seg)
        self.msg.emit(WorkerMsg("log", f"[SEG] queued ({len(seg)/TARGET_SR:.2f}s) reason={reason}"))

    def _denoise_block(self, x: np.ndarray, sr: int) -> np.ndarray:
        # RNNoise: 480프레임 고정, 테일은 그대로 둠
        if self.ns_mode == "rnnoise" and self._rnnoise is not None:
            frame = 480
            n = (len(x) // frame) * frame
            if n <= 0:
                return x
            y = np.copy(x)
            for i in range(0, n, frame):
                y[i:i+frame, 0] = self._rnnoise.filter(x[i:i+frame, 0])
            return y

        # Spectral: 짧은 버퍼는 통과, 프로필 학습 후 stationary 감쇠 완만
        if self.ns_mode == "spectral" and HAVE_NR:
            if len(x) < int(sr * 0.25):
                return x
            try:
                if self._spectral_profile is None and len(x) >= int(sr * 0.5):
                    noise_ref = x[:int(sr*0.5),0]
                    self._spectral_profile = noise_ref.copy()
                y = nr.reduce_noise(
                    y=x[:,0], sr=sr,
                    y_noise=self._spectral_profile if self._spectral_profile is not None else None,
                    stationary=True if self._spectral_profile is not None else False,
                    prop_decrease=0.8
                )
                return y.reshape(-1,1).astype(np.float32)
            except Exception:
                return x
        return x

    def _auto_calibrate(self):
        self.msg.emit(WorkerMsg("log", "[AUTO] calibrating input sensitivity... (1.2s)"))
        base = np.zeros((0,1), np.float32)
        t_end = time.perf_counter() + 1.2
        while time.perf_counter() < t_end:
            with self._buf_lock:
                if self._buf.shape[0] > 0:
                    grab = self._buf.copy()
                    self._buf = np.zeros((0,1), np.float32)
                else:
                    grab = None
            if grab is not None:
                base = np.concatenate((base, grab), axis=0)
            self.msleep(30)
        if len(base) < int(TARGET_SR * 0.2):
            self.msg.emit(WorkerMsg("log", "[AUTO] insufficient data; keep defaults"))
            return
        rms = float(np.sqrt(np.mean(base**2)))
        std = float(np.std(base))
        self.sil_thr = max(0.01, min(0.15, rms + 2.5*std))
        self._start_rms_gate = max(0.02, min(0.12, rms + 3.0*std))
        self.msg.emit(WorkerMsg("log", f"[AUTO] sil_thr={self.sil_thr:.3f}, start_gate={self._start_rms_gate:.3f}"))

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            if status:
                self.msg.emit(WorkerMsg("log", f"[AUDIO] status: {status}"))

            x_in = indata.astype(np.float32, copy=False)

            # 레벨 바 표시용 RMS
            mono_for_rms = to_mono(x_in)
            rms = float(np.sqrt(np.mean(mono_for_rms**2)))
            self.msg.emit(WorkerMsg("level", min(max(rms,0.0),1.0)))

            # 모노/리샘플
            x = to_mono(x_in)
            x = resample_linear(x, self._device_sr, TARGET_SR)

            # 소음 억제
            if self.ns_mode != "off":
                x = self._denoise_block(x, TARGET_SR)

            # 메인 버퍼에 누적
            with self._buf_lock:
                self._buf = np.concatenate((self._buf, x), axis=0)

            # 침묵 모드에서만 VAD 분절
            if self.mode != "silence" or not self._use_vad:
                return

            # 프리롤 링버퍼 갱신
            self._ring = np.concatenate((self._ring, x), axis=0)
            if self._ring.shape[0] > self._preroll:
                self._ring = self._ring[-self._preroll:]

            # ── VAD 기반 분절 ──
            while True:
                with self._buf_lock:
                    available = self._buf.shape[0] - self._seg_start_idx
                    if available < int(TARGET_SR*0.02):
                        break
                    frame = self._buf[self._seg_start_idx:self._seg_start_idx + int(TARGET_SR*0.02)]
                    self._seg_start_idx += int(TARGET_SR*0.02)

                r = float(np.sqrt(np.mean(frame**2)))
                pcm16 = self._float_to_pcm16(frame)
                is_speech = self.vad.is_speech(pcm16, TARGET_SR)

                if is_speech and r >= self._start_rms_gate:
                    self._nspeech += 1; self._nsil = 0
                    self._speech_samples += int(TARGET_SR*0.02)
                    self._vad_total_frames += 1
                    self._vad_voiced_frames += 1
                    if (not self._speech_active) and (self._nspeech >= 3):
                        self._speech_active = True
                        self._speech_samples = int(TARGET_SR*0.02)
                        self._vad_total_frames = 1
                        self._vad_voiced_frames = 1
                        # 프리롤을 본 버퍼 앞에 부착
                        with self._buf_lock:
                            self._buf = np.concatenate((self._ring, self._buf), axis=0)
                            self._seg_start_idx += self._ring.shape[0]
                        self.msg.emit(WorkerMsg("log", "[VAD] speech START (with preroll)"))
                else:
                    self._nsil += 1; self._nspeech = 0
                    if self._speech_active:
                        self._vad_total_frames += 1

                # 종료 조건 A: 무음 지속
                if self._speech_active and self._nsil >= self._need_consec_sil:
                    with self._buf_lock:
                        seg = self._buf.copy()
                        self._buf = np.zeros((0,1), np.float32)
                        self._seg_start_idx = 0
                    voiced = self._vad_voiced_frames; total = self._vad_total_frames
                    # reset
                    self._nsil = self._nspeech = 0
                    self._speech_active = False
                    self._speech_samples = 0
                    self._vad_total_frames = 0
                    self._vad_voiced_frames = 0

                    if self._segment_ok(seg, voiced, total):
                        self._push_segment(seg, "silence")
                    else:
                        self.msg.emit(WorkerMsg("log", "[SEG] drop (short/quiet/low-voice)"))

                # 종료 조건 B: 워치독
                elif self._speech_active and self._speech_samples >= self._max_speech_samples:
                    with self._buf_lock:
                        seg = self._buf.copy()
                        self._buf = np.zeros((0,1), np.float32)
                        self._seg_start_idx = 0
                    voiced = self._vad_voiced_frames; total = self._vad_total_frames
                    self._nsil = self._nspeech = 0
                    self._speech_active = False
                    self._speech_samples = 0
                    self._vad_total_frames = 0
                    self._vad_voiced_frames = 0

                    if self._segment_ok(seg, voiced, total):
                        self.msg.emit(WorkerMsg("log", f"[VAD] force CUT ({self._max_speech_sec:.1f}s)"))
                        self._push_segment(seg, "watchdog")
                    else:
                        self.msg.emit(WorkerMsg("log", "[SEG] drop (watchdog invalid)"))

        except Exception:
            self.msg.emit(WorkerMsg("error", traceback.format_exc()))

    def run(self):
        try:
            d = sd.query_devices(self.device_index)
            self._device_sr = int(d.get("default_samplerate", 48000) or 48000)
            self._device_ch = min(int(d.get("max_input_channels", 1) or 1), 2)
            self.msg.emit(WorkerMsg("log",
                f"[INFO] device idx={self.device_index}, name={d['name']}, sr={self._device_sr}, ch={self._device_ch}"))
            self.msg.emit(WorkerMsg("log",
                f"[INFO] mode={self.mode} | chunk={self.chunk_sec:.2f}s | "
                f"sil_thr={self.sil_thr:.3f} sil_min={self.sil_min_sec:.2f}s | "
                f"VAD={'ON' if self._use_vad else 'OFF'}(aggr={self.vad_aggr if self._use_vad else '-'}) | "
                f"NoiseSupp={self.ns_mode} "
                f"(RNNoise={'Y' if HAVE_RNNOISE else 'N'}, Spectral={'Y' if HAVE_NR else 'N'})"))

            with sd.InputStream(
                device=self.device_index,
                samplerate=self._device_sr,
                channels=self._device_ch,
                dtype="float32",
                blocksize=0,
                callback=self._audio_callback,
            ):
                self.msg.emit(WorkerMsg("log", "[INFO] audio stream started"))
                if self.auto_sensitivity:
                    self._auto_calibrate()

                sent = 0
                MEANINGLESS = {"에","음","어","아","어어","네","예"}

                while not self._stop:
                    # 침묵 모드: 큐 처리
                    if self.mode == "silence":
                        seg = None
                        with self._seg_lock:
                            if self._seg_queue:
                                seg = self._seg_queue.popleft()
                        if seg is not None:
                            wavb = wav_bytes_from_float(seg, TARGET_SR)
                            self.msg.emit(WorkerMsg("log", f"[SEND-SILENCE] #{sent+1} {len(seg)} (~{len(seg)/TARGET_SR:.2f}s)"))
                            result = call_etri_asr(wavb, self.language)
                            self.msg.emit(WorkerMsg("stat", result))
                            if result["ok"]:
                                txt = result["text"].strip()
                                if len(txt) <= 2 and txt in MEANINGLESS:
                                    self.msg.emit(WorkerMsg("log", f"[POST] drop meaningless: {txt!r}"))
                                else:
                                    for sent_text in split_sentences_kor(txt):
                                        self.msg.emit(WorkerMsg("text", sent_text))
                                    self.msg.emit(WorkerMsg("log",
                                        f"[RECV] #{sent+1} OK {result['elapsed_ms']}ms (HTTP {result['http']}) -> {txt[:80]}"))
                            else:
                                self.msg.emit(WorkerMsg("log",
                                    f"[RECV] #{sent+1} FAIL {result['elapsed_ms']}ms (HTTP {result['http']}) err={result['error']}"))
                            sent += 1

                    # 인터벌 모드
                    if self.mode == "interval":
                        with self._buf_lock:
                            enough = self._buf.shape[0] >= self._chunk_len
                        if enough:
                            with self._buf_lock:
                                to_send = self._buf[:self._chunk_len].copy()
                                self._buf = self._buf[self._chunk_len:]
                                if self._buf.shape[0] > self._max_buf:
                                    self._buf = self._buf[-self._max_buf:]
                            wavb = wav_bytes_from_float(to_send, TARGET_SR)
                            self.msg.emit(WorkerMsg("log", f"[SEND] #{sent+1} {len(to_send)} (~{len(to_send)/TARGET_SR:.2f}s)"))
                            result = call_etri_asr(wavb, self.language)
                            self.msg.emit(WorkerMsg("stat", result))
                            if result["ok"]:
                                txt = result["text"].strip()
                                if len(txt) <= 2 and txt in MEANINGLESS:
                                    self.msg.emit(WorkerMsg("log", f"[POST] drop meaningless: {txt!r}"))
                                else:
                                    for sent_text in split_sentences_kor(txt):
                                        self.msg.emit(WorkerMsg("text", sent_text))
                                    self.msg.emit(WorkerMsg("log",
                                        f"[RECV] #{sent+1} OK {result['elapsed_ms']}ms (HTTP {result['http']}) -> {txt[:80]}"))
                            else:
                                self.msg.emit(WorkerMsg("log",
                                    f"[RECV] #{sent+1} FAIL {result['elapsed_ms']}ms (HTTP {result['http']}) err={result['error']}"))
                            sent += 1

                    self.msleep(40)

        except Exception:
            self.msg.emit(WorkerMsg("error", traceback.format_exc()))
        finally:
            self.msg.emit(WorkerMsg("log", "[INFO] audio stream stopped"))

# ===================== GUI =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙️ ETRI 실시간 STT (PyQt5 — Auto Sensitivity + AI Noise Suppression)")
        self.resize(1100, 800)

        cw = QtWidgets.QWidget(self); self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(10)

        # 스타일(그룹 테두리/여백 정돈)
        self.setStyleSheet("""
        QGroupBox {
            font-weight: 600;
            border: 1px solid #b8b8b8;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QLabel { color: #333; }
        """)

        # 상단 상태 라벨
        self.lbl_state = QtWidgets.QLabel("상태: 대기")
        root.addWidget(self.lbl_state)

        # ───────── 상단: 공통 컨트롤 ─────────
        top_row = QtWidgets.QGridLayout()
        top_row.setContentsMargins(0,0,0,0)
        top_row.setHorizontalSpacing(10)
        top_row.setVerticalSpacing(6)

        self.cmb_lang = QtWidgets.QComboBox(); self.cmb_lang.addItems(["korean","english"])

        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["고정 간격", "침묵 감지"])
        self.cmb_mode.setCurrentText("침묵 감지")
        self.cmb_mode.setToolTip("전송 모드 선택: 고정 간격 / 침묵 감지")

        self.chk_auto = QtWidgets.QCheckBox("입력 감도 자동 감지 (권장)")
        self.chk_auto.setChecked(True)
        self.btn_recalib = QtWidgets.QPushButton("재보정")

        self.cmb_ns = QtWidgets.QComboBox()
        ns_opts = ["off"]
        missing = []
        if HAVE_RNNOISE: ns_opts.append("rnnoise")
        else: missing.append("rnnoise")
        if HAVE_NR: ns_opts.append("spectral")
        else: missing.append("noisereduce")
        self.cmb_ns.addItems(ns_opts)
        if missing:
            self.cmb_ns.setToolTip("미설치: " + ", ".join(missing) + "\n설치 예: pip install rnnoise noisereduce scipy")

        self.cmb_device = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("장치 새로고침")

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_clear = QtWidgets.QPushButton("초기화(자막 비움)")
        for b in (self.btn_start, self.btn_stop, self.btn_clear):
            b.setMinimumWidth(140)

        r = 0
        top_row.addWidget(QtWidgets.QLabel("언어"), r,0);       top_row.addWidget(self.cmb_lang, r,1)
        top_row.addWidget(QtWidgets.QLabel("전송 모드"), r,2);  top_row.addWidget(self.cmb_mode, r,3)
        top_row.addWidget(QtWidgets.QLabel("소음 제거"), r,4);  top_row.addWidget(self.cmb_ns, r,5)

        r += 1
        top_row.addWidget(self.chk_auto, r,0,1,2)
        top_row.addWidget(self.btn_recalib, r,2)
        top_row.addWidget(QtWidgets.QLabel("오디오 입력 장치"), r,3); top_row.addWidget(self.cmb_device, r,4)
        top_row.addWidget(self.btn_refresh, r,5)

        r += 1
        top_row.addWidget(self.btn_start, r,0,1,2)
        top_row.addWidget(self.btn_stop,  r,2,1,2)
        top_row.addWidget(self.btn_clear, r,4,1,2)

        # 컬럼 스트레치로 좁아짐 방지
        top_row.setColumnStretch(1, 1)
        top_row.setColumnStretch(3, 1)
        top_row.setColumnStretch(5, 2)

        root.addLayout(top_row)

        # ───────── 중앙: 모드별 그룹 (QSplitter 사용) ─────────
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # [그룹 A] 고정 간격
        self.grp_interval = QtWidgets.QGroupBox("⏱️ 고정 간격 모드 설정 (interval)")
        ly_i = QtWidgets.QFormLayout(self.grp_interval)
        ly_i.setLabelAlignment(QtCore.Qt.AlignRight)
        ly_i.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        ly_i.setHorizontalSpacing(12); ly_i.setVerticalSpacing(6)
        ly_i.setContentsMargins(10,8,10,10)

        self.spn_chunk = QtWidgets.QDoubleSpinBox()
        self.spn_chunk.setRange(1.0, 10.0); self.spn_chunk.setValue(3.0)
        self.spn_chunk.setSuffix(" s")
        self.spn_chunk.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        ly_i.addRow("고정 간격(초)", self.spn_chunk)

        # [그룹 B] 침묵 감지
        self.grp_silence = QtWidgets.QGroupBox("🤫 침묵 감지 모드 설정 (silence)")
        ly_s = QtWidgets.QFormLayout(self.grp_silence)
        ly_s.setLabelAlignment(QtCore.Qt.AlignRight)
        ly_s.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        ly_s.setHorizontalSpacing(12); ly_s.setVerticalSpacing(6)
        ly_s.setContentsMargins(10,8,10,10)

        self.spn_vad = QtWidgets.QSpinBox(); self.spn_vad.setRange(0,3); self.spn_vad.setValue(3)
        self.spn_thr = QtWidgets.QDoubleSpinBox(); self.spn_thr.setRange(0.005,0.2); self.spn_thr.setSingleStep(0.005); self.spn_thr.setValue(0.06); self.spn_thr.setDecimals(3)
        self.spn_sil = QtWidgets.QDoubleSpinBox(); self.spn_sil.setRange(0.2,3.0); self.spn_sil.setSingleStep(0.1); self.spn_sil.setValue(1.1); self.spn_sil.setSuffix(" s")

        for w in (self.spn_vad, self.spn_thr, self.spn_sil):
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        ly_s.addRow("VAD Aggressiveness", self.spn_vad)
        ly_s.addRow("침묵 임계(RMS)",      self.spn_thr)
        ly_s.addRow("침묵 지속(초)",       self.spn_sil)

        split.addWidget(self.grp_interval)
        split.addWidget(self.grp_silence)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        root.addWidget(split)

        # ───────── 하단: 레벨/자막/로그 ─────────
        self.prg_level = QtWidgets.QProgressBar(); self.prg_level.setRange(0,100); self.prg_level.setFormat("입력 레벨: %p%")
        root.addWidget(self.prg_level)

        root.addWidget(QtWidgets.QLabel("자막 (세그먼트마다 자동 줄바꿈)"))
        self.txt_caption = QtWidgets.QTextEdit(); self.txt_caption.setReadOnly(True)
        self.txt_caption.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.txt_caption.setMinimumHeight(220)
        root.addWidget(self.txt_caption, stretch=1)

        root.addWidget(QtWidgets.QLabel("실시간 로그"))
        self.txt_log = QtWidgets.QTextEdit(); self.txt_log.setReadOnly(True)
        self.txt_log.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.txt_log.setMinimumHeight(180)
        root.addWidget(self.txt_log, stretch=1)

        # 상태 & 이벤트
        self.worker: Optional[AudioWorker] = None
        self._captions = deque(maxlen=300)

        # 이벤트 연결
        self.btn_refresh.clicked.connect(self.refresh_devices)
        self.btn_start.clicked.connect(self.start_record)
        self.btn_stop.clicked.connect(self.stop_record)
        self.btn_clear.clicked.connect(self.clear_captions)
        self.btn_recalib.clicked.connect(self.recalibrate_request)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_change)

        # (선택) 실시간 파라미터 반영
        for w in (self.spn_vad, self.spn_sil, self.spn_thr):
            try:
                w.valueChanged.connect(self._wire_live_params)
            except Exception:
                pass

        self.refresh_devices()
        self._on_mode_change(self.cmb_mode.currentText())

    def _wire_live_params(self):
        """실험용: 실행 중 파라미터 값 즉시 워커에 반영"""
        if not self.worker:
            return
        self.worker.vad_aggr = int(self.spn_vad.value())
        if getattr(self.worker, "_use_vad", False):
            try:
                self.worker.vad.set_mode(self.worker.vad_aggr)
            except Exception:
                pass
            self.worker.sil_min_sec = float(self.spn_sil.value())
            self.worker._need_consec_sil = max(1, int(self.worker.sil_min_sec / 0.02))
        self.worker.sil_thr = float(self.spn_thr.value())

    def _on_mode_change(self, mode_text: str):
        use_sil = (mode_text == "침묵 감지")
        self.grp_interval.setVisible(not use_sil)
        self.grp_silence.setVisible(use_sil)

    def refresh_devices(self):
        self.cmb_device.clear()
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0:
                    name = d["name"]; sr = int(d.get("default_samplerate", 0) or 0)
                    self.cmb_device.addItem(f"[{i}] {name} (sr={sr})", i)
        except Exception as e:
            self.log(f"[ERR] 장치 조회 실패: {e}")

    def start_record(self):
        if self.worker:
            return
        if not ETRI_ACCESS_KEY:
            QtWidgets.QMessageBox.warning(self, "키 없음", ".env 의 ETRI_ACCESS_KEY 를 설정하세요.")
            return
        if self.cmb_device.count() == 0:
            self.log("[ERR] 입력 장치가 없습니다."); return

        device_index = self.cmb_device.currentData()
        language = self.cmb_lang.currentText()
        mode_text = self.cmb_mode.currentText()
        mode = "silence" if mode_text == "침묵 감지" else "interval"
        chunk_sec = float(self.spn_chunk.value())
        vad_aggr = int(self.spn_vad.value())
        sil_thr = float(self.spn_thr.value())
        sil_min = float(self.spn_sil.value())
        auto = self.chk_auto.isChecked()
        ns = self.cmb_ns.currentText()

        self.worker = AudioWorker(
            device_index=device_index,
            language=language,
            mode=mode,
            chunk_sec=chunk_sec,
            vad_aggr=vad_aggr,
            sil_thr=sil_thr,
            sil_min_sec=sil_min,
            auto_sensitivity=auto,
            noise_suppression=ns,
        )
        self.worker.msg.connect(self.on_worker_msg)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_state.setText("상태: 대기")
        self.log("[INFO] Start 요청")

    def stop_record(self):
        if self.worker:
            self.worker.stop()
            if not self.worker.wait(3000):
                self.log("[WARN] 워커 종료 지연")
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_state.setText("상태: 대기")
        self.log("[INFO] Stop 요청")

    def clear_captions(self):
        self._captions.clear()
        self.txt_caption.clear()

    def recalibrate_request(self):
        if self.worker:
            self.log("[INFO] 재보정 실행")
            with self.worker._buf_lock:
                self.worker._buf = np.zeros((0,1), np.float32)
            self.worker._auto_calibrate()

    @QtCore.pyqtSlot(object)
    def on_worker_msg(self, m: WorkerMsg):
        if m.kind == "level":
            self.prg_level.setValue(int(float(m.data) * 100))
        elif m.kind == "text":
            self._captions.append(m.data)
            self.txt_caption.setPlainText("\n".join(self._captions))
            self.txt_caption.moveCursor(self.txt_caption.textCursor().End)
        elif m.kind == "stat":
            s = m.data
            self.log(f"[STAT] HTTP {s['http']}, {s['elapsed_ms']}ms" + ("" if not s["error"] else f", error={s['error']}"))
        elif m.kind == "log":
            s = str(m.data)
            if "[VAD] speech START" in s:
                self.lbl_state.setText("상태: 듣는 중")
            elif s.startswith("[SEND"):
                self.lbl_state.setText("상태: 전송 중")
            elif "audio stream started" in s:
                self.lbl_state.setText("상태: 대기")
            self.log(s)
        elif m.kind == "error":
            self.log("[EXC]\n" + str(m.data))

    def log(self, msg: str):
        self.txt_log.append(msg)

    def closeEvent(self, e):
        try:
            self.stop_record()
        finally:
            e.accept()

# ===================== main =====================
def main():
    # High-DPI (윈도우/4K에서 뭉개짐 방지)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")  # 통일감 있는 스타일
    w = MainWindow(); w.show()
    app.exec_()

if __name__ == "__main__":
    main()