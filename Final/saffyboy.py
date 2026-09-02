#!/usr/bin/env python3
"""
SaffyBoy — Game Boy / Game Boy Color Emulator
Improved accuracy, cleaner code, simpler comments.
"""

import sys
import os
import time
import json
import hashlib
import traceback
from pathlib import Path
from datetime import datetime

import pygame
import numpy as np

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "saffyboy_config.json"
STATES_DIR = BASE_DIR / "states"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SAVES_DIR = BASE_DIR / "saves"

for d in (STATES_DIR, SCREENSHOTS_DIR, SAVES_DIR):
    d.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "keys": {
        "right": "RIGHT", "left": "LEFT", "up": "UP", "down": "DOWN",
        "a": "X", "b": "Z", "select": "SPACE", "start": "RETURN",
        "turbo": "TAB", "pause": "P", "frame_advance": "N",
        "screenshot": "F12", "quick_save": "F5", "quick_load": "F8",
        "fullscreen": "F11", "toggle_overlay": "O", "reset": "R",
        "show_rom_info": "I",
    },
    "turbo_speed": 4.0,
    "unlimited_turbo": False,
    "volume": 0.7,
    "integer_scale": True,
    "show_overlay": True,
    "dmg_palette": 0,
    "custom_palette": [[224, 248, 208], [136, 192, 112], [52, 104, 86], [8, 24, 32]],
    "cgb_color_correction": True,
    "autosave_interval": 5.0,
    "window_width": 480,
    "window_height": 432,
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception as e:
            print(f"[config] Failed to load, using defaults: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[config] Failed to save: {e}")


CONFIG = load_config()

# Map key names to pygame constants
KEY_MAP = {}
for name in [
    "RIGHT", "LEFT", "UP", "DOWN", "X", "Z", "SPACE", "RETURN",
    "TAB", "P", "N", "F12", "F5", "F8", "F11", "O", "R", "I",
    "F1", "F2", "F3", "F4", "F6", "F7", "F9", "F10",
]:
    val = getattr(pygame, f"K_{name}", None)
    if val is None and len(name) == 1:
        val = getattr(pygame, f"K_{name.lower()}", None)
    if val is not None:
        KEY_MAP[name.upper()] = val
        KEY_MAP[name.lower()] = val


def get_key(name):
    val = CONFIG["keys"].get(name, DEFAULT_CONFIG["keys"].get(name))
    if val is None:
        return None
    s = str(val)
    return KEY_MAP.get(s.upper()) or KEY_MAP.get(s.lower())


# ---------------------------------------------------------------------------
# APU - lightweight but cleaner
# ---------------------------------------------------------------------------
class APU:
    SAMPLE_RATE = 22050

    def __init__(self):
        try:
            pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=2)
            self.ch1_channel = pygame.mixer.Channel(0)
            self.ch2_channel = pygame.mixer.Channel(1)
            self.ch3_channel = pygame.mixer.Channel(2)
            self.ch4_channel = pygame.mixer.Channel(3)
        except Exception:
            self.ch1_channel = self.ch2_channel = self.ch3_channel = self.ch4_channel = None

        self.ch1_last_freq = self.ch2_last_freq = self.ch3_last_freq = 0
        self.ch1_last_time = self.ch2_last_time = self.ch3_last_time = self.ch4_last_time = 0
        self.ch1_end_time = self.ch2_end_time = self.ch3_end_time = self.ch4_end_time = 0
        self._lfsr15 = self._build_lfsr(False, 8192)
        self._lfsr7 = self._build_lfsr(True, 127)
        self.volume = CONFIG.get("volume", 0.7)

    @staticmethod
    def _build_lfsr(step7, length):
        # Precompute noise LFSR output
        lfsr = 0x7FFF
        bits = np.empty(length, dtype=np.int8)
        for i in range(length):
            bit = (lfsr ^ (lfsr >> 1)) & 1
            lfsr = (lfsr >> 1) | (bit << 14)
            if step7:
                lfsr = (lfsr & ~0x40) | (bit << 6)
            bits[i] = 1 if (lfsr & 1) else -1
        return bits

    def stop_all(self):
        for ch in (self.ch1_channel, self.ch2_channel, self.ch3_channel, self.ch4_channel):
            if ch:
                try:
                    ch.stop()
                except Exception:
                    pass
        self.ch1_end_time = self.ch2_end_time = self.ch3_end_time = self.ch4_end_time = 0

    def _compute_duration(self, length_enabled, length_load, max_len, vol_init=None,
                          vol_dir=None, vol_period=None, default=0.5, cap=3.0):
        # Length counter or volume envelope decides how long the sound lasts
        duration = (max_len - length_load) / 256.0 if length_enabled else default
        if vol_init is not None and vol_dir == 0 and vol_period and vol_period > 0:
            decay = vol_init * (vol_period / 64.0)
            duration = min(duration, decay + 0.03) if length_enabled else (decay + 0.03)
        return max(0.02, min(duration, cap))

    def _envelope(self, n, vol_init, vol_dir, vol_period):
        if vol_period > 0:
            t = np.arange(n) / self.SAMPLE_RATE
            steps = np.floor(t / (vol_period / 64.0))
            vol = vol_init + steps * (1 if vol_dir else -1)
            return np.clip(vol, 0, 15)
        return np.full(n, vol_init, dtype=float)

    def _apply_panning(self, mono_i16, mmu, channel_num):
        nr50 = mmu.read_byte(0xFF24)
        nr51 = mmu.read_byte(0xFF25)
        left_on = bool(nr51 & (1 << (channel_num - 1 + 4)))
        right_on = bool(nr51 & (1 << (channel_num - 1)))
        left_vol = ((nr50 >> 4) & 0x07) / 7.0
        right_vol = (nr50 & 0x07) / 7.0
        n = len(mono_i16)
        stereo = np.zeros((n, 2), dtype=np.float32)
        if left_on:
            stereo[:, 0] = mono_i16.astype(np.float32) * left_vol * self.volume
        if right_on:
            stereo[:, 1] = mono_i16.astype(np.float32) * right_vol * self.volume
        return np.clip(stereo, -32767, 32767).astype(np.int16)

    def _fade_edges(self, wave, fade_ms=4):
        # Soft fade to avoid clicks
        n = len(wave)
        fade_n = min(int(self.SAMPLE_RATE * fade_ms / 1000.0), n // 2)
        if fade_n <= 0:
            return wave
        ramp = np.linspace(0.0, 1.0, fade_n)
        wave = wave.astype(np.float32)
        wave[:fade_n] *= ramp
        wave[-fade_n:] *= ramp[::-1]
        return wave.astype(np.int16)

    def _square_wave(self, raw_freq_start, duty, vol_init, vol_dir, vol_period,
                     sweep_period, sweep_dir, sweep_shift, duration):
        n = int(self.SAMPLE_RATE * duration)
        if n <= 0:
            return None
        t = np.arange(n)
        if sweep_period > 0 and sweep_shift > 0:
            # Frequency sweep (channel 1 only)
            sweep_steps_f = (t / self.SAMPLE_RATE) / (sweep_period / 128.0)
            max_step = int(sweep_steps_f[-1]) + 2
            raw = raw_freq_start
            freqs_list = []
            for _ in range(max_step):
                f = 131072.0 / (2048 - raw) if 0 < raw < 2048 else 0.0
                freqs_list.append(f)
                delta = raw >> sweep_shift
                raw = raw - delta if sweep_dir else raw + delta
                if raw < 0 or raw > 2047:
                    raw = max(0, min(raw, 2047))
            freqs_arr = np.array(freqs_list) if freqs_list else np.array([0.0])
            idx = np.clip(sweep_steps_f.astype(int), 0, len(freqs_arr) - 1)
            freq_t = freqs_arr[idx]
        else:
            freq_t = np.full(n, 131072.0 / (2048 - raw_freq_start) if raw_freq_start < 2048 else 0.0)
        freq_t = np.clip(freq_t, 0, 8000)
        phase = np.cumsum(freq_t) / self.SAMPLE_RATE
        frac = phase - np.floor(phase)
        square = np.where(frac < duty, 1.0, -1.0)
        vol = self._envelope(n, vol_init, vol_dir, vol_period)
        amplitude = (vol / 15.0) * 0.07
        wave = (square * amplitude * 32767).astype(np.int16)
        return self._fade_edges(wave)

    def _wave_channel(self, freq, wave_ram, vol_shift, duration):
        if freq <= 0:
            return None
        n = int(self.SAMPLE_RATE * duration)
        if n <= 0:
            return None
        samples4 = []
        for b in wave_ram:
            samples4.append((b >> 4) & 0x0F)
            samples4.append(b & 0x0F)
        samples4 = np.array(samples4, dtype=float) - 8.0
        t = np.arange(n)
        pos = (t * freq / self.SAMPLE_RATE) % 1.0
        idx = (pos * 32).astype(int) % 32
        vals = samples4[idx]
        scale = [0.0, 1.0, 0.5, 0.25][vol_shift] if vol_shift in (0, 1, 2, 3) else 1.0
        wave = (vals / 8.0 * scale * 0.08 * 32767).astype(np.int16)
        return self._fade_edges(wave)

    def _noise_channel(self, nr43, vol_init, vol_dir, vol_period, duration):
        shift = (nr43 >> 4) & 0x0F
        step7 = bool(nr43 & 0x08)
        div = nr43 & 0x07
        divisor = [8, 16, 32, 48, 64, 80, 96, 112][div]
        freq = 524288.0 / (divisor * (1 << shift)) if (divisor * (1 << shift)) > 0 else 1000.0
        if freq <= 0:
            freq = 1000.0
        n = int(self.SAMPLE_RATE * duration)
        if n <= 0:
            return None
        period_samples = max(1.0, self.SAMPLE_RATE / freq)
        lut = self._lfsr7 if step7 else self._lfsr15
        idx = (np.arange(n) / period_samples).astype(int) % len(lut)
        square = lut[idx].astype(float)
        vol = self._envelope(n, vol_init, vol_dir, vol_period)
        amplitude = (vol / 15.0) * 0.055
        wave = (square * amplitude * 32767).astype(np.int16)
        return self._fade_edges(wave)

    def trigger_channel1(self, mmu):
        if not self.ch1_channel or not (mmu.read_byte(0xFF26) & 0x80):
            return
        low = mmu.read_byte(0xFF13)
        high = mmu.read_byte(0xFF14) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq >= 2048:
            return
        nr10 = mmu.read_byte(0xFF10)
        sweep_period = (nr10 >> 4) & 0x07
        sweep_dir = (nr10 >> 3) & 0x01
        sweep_shift = nr10 & 0x07
        nr11 = mmu.read_byte(0xFF11)
        duty = [0.125, 0.25, 0.5, 0.75][(nr11 >> 6) & 0x03]
        length_load = nr11 & 0x3F
        nr12 = mmu.read_byte(0xFF12)
        vol_init = (nr12 >> 4) & 0x0F
        vol_dir = (nr12 >> 3) & 0x01
        vol_period = nr12 & 0x07
        if vol_init == 0 and vol_dir == 0:
            return
        length_enabled = bool(mmu.read_byte(0xFF14) & 0x40)
        freq = 131072.0 / (2048 - raw_freq)
        now = pygame.time.get_ticks()
        if abs(freq - self.ch1_last_freq) < 3 and (now - self.ch1_last_time) < 30:
            return
        self.ch1_last_freq, self.ch1_last_time = freq, now
        duration = self._compute_duration(length_enabled, length_load, 64, vol_init, vol_dir, vol_period)
        self.ch1_end_time = now + duration * 1000
        mono = self._square_wave(raw_freq, duty, vol_init, vol_dir, vol_period,
                                 sweep_period, sweep_dir, sweep_shift, duration)
        if mono is None:
            return
        try:
            self.ch1_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 1).tobytes()))
        except Exception:
            pass

    def trigger_channel2(self, mmu):
        if not self.ch2_channel or not (mmu.read_byte(0xFF26) & 0x80):
            return
        low = mmu.read_byte(0xFF18)
        high = mmu.read_byte(0xFF19) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq >= 2048:
            return
        nr21 = mmu.read_byte(0xFF16)
        duty = [0.125, 0.25, 0.5, 0.75][(nr21 >> 6) & 0x03]
        length_load = nr21 & 0x3F
        nr22 = mmu.read_byte(0xFF17)
        vol_init = (nr22 >> 4) & 0x0F
        vol_dir = (nr22 >> 3) & 0x01
        vol_period = nr22 & 0x07
        if vol_init == 0 and vol_dir == 0:
            return
        length_enabled = bool(mmu.read_byte(0xFF19) & 0x40)
        freq = 131072.0 / (2048 - raw_freq)
        now = pygame.time.get_ticks()
        if abs(freq - self.ch2_last_freq) < 3 and (now - self.ch2_last_time) < 30:
            return
        self.ch2_last_freq, self.ch2_last_time = freq, now
        duration = self._compute_duration(length_enabled, length_load, 64, vol_init, vol_dir, vol_period)
        self.ch2_end_time = now + duration * 1000
        mono = self._square_wave(raw_freq, duty, vol_init, vol_dir, vol_period, 0, 0, 0, duration)
        if mono is None:
            return
        try:
            self.ch2_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 2).tobytes()))
        except Exception:
            pass

    def trigger_channel3(self, mmu):
        if not self.ch3_channel or not (mmu.read_byte(0xFF26) & 0x80) or not (mmu.read_byte(0xFF1A) & 0x80):
            return
        low = mmu.read_byte(0xFF1D)
        high = mmu.read_byte(0xFF1E) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq >= 2048:
            return
        length_load = mmu.read_byte(0xFF1B)
        vol_reg = (mmu.read_byte(0xFF1C) >> 5) & 0x03
        length_enabled = bool(mmu.read_byte(0xFF1E) & 0x40)
        freq = 65536.0 / (2048 - raw_freq)
        now = pygame.time.get_ticks()
        if abs(freq - self.ch3_last_freq) < 3 and (now - self.ch3_last_time) < 30:
            return
        self.ch3_last_freq, self.ch3_last_time = freq, now
        duration = self._compute_duration(length_enabled, length_load, 256, default=0.5, cap=3.0)
        self.ch3_end_time = now + duration * 1000
        wave_ram = bytearray(mmu.read_byte(0xFF30 + i) for i in range(16))
        mono = self._wave_channel(freq, wave_ram, vol_reg, duration)
        if mono is None:
            return
        try:
            self.ch3_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 3).tobytes()))
        except Exception:
            pass

    def trigger_channel4(self, mmu):
        if not self.ch4_channel or not (mmu.read_byte(0xFF26) & 0x80):
            return
        nr41 = mmu.read_byte(0xFF20)
        length_load = nr41 & 0x3F
        nr42 = mmu.read_byte(0xFF21)
        vol_init = (nr42 >> 4) & 0x0F
        vol_dir = (nr42 >> 3) & 0x01
        vol_period = nr42 & 0x07
        if vol_init == 0 and vol_dir == 0:
            return
        nr43 = mmu.read_byte(0xFF22)
        length_enabled = bool(mmu.read_byte(0xFF23) & 0x40)
        now = pygame.time.get_ticks()
        if (now - self.ch4_last_time) < 30:
            return
        self.ch4_last_time = now
        duration = self._compute_duration(length_enabled, length_load, 64, vol_init, vol_dir, vol_period, default=0.3)
        self.ch4_end_time = now + duration * 1000
        mono = self._noise_channel(nr43, vol_init, vol_dir, vol_period, duration)
        if mono is None:
            return
        try:
            self.ch4_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 4).tobytes()))
        except Exception:
            pass

    def get_state(self):
        return {
            "ch1_end_time": self.ch1_end_time, "ch2_end_time": self.ch2_end_time,
            "ch3_end_time": self.ch3_end_time, "ch4_end_time": self.ch4_end_time,
            "volume": self.volume,
        }

    def set_state(self, state):
        self.ch1_end_time = state.get("ch1_end_time", 0)
        self.ch2_end_time = state.get("ch2_end_time", 0)
        self.ch3_end_time = state.get("ch3_end_time", 0)
        self.ch4_end_time = state.get("ch4_end_time", 0)
        self.volume = state.get("volume", CONFIG.get("volume", 0.7))
        self.stop_all()


# ---------------------------------------------------------------------------
# MMU - memory and cartridge
# ---------------------------------------------------------------------------
class MMU:
    def __init__(self):
        self.rom = bytearray()
        self.cart_type = 0
        self.rom_banks = 2
        self.ram_size = 0
        self.rom_bank = 1
        self.ram_bank = 0
        self.vram = bytearray(0x4000)          # 16 KB (2 banks)
        self.vram_bank = 0
        self.eram = bytearray()
        self.wram_banks = [bytearray(0x1000) for _ in range(8)]
        self.wram_bank = 1
        self.oam = bytearray(0xA0)
        self.hram = bytearray(0x7F)
        self.ie = 0
        self.io = bytearray(0x80)
        self.ppu = None
        self.apu = None
        self.timer = None
        self.joy_buttons = 0x0F
        self.joy_directions = 0x0F

        self.ram_enabled = False
        self.mbc1_mode = 0
        self.mbc1_rom_bank_low = 1
        self.mbc1_rom_bank_high = 0
        self.mbc5_rom_bank_low = 1
        self.mbc5_rom_bank_high = 0

        self.rtc_regs = [0] * 5
        self.rtc_latch = [0] * 5
        self._rtc_latch_pending = False
        self._rtc_cycle_acc = 0

        self.cgb_mode = False
        # CRAM: 8 BG palettes + 8 OBJ palettes
        self.bg_palette_ram = bytearray(64)
        self.obj_palette_ram = bytearray(64)
        for i in range(0, 64, 2):
            self.bg_palette_ram[i] = 0xFF
            self.bg_palette_ram[i + 1] = 0x7F
            self.obj_palette_ram[i] = 0xFF
            self.obj_palette_ram[i + 1] = 0x7F
        self.bgpi = 0
        self.obpi = 0

        self.has_battery = False
        self.save_path = None
        self.rom_title = ""
        self.rom_hash = ""
        self.double_speed = False
        self.speed_switch_armed = False

    def load_rom(self, data):
        self.rom = bytearray(data)
        self.cart_type = self.rom[0x0147] if len(self.rom) > 0x0147 else 0
        rom_size_code = self.rom[0x0148] if len(self.rom) > 0x0148 else 0
        self.rom_banks = 2 << rom_size_code
        ram_code = self.rom[0x0149] if len(self.rom) > 0x0149 else 0
        ram_sizes = {0: 0, 1: 2048, 2: 8192, 3: 32768, 4: 131072, 5: 65536}
        self.ram_size = ram_sizes.get(ram_code, 0)
        self.eram = bytearray(self.ram_size if self.ram_size > 0 else 8192)

        battery_types = {0x03, 0x06, 0x09, 0x0D, 0x0F, 0x10, 0x13, 0x1B, 0x1E, 0x22, 0xFF}
        self.has_battery = self.cart_type in battery_types

        # CGB flag
        self.cgb_mode = len(self.rom) > 0x0143 and (self.rom[0x0143] & 0x80) != 0

        title = ""
        for i in range(0x0134, 0x0143):
            if i < len(self.rom) and 32 <= self.rom[i] < 127:
                title += chr(self.rom[i])
        self.rom_title = title.strip()
        self.rom_hash = hashlib.sha1(self.rom[: min(len(self.rom), 0x200)]).hexdigest()[:16]

        print(
            f"ROM: {self.rom_title} | Type: 0x{self.cart_type:02X} | Banks: {self.rom_banks} | "
            f"RAM: {self.ram_size} B | CGB: {self.cgb_mode} | Battery: {self.has_battery}"
        )

    def print_rom_info(self):
        mapper_names = {
            0x00: "ROM ONLY", 0x01: "MBC1", 0x02: "MBC1+RAM", 0x03: "MBC1+RAM+BATTERY",
            0x05: "MBC2", 0x06: "MBC2+BATTERY", 0x0F: "MBC3+TIMER+BATTERY",
            0x10: "MBC3+TIMER+RAM+BATTERY", 0x11: "MBC3", 0x12: "MBC3+RAM",
            0x13: "MBC3+RAM+BATTERY", 0x19: "MBC5", 0x1A: "MBC5+RAM",
            0x1B: "MBC5+RAM+BATTERY", 0x1C: "MBC5+RUMBLE", 0x1D: "MBC5+RUMBLE+RAM",
            0x1E: "MBC5+RUMBLE+RAM+BATTERY",
        }
        print("=" * 50)
        print(f"Title          : {self.rom_title}")
        print(f"Cart type      : 0x{self.cart_type:02X} ({mapper_names.get(self.cart_type, 'Unknown')})")
        print(f"ROM banks      : {self.rom_banks}")
        print(f"RAM size       : {self.ram_size} bytes")
        print(f"CGB            : {self.cgb_mode}")
        print(f"Battery        : {self.has_battery}")
        print(f"Hash (partial) : {self.rom_hash}")
        print("=" * 50)

    def load_save(self, path):
        self.save_path = path
        if self.has_battery and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                n = min(len(data), len(self.eram))
                self.eram[:n] = data[:n]
                print(f"Loaded save: {path} ({n} bytes)")
            except Exception as e:
                print(f"Could not load save: {e}")

    def write_save(self):
        if self.has_battery and self.save_path:
            try:
                with open(self.save_path, "wb") as f:
                    f.write(bytes(self.eram))
            except Exception as e:
                print(f"Could not write save: {e}")

    def tick_rtc(self, cycles):
        if self.cart_type not in (0x0F, 0x10, 0x11, 0x12, 0x13):
            return
        if self.rtc_regs[4] & 0x40:
            return
        self._rtc_cycle_acc += cycles
        while self._rtc_cycle_acc >= 4194304:
            self._rtc_cycle_acc -= 4194304
            self.rtc_regs[0] += 1
            if self.rtc_regs[0] >= 60:
                self.rtc_regs[0] = 0
                self.rtc_regs[1] += 1
                if self.rtc_regs[1] >= 60:
                    self.rtc_regs[1] = 0
                    self.rtc_regs[2] += 1
                    if self.rtc_regs[2] >= 24:
                        self.rtc_regs[2] = 0
                        day = ((self.rtc_regs[4] & 0x01) << 8) | self.rtc_regs[3]
                        day += 1
                        if day > 511:
                            day = 0
                            self.rtc_regs[4] |= 0x80
                        self.rtc_regs[3] = day & 0xFF
                        self.rtc_regs[4] = (self.rtc_regs[4] & ~0x01) | ((day >> 8) & 0x01)

    def read_byte(self, addr):
        if 0x0000 <= addr < 0x4000:
            if self.cart_type in (1, 2, 3) and self.mbc1_mode == 1:
                bank = (self.mbc1_rom_bank_high << 5) % self.rom_banks
                rom_addr = bank * 0x4000 + addr
                return self.rom[rom_addr] if rom_addr < len(self.rom) else 0xFF
            return self.rom[addr] if addr < len(self.rom) else 0xFF

        if 0x4000 <= addr < 0x8000:
            bank = 1
            if self.cart_type in (1, 2, 3):
                low = self.mbc1_rom_bank_low if self.mbc1_rom_bank_low != 0 else 1
                bank = ((self.mbc1_rom_bank_high << 5) | low) % self.rom_banks
            elif self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                bank = (self.rom_bank if self.rom_bank != 0 else 1) % self.rom_banks
            elif self.cart_type in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                bank = ((self.mbc5_rom_bank_high << 8) | self.mbc5_rom_bank_low) % self.rom_banks
            if self.rom:
                rom_addr = bank * 0x4000 + (addr - 0x4000)
                return self.rom[rom_addr] if rom_addr < len(self.rom) else 0xFF
            return 0xFF

        if 0x8000 <= addr < 0xA000:
            return self.vram[self.vram_bank * 0x2000 + (addr - 0x8000)]

        if 0xA000 <= addr < 0xC000:
            if not self.ram_enabled:
                return 0xFF
            if self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13) and 0x08 <= self.ram_bank <= 0x0C:
                return self.rtc_latch[self.ram_bank - 0x08]
            bank = 0
            if self.cart_type in (1, 2, 3) and self.mbc1_mode == 1:
                bank = self.ram_bank
            elif self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                bank = self.ram_bank & 0x03
            elif self.cart_type in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                bank = self.ram_bank & 0x0F
            if self.ram_size > 0:
                return self.eram[((bank * 0x2000) + (addr - 0xA000)) % self.ram_size]
            return 0xFF

        if 0xC000 <= addr < 0xD000:
            return self.wram_banks[0][addr - 0xC000]
        if 0xD000 <= addr < 0xE000:
            bank = self.wram_bank if self.wram_bank != 0 else 1
            return self.wram_banks[bank][addr - 0xD000]
        if 0xE000 <= addr < 0xFE00:
            return self.read_byte(addr - 0x2000)
        if 0xFE00 <= addr < 0xFEA0:
            return self.oam[addr - 0xFE00]
        if 0xFEA0 <= addr < 0xFF00:
            return 0x00

        if 0xFF00 <= addr < 0xFF80:
            offset = addr - 0xFF00
            if offset == 0x00:
                val = self.io[0x00] | 0xC0
                joy_val = 0x0F
                if not (val & 0x10):
                    joy_val &= self.joy_directions
                if not (val & 0x20):
                    joy_val &= self.joy_buttons
                return (val & 0xF0) | joy_val
            if offset == 0x0F:
                return self.io[0x0F] | 0xE0
            if offset == 0x26:
                val = self.io[0x26] & 0x80
                if self.apu:
                    now = pygame.time.get_ticks()
                    if now < self.apu.ch1_end_time:
                        val |= 0x01
                    if now < self.apu.ch2_end_time:
                        val |= 0x02
                    if now < self.apu.ch3_end_time:
                        val |= 0x04
                    if now < self.apu.ch4_end_time:
                        val |= 0x08
                return val | 0x70
            if offset == 0x4D:  # KEY1
                return (0x80 if self.double_speed else 0x00) | (0x01 if self.speed_switch_armed else 0x00) | 0x7E
            if offset == 0x69:
                return self.bg_palette_ram[self.bgpi & 0x3F]
            if offset == 0x6B:
                return self.obj_palette_ram[self.obpi & 0x3F]
            return self.io[offset]

        if 0xFF80 <= addr < 0xFFFF:
            return self.hram[addr - 0xFF80]
        if addr == 0xFFFF:
            return self.ie | 0xE0
        return 0xFF

    def write_byte(self, addr, val):
        val &= 0xFF
        if 0x0000 <= addr < 0x2000:
            self.ram_enabled = (val & 0x0F) == 0x0A
        elif 0x2000 <= addr < 0x4000:
            if self.cart_type in (1, 2, 3):
                self.mbc1_rom_bank_low = val & 0x1F
            elif self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                self.rom_bank = val & 0x7F
                if self.rom_bank == 0:
                    self.rom_bank = 1
            elif self.cart_type in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                if addr < 0x3000:
                    self.mbc5_rom_bank_low = val
                else:
                    self.mbc5_rom_bank_high = val & 1
        elif 0x4000 <= addr < 0x6000:
            if self.cart_type in (1, 2, 3):
                self.mbc1_rom_bank_high = val & 3
                self.ram_bank = val & 3
            elif self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                self.ram_bank = val
            elif self.cart_type in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                self.ram_bank = val & 0x0F
        elif 0x6000 <= addr < 0x8000:
            if self.cart_type in (1, 2, 3):
                self.mbc1_mode = val & 1
            elif self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                if val == 0x00:
                    self._rtc_latch_pending = True
                elif val == 0x01 and self._rtc_latch_pending:
                    self.rtc_latch = list(self.rtc_regs)
                    self._rtc_latch_pending = False
                else:
                    self._rtc_latch_pending = False
        elif 0x8000 <= addr < 0xA000:
            self.vram[self.vram_bank * 0x2000 + (addr - 0x8000)] = val
        elif 0xA000 <= addr < 0xC000:
            if not self.ram_enabled:
                return
            if self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13) and 0x08 <= self.ram_bank <= 0x0C:
                self.rtc_regs[self.ram_bank - 0x08] = val
                return
            bank = 0
            if self.cart_type in (1, 2, 3) and self.mbc1_mode == 1:
                bank = self.ram_bank
            elif self.cart_type in (0x0F, 0x10, 0x11, 0x12, 0x13):
                bank = self.ram_bank & 0x03
            elif self.cart_type in (0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E):
                bank = self.ram_bank & 0x0F
            if self.ram_size > 0:
                self.eram[((bank * 0x2000) + (addr - 0xA000)) % self.ram_size] = val
        elif 0xC000 <= addr < 0xD000:
            self.wram_banks[0][addr - 0xC000] = val
        elif 0xD000 <= addr < 0xE000:
            bank = self.wram_bank if self.wram_bank != 0 else 1
            self.wram_banks[bank][addr - 0xD000] = val
        elif 0xE000 <= addr < 0xFE00:
            self.write_byte(addr - 0x2000, val)
        elif 0xFE00 <= addr < 0xFEA0:
            self.oam[addr - 0xFE00] = val
        elif 0xFEA0 <= addr < 0xFF00:
            return
        elif 0xFF00 <= addr < 0xFF80:
            offset = addr - 0xFF00
            if offset == 0x00:
                self.io[0x00] = (self.io[0x00] & 0xCF) | (val & 0x30)
            elif offset == 0x02:
                if val & 0x80:
                    self.io[0x02] = val & 0x7F
                    self.io[0x0F] |= 0x08
                else:
                    self.io[0x02] = val
            elif offset == 0x04:
                self.io[0x04] = 0
                if self.timer:
                    self.timer.div_cycles = 0
            elif offset == 0x0F:
                self.io[0x0F] = val & 0x1F
            elif offset == 0x14:
                self.io[0x14] = val
                if val & 0x80 and self.apu:
                    self.apu.trigger_channel1(self)
            elif offset == 0x19:
                self.io[0x19] = val
                if val & 0x80 and self.apu:
                    self.apu.trigger_channel2(self)
            elif offset == 0x1E:
                self.io[0x1E] = val
                if val & 0x80 and self.apu:
                    self.apu.trigger_channel3(self)
            elif offset == 0x23:
                self.io[0x23] = val
                if val & 0x80 and self.apu:
                    self.apu.trigger_channel4(self)
            elif offset == 0x26:
                self.io[0x26] = (self.io[0x26] & 0x0F) | (val & 0x80)
                if not (val & 0x80) and self.apu:
                    self.apu.stop_all()
            elif offset == 0x41:
                self.io[0x41] = (val & 0x78) | (self.io[0x41] & 0x07)
            elif offset == 0x44:
                pass  # LY is read-only
            elif offset == 0x46:  # DMA
                self.io[0x46] = val
                src = val << 8
                for i in range(160):
                    self.oam[i] = self.read_byte(src + i)
            elif offset == 0x4D:  # KEY1
                self.speed_switch_armed = bool(val & 0x01)
                self.io[0x4D] = val
            elif offset == 0x4F:  # VBK
                self.vram_bank = val & 0x01
                self.io[0x4F] = 0xFE | self.vram_bank
            elif offset == 0x55:  # HDMA
                self.io[0x55] = val
                src = ((self.io[0x51] << 8) | self.io[0x52]) & 0xFFF0
                dst = 0x8000 + (((self.io[0x53] << 8) | self.io[0x54]) & 0x1FF0)
                length = ((val & 0x7F) + 1) * 16
                for i in range(length):
                    self.write_byte(dst + i, self.read_byte(src + i))
                self.io[0x55] = 0xFF
            elif offset == 0x68:  # BCPS
                self.bgpi = val
                self.io[0x68] = val
            elif offset == 0x69:  # BCPD
                idx = self.bgpi & 0x3F
                self.bg_palette_ram[idx] = val
                if self.bgpi & 0x80:
                    self.bgpi = 0x80 | ((idx + 1) & 0x3F)
            elif offset == 0x6A:  # OCPS
                self.obpi = val
                self.io[0x6A] = val
            elif offset == 0x6B:  # OCPD
                idx = self.obpi & 0x3F
                self.obj_palette_ram[idx] = val
                if self.obpi & 0x80:
                    self.obpi = 0x80 | ((idx + 1) & 0x3F)
            elif offset == 0x70:  # SVBK
                bank = val & 0x07
                self.wram_bank = bank if bank != 0 else 1
                self.io[0x70] = val
            else:
                self.io[offset] = val
        elif 0xFF80 <= addr < 0xFFFF:
            self.hram[addr - 0xFF80] = val
        elif addr == 0xFFFF:
            self.ie = val

    def read_word(self, addr):
        return self.read_byte(addr) | (self.read_byte(addr + 1) << 8)

    def write_word(self, addr, val):
        self.write_byte(addr, val & 0xFF)
        self.write_byte(addr + 1, (val >> 8) & 0xFF)

    def update_input(self, keys_pressed, joy=None):
        old_dir = self.joy_directions
        old_but = self.joy_buttons
        joy_dir = 0x0F
        joy_but = 0x0F

        # Keyboard
        for bit, name in ((0x01, "right"), (0x02, "left"), (0x04, "up"), (0x08, "down")):
            k = get_key(name)
            if k is not None and keys_pressed[k]:
                joy_dir &= ~bit
        for bit, name in ((0x01, "a"), (0x02, "b"), (0x04, "select"), (0x08, "start")):
            k = get_key(name)
            if k is not None and keys_pressed[k]:
                joy_but &= ~bit

        # Controller (Xbox style)
        if joy is not None:
            try:
                if joy.get_numhats() > 0:
                    hx, hy = joy.get_hat(0)
                    if hx > 0:
                        joy_dir &= ~0x01
                    if hx < 0:
                        joy_dir &= ~0x02
                    if hy > 0:
                        joy_dir &= ~0x04
                    if hy < 0:
                        joy_dir &= ~0x08

                if joy.get_numaxes() >= 2:
                    ax = joy.get_axis(0)
                    ay = joy.get_axis(1)
                    if ax > 0.4:
                        joy_dir &= ~0x01
                    if ax < -0.4:
                        joy_dir &= ~0x02
                    if ay < -0.4:
                        joy_dir &= ~0x04
                    if ay > 0.4:
                        joy_dir &= ~0x08

                nbtn = joy.get_numbuttons()
                if (nbtn > 0 and joy.get_button(0)) or (nbtn > 2 and joy.get_button(2)):
                    joy_but &= ~0x01
                if (nbtn > 1 and joy.get_button(1)) or (nbtn > 3 and joy.get_button(3)):
                    joy_but &= ~0x02
                if nbtn > 6 and joy.get_button(6):
                    joy_but &= ~0x04
                if nbtn > 7 and joy.get_button(7):
                    joy_but &= ~0x08
            except Exception:
                pass

        self.joy_directions = joy_dir
        self.joy_buttons = joy_but
        if joy_dir != old_dir or joy_but != old_but:
            if ((old_dir & ~joy_dir) & 0x0F) or ((old_but & ~joy_but) & 0x0F):
                self.io[0x0F] |= 0x10

    def get_state(self):
        return {
            "vram": bytes(self.vram), "vram_bank": self.vram_bank,
            "eram": bytes(self.eram),
            "wram_banks": [bytes(b) for b in self.wram_banks], "wram_bank": self.wram_bank,
            "oam": bytes(self.oam), "hram": bytes(self.hram), "ie": self.ie, "io": bytes(self.io),
            "joy_buttons": self.joy_buttons, "joy_directions": self.joy_directions,
            "ram_enabled": self.ram_enabled, "mbc1_mode": self.mbc1_mode,
            "mbc1_rom_bank_low": self.mbc1_rom_bank_low, "mbc1_rom_bank_high": self.mbc1_rom_bank_high,
            "mbc5_rom_bank_low": self.mbc5_rom_bank_low, "mbc5_rom_bank_high": self.mbc5_rom_bank_high,
            "rom_bank": self.rom_bank, "ram_bank": self.ram_bank,
            "rtc_regs": list(self.rtc_regs), "rtc_latch": list(self.rtc_latch),
            "_rtc_latch_pending": self._rtc_latch_pending, "_rtc_cycle_acc": self._rtc_cycle_acc,
            "cgb_mode": self.cgb_mode,
            "bg_palette_ram": bytes(self.bg_palette_ram), "obj_palette_ram": bytes(self.obj_palette_ram),
            "bgpi": self.bgpi, "obpi": self.obpi,
            "double_speed": self.double_speed, "speed_switch_armed": self.speed_switch_armed,
        }

    def set_state(self, state):
        self.vram[:] = state["vram"]
        self.vram_bank = state["vram_bank"]
        self.eram[:] = state["eram"]
        for i, b in enumerate(state["wram_banks"]):
            self.wram_banks[i][:] = b
        self.wram_bank = state["wram_bank"]
        self.oam[:] = state["oam"]
        self.hram[:] = state["hram"]
        self.ie = state["ie"]
        self.io[:] = state["io"]
        self.joy_buttons = state["joy_buttons"]
        self.joy_directions = state["joy_directions"]
        self.ram_enabled = state["ram_enabled"]
        self.mbc1_mode = state["mbc1_mode"]
        self.mbc1_rom_bank_low = state["mbc1_rom_bank_low"]
        self.mbc1_rom_bank_high = state["mbc1_rom_bank_high"]
        self.mbc5_rom_bank_low = state["mbc5_rom_bank_low"]
        self.mbc5_rom_bank_high = state["mbc5_rom_bank_high"]
        self.rom_bank = state["rom_bank"]
        self.ram_bank = state["ram_bank"]
        self.rtc_regs = list(state["rtc_regs"])
        self.rtc_latch = list(state["rtc_latch"])
        self._rtc_latch_pending = state["_rtc_latch_pending"]
        self._rtc_cycle_acc = state["_rtc_cycle_acc"]
        self.cgb_mode = state["cgb_mode"]
        self.bg_palette_ram[:] = state["bg_palette_ram"]
        self.obj_palette_ram[:] = state["obj_palette_ram"]
        self.bgpi = state["bgpi"]
        self.obpi = state["obpi"]
        self.double_speed = state.get("double_speed", False)
        self.speed_switch_armed = state.get("speed_switch_armed", False)


# ---------------------------------------------------------------------------
# CPU - improved timing and cleaner code
# ---------------------------------------------------------------------------
class CPU:
    def __init__(self, mmu):
        self.mmu = mmu
        self.a = self.f = self.b = self.c = self.d = self.e = self.h = self.l = 0
        self.sp = self.pc = 0
        self.ime = False
        self.halted = False
        self.ei_delay = 0
        self.halt_bug = False
        self._warned_opcodes = set()
        self.last_opcode = 0
        self.init_registers_post_boot()

    def init_registers_post_boot(self):
        # Correct post-boot values
        if self.mmu.cgb_mode:
            self.a, self.f = 0x11, 0x80
            self.b, self.c = 0x00, 0x00
            self.d, self.e = 0xFF, 0x56
            self.h, self.l = 0x00, 0x0D
        else:
            self.a, self.f = 0x01, 0xB0
            self.b, self.c = 0x00, 0x13
            self.d, self.e = 0x00, 0xD8
            self.h, self.l = 0x01, 0x4D
        self.sp, self.pc = 0xFFFE, 0x0100

        # Common I/O defaults
        for addr, val in (
            (0xFF05, 0x00), (0xFF06, 0x00), (0xFF07, 0x00),
            (0xFF10, 0x80), (0xFF11, 0xBF), (0xFF12, 0xF3), (0xFF14, 0xBF),
            (0xFF16, 0x3F), (0xFF17, 0x00), (0xFF19, 0xBF),
            (0xFF1A, 0x7F), (0xFF1B, 0xFF), (0xFF1C, 0x9F), (0xFF1E, 0xBF),
            (0xFF20, 0xFF), (0xFF21, 0x00), (0xFF22, 0x00), (0xFF23, 0xBF),
            (0xFF24, 0x77), (0xFF25, 0xF3), (0xFF26, 0xF1),
            (0xFF40, 0x91), (0xFF42, 0x00), (0xFF43, 0x00), (0xFF45, 0x00),
            (0xFF47, 0xFC), (0xFF48, 0xFF), (0xFF49, 0xFF),
            (0xFF4A, 0x00), (0xFF4B, 0x00), (0xFFFF, 0x00),
        ):
            self.mmu.write_byte(addr, val)

        if self.mmu.cgb_mode:
            self.mmu.write_byte(0xFF4F, 0x00)
            self.mmu.write_byte(0xFF70, 0x01)

    @property
    def bc(self):
        return (self.b << 8) | self.c

    @bc.setter
    def bc(self, val):
        self.b, self.c = (val >> 8) & 0xFF, val & 0xFF

    @property
    def de(self):
        return (self.d << 8) | self.e

    @de.setter
    def de(self, val):
        self.d, self.e = (val >> 8) & 0xFF, val & 0xFF

    @property
    def hl(self):
        return (self.h << 8) | self.l

    @hl.setter
    def hl(self, val):
        self.h, self.l = (val >> 8) & 0xFF, val & 0xFF

    @property
    def af(self):
        return (self.a << 8) | (self.f & 0xF0)

    @af.setter
    def af(self, val):
        self.a, self.f = (val >> 8) & 0xFF, val & 0xF0

    @property
    def flag_z(self):
        return bool(self.f & 0x80)

    @flag_z.setter
    def flag_z(self, b):
        self.f = (self.f & 0x7F) | (0x80 if b else 0)

    @property
    def flag_n(self):
        return bool(self.f & 0x40)

    @flag_n.setter
    def flag_n(self, b):
        self.f = (self.f & 0xBF) | (0x40 if b else 0)

    @property
    def flag_h(self):
        return bool(self.f & 0x20)

    @flag_h.setter
    def flag_h(self, b):
        self.f = (self.f & 0xDF) | (0x20 if b else 0)

    @property
    def flag_c(self):
        return bool(self.f & 0x10)

    @flag_c.setter
    def flag_c(self, b):
        self.f = (self.f & 0xEF) | (0x10 if b else 0)

    def get_reg8(self, reg):
        if reg == 0: return self.b
        if reg == 1: return self.c
        if reg == 2: return self.d
        if reg == 3: return self.e
        if reg == 4: return self.h
        if reg == 5: return self.l
        if reg == 6: return self.mmu.read_byte(self.hl)
        return self.a

    def set_reg8(self, reg, val):
        val &= 0xFF
        if reg == 0: self.b = val
        elif reg == 1: self.c = val
        elif reg == 2: self.d = val
        elif reg == 3: self.e = val
        elif reg == 4: self.h = val
        elif reg == 5: self.l = val
        elif reg == 6: self.mmu.write_byte(self.hl, val)
        else: self.a = val

    def get_reg16(self, reg, stack=False):
        if reg == 0: return self.bc
        if reg == 1: return self.de
        if reg == 2: return self.hl
        return self.af if stack else self.sp

    def set_reg16(self, reg, val, stack=False):
        val &= 0xFFFF
        if reg == 0: self.bc = val
        elif reg == 1: self.de = val
        elif reg == 2: self.hl = val
        else:
            if stack: self.af = val
            else: self.sp = val

    def check_condition(self, cond):
        if cond == 0: return not self.flag_z
        if cond == 1: return self.flag_z
        if cond == 2: return not self.flag_c
        return self.flag_c

    def fetch_byte(self):
        val = self.mmu.read_byte(self.pc)
        if self.halt_bug:
            self.halt_bug = False
        else:
            self.pc = (self.pc + 1) & 0xFFFF
        return val

    def fetch_signed_byte(self):
        val = self.fetch_byte()
        return val - 256 if val >= 128 else val

    def fetch_word(self):
        low = self.fetch_byte()
        high = self.fetch_byte()
        return (high << 8) | low

    def push_word(self, val):
        self.sp = (self.sp - 1) & 0xFFFF
        self.mmu.write_byte(self.sp, (val >> 8) & 0xFF)
        self.sp = (self.sp - 1) & 0xFFFF
        self.mmu.write_byte(self.sp, val & 0xFF)

    def pop_word(self):
        low = self.mmu.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        high = self.mmu.read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return (high << 8) | low

    def execute_alu(self, op, val):
        if op == 0:  # ADD
            res = self.a + val
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = False
            self.flag_h = ((self.a & 0x0F) + (val & 0x0F)) > 0x0F
            self.flag_c = res > 0xFF
            self.a = res & 0xFF
        elif op == 1:  # ADC
            c = 1 if self.flag_c else 0
            res = self.a + val + c
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = False
            self.flag_h = ((self.a & 0x0F) + (val & 0x0F) + c) > 0x0F
            self.flag_c = res > 0xFF
            self.a = res & 0xFF
        elif op == 2:  # SUB
            res = self.a - val
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = True
            self.flag_h = (self.a & 0x0F) < (val & 0x0F)
            self.flag_c = self.a < val
            self.a = res & 0xFF
        elif op == 3:  # SBC
            c = 1 if self.flag_c else 0
            res = self.a - val - c
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = True
            self.flag_h = (self.a & 0x0F) < ((val & 0x0F) + c)
            self.flag_c = self.a < (val + c)
            self.a = res & 0xFF
        elif op == 4:  # AND
            self.a &= val
            self.flag_z = self.a == 0
            self.flag_n, self.flag_h, self.flag_c = False, True, False
        elif op == 5:  # XOR
            self.a ^= val
            self.flag_z = self.a == 0
            self.flag_n, self.flag_h, self.flag_c = False, False, False
        elif op == 6:  # OR
            self.a |= val
            self.flag_z = self.a == 0
            self.flag_n, self.flag_h, self.flag_c = False, False, False
        else:  # CP
            res = self.a - val
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = True
            self.flag_h = (self.a & 0x0F) < (val & 0x0F)
            self.flag_c = self.a < val

    def execute_cb(self):
        cb_op = self.fetch_byte()
        op_type = (cb_op >> 6) & 0x03
        bit = (cb_op >> 3) & 0x07
        reg = cb_op & 0x07
        if op_type == 1:  # BIT
            val = self.get_reg8(reg)
            self.flag_z = (val & (1 << bit)) == 0
            self.flag_n, self.flag_h = False, True
        elif op_type == 2:  # RES
            self.set_reg8(reg, self.get_reg8(reg) & ~(1 << bit))
        elif op_type == 3:  # SET
            self.set_reg8(reg, self.get_reg8(reg) | (1 << bit))
        else:  # rotates / shifts / swap
            val = self.get_reg8(reg)
            if bit == 0:  # RLC
                c = (val >> 7) & 1
                res = ((val << 1) | c) & 0xFF
                self.flag_c = bool(c)
            elif bit == 1:  # RRC
                c = val & 1
                res = ((val >> 1) | (c << 7)) & 0xFF
                self.flag_c = bool(c)
            elif bit == 2:  # RL
                c = 1 if self.flag_c else 0
                res = ((val << 1) | c) & 0xFF
                self.flag_c = bool((val >> 7) & 1)
            elif bit == 3:  # RR
                c = 0x80 if self.flag_c else 0
                res = ((val >> 1) | c) & 0xFF
                self.flag_c = bool(val & 1)
            elif bit == 4:  # SLA
                res = (val << 1) & 0xFF
                self.flag_c = bool((val >> 7) & 1)
            elif bit == 5:  # SRA
                res = (val >> 1) | (val & 0x80)
                self.flag_c = bool(val & 1)
            elif bit == 6:  # SWAP
                res = ((val & 0x0F) << 4) | ((val & 0xF0) >> 4)
                self.flag_c = False
            else:  # SRL
                res = (val >> 1) & 0xFF
                self.flag_c = bool(val & 1)
            self.flag_z = res == 0
            self.flag_n = self.flag_h = False
            self.set_reg8(reg, res)
        return cb_op

    def handle_interrupts(self):
        if_reg = self.mmu.io[0x0F]
        ie_reg = self.mmu.ie
        fired = if_reg & ie_reg & 0x1F
        if fired:
            self.halted = False
            if self.ime:
                for i in range(5):
                    if fired & (1 << i):
                        self.ime = False
                        self.mmu.io[0x0F] = if_reg & ~(1 << i)
                        self.push_word(self.pc)
                        self.pc = [0x0040, 0x0048, 0x0050, 0x0058, 0x0060][i]
                        return 20
        return 0

    def step(self):
        if self.ei_delay > 0:
            self.ei_delay -= 1
            if self.ei_delay == 0:
                self.ime = True

        cycles = self.handle_interrupts()
        if cycles > 0:
            return cycles

        if self.halted:
            return 4

        opcode = self.fetch_byte()
        self.last_opcode = opcode

        # LD r,r' and HALT
        if 0x40 <= opcode <= 0x7F:
            if opcode == 0x76:
                if self.ime:
                    self.halted = True
                else:
                    if (self.mmu.read_byte(0xFFFF) & self.mmu.read_byte(0xFF0F) & 0x1F) != 0:
                        self.halt_bug = True
                    else:
                        self.halted = True
                return 4
            y, x = (opcode >> 3) & 0x07, opcode & 0x07
            self.set_reg8(y, self.get_reg8(x))
            return 8 if (y == 6 or x == 6) else 4

        # ALU A,r
        if 0x80 <= opcode <= 0xBF:
            op, reg = (opcode >> 3) & 0x07, opcode & 0x07
            self.execute_alu(op, self.get_reg8(reg))
            return 8 if reg == 6 else 4

        # LD r,n
        if (opcode & 0xC7) == 0x06:
            y = (opcode >> 3) & 0x07
            self.set_reg8(y, self.fetch_byte())
            return 12 if y == 6 else 8

        # INC r
        if (opcode & 0xC7) == 0x04:
            y = (opcode >> 3) & 0x07
            orig = self.get_reg8(y)
            val = (orig + 1) & 0xFF
            self.flag_z = val == 0
            self.flag_n, self.flag_h = False, (orig & 0x0F) == 0x0F
            self.set_reg8(y, val)
            return 12 if y == 6 else 4

        # DEC r
        if (opcode & 0xC7) == 0x05:
            y = (opcode >> 3) & 0x07
            orig = self.get_reg8(y)
            val = (orig - 1) & 0xFF
            self.flag_z = val == 0
            self.flag_n, self.flag_h = True, (orig & 0x0F) == 0x00
            self.set_reg8(y, val)
            return 12 if y == 6 else 4

        # JR cc,e
        if (opcode & 0xE7) == 0x20:
            cc = (opcode >> 3) & 0x03
            offset = self.fetch_signed_byte()
            if self.check_condition(cc):
                self.pc = (self.pc + offset) & 0xFFFF
                return 12
            return 8

        # JP cc,nn
        if (opcode & 0xE7) == 0xC2:
            cc = (opcode >> 3) & 0x03
            addr = self.fetch_word()
            if self.check_condition(cc):
                self.pc = addr & 0xFFFF
                return 16
            return 12

        # CALL cc,nn
        if (opcode & 0xE7) == 0xC4:
            cc = (opcode >> 3) & 0x03
            addr = self.fetch_word()
            if self.check_condition(cc):
                self.push_word(self.pc)
                self.pc = addr & 0xFFFF
                return 24
            return 12

        # RET cc
        if (opcode & 0xE7) == 0xC0:
            cc = (opcode >> 3) & 0x03
            if self.check_condition(cc):
                self.pc = self.pop_word() & 0xFFFF
                return 20
            return 8

        # INC rr
        if (opcode & 0xCF) == 0x03:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, (self.get_reg16(q) + 1) & 0xFFFF)
            return 8

        # DEC rr
        if (opcode & 0xCF) == 0x0B:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, (self.get_reg16(q) - 1) & 0xFFFF)
            return 8

        # ADD HL,rr
        if (opcode & 0xCF) == 0x09:
            q = (opcode >> 4) & 0x03
            val = self.get_reg16(q)
            hl_val = self.hl
            res = hl_val + val
            self.flag_n = False
            self.flag_h = ((hl_val & 0x0FFF) + (val & 0x0FFF)) > 0x0FFF
            self.flag_c = res > 0xFFFF
            self.hl = res & 0xFFFF
            return 8

        # LD rr,nn
        if (opcode & 0xCF) == 0x01:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, self.fetch_word() & 0xFFFF)
            return 12

        # POP rr
        if (opcode & 0xCF) == 0xC1:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, self.pop_word() & 0xFFFF, stack=True)
            return 12

        # PUSH rr
        if (opcode & 0xCF) == 0xC5:
            q = (opcode >> 4) & 0x03
            self.push_word(self.get_reg16(q, stack=True))
            return 16

        # RST
        if (opcode & 0xC7) == 0xC7:
            self.push_word(self.pc)
            self.pc = (opcode & 0x38) & 0xFFFF
            return 16

        # ALU A,n
        if (opcode & 0xC7) == 0xC6:
            op = (opcode >> 3) & 0x07
            self.execute_alu(op, self.fetch_byte())
            return 8

        # Individual opcodes
        if opcode == 0x00: return 4
        if opcode == 0x10:  # STOP
            self.fetch_byte()
            if self.mmu.cgb_mode and self.mmu.speed_switch_armed:
                self.mmu.double_speed = not self.mmu.double_speed
                self.mmu.speed_switch_armed = False
            return 4
        if opcode == 0x18:
            offset = self.fetch_signed_byte()
            self.pc = (self.pc + offset) & 0xFFFF
            return 12
        if opcode == 0xCD:
            addr = self.fetch_word()
            self.push_word(self.pc)
            self.pc = addr & 0xFFFF
            return 24
        if opcode == 0xC9:
            self.pc = self.pop_word() & 0xFFFF
            return 16
        if opcode == 0xD9:
            self.pc = self.pop_word() & 0xFFFF
            self.ime = True
            return 16
        if opcode == 0xC3:
            self.pc = self.fetch_word() & 0xFFFF
            return 16
        if opcode == 0xE9:
            self.pc = self.hl
            return 4
        if opcode == 0x22:
            self.mmu.write_byte(self.hl, self.a)
            self.hl = (self.hl + 1) & 0xFFFF
            return 8
        if opcode == 0x32:
            self.mmu.write_byte(self.hl, self.a)
            self.hl = (self.hl - 1) & 0xFFFF
            return 8
        if opcode == 0x2A:
            self.a = self.mmu.read_byte(self.hl)
            self.hl = (self.hl + 1) & 0xFFFF
            return 8
        if opcode == 0x3A:
            self.a = self.mmu.read_byte(self.hl)
            self.hl = (self.hl - 1) & 0xFFFF
            return 8
        if opcode == 0xE0:
            self.mmu.write_byte(0xFF00 + self.fetch_byte(), self.a)
            return 12
        if opcode == 0xF0:
            self.a = self.mmu.read_byte(0xFF00 + self.fetch_byte())
            return 12
        if opcode == 0xE2:
            self.mmu.write_byte(0xFF00 + self.c, self.a)
            return 8
        if opcode == 0xF2:
            self.a = self.mmu.read_byte(0xFF00 + self.c)
            return 8
        if opcode == 0xEA:
            self.mmu.write_byte(self.fetch_word(), self.a)
            return 16
        if opcode == 0xFA:
            self.a = self.mmu.read_byte(self.fetch_word())
            return 16
        if opcode == 0x08:
            self.mmu.write_word(self.fetch_word(), self.sp)
            return 20
        if opcode == 0xF9:
            self.sp = self.hl
            return 8
        if opcode == 0x02:
            self.mmu.write_byte(self.bc, self.a)
            return 8
        if opcode == 0x12:
            self.mmu.write_byte(self.de, self.a)
            return 8
        if opcode == 0x0A:
            self.a = self.mmu.read_byte(self.bc)
            return 8
        if opcode == 0x1A:
            self.a = self.mmu.read_byte(self.de)
            return 8
        if opcode == 0xCB:
            cb_op = self.execute_cb()
            reg = cb_op & 0x07
            op_type = (cb_op >> 6) & 0x03
            if reg == 6:
                return 12 if op_type == 1 else 16
            return 8
        if opcode == 0xF3:
            self.ime = False
            self.ei_delay = 0
            return 4
        if opcode == 0xFB:
            self.ei_delay = 2
            return 4
        if opcode == 0x2F:
            self.a ^= 0xFF
            self.flag_n = self.flag_h = True
            return 4
        if opcode == 0x37:
            self.flag_n = self.flag_h = False
            self.flag_c = True
            return 4
        if opcode == 0x3F:
            self.flag_n = self.flag_h = False
            self.flag_c = not self.flag_c
            return 4
        if opcode == 0x27:  # DAA
            correction = 0
            set_carry = False
            if self.flag_h or (not self.flag_n and (self.a & 0x0F) > 9):
                correction |= 0x06
            if self.flag_c or (not self.flag_n and self.a > 0x99):
                correction |= 0x60
                set_carry = True
            if self.flag_n:
                self.a = (self.a - correction) & 0xFF
            else:
                self.a = (self.a + correction) & 0xFF
            self.flag_z = self.a == 0
            self.flag_h = False
            self.flag_c = set_carry
            return 4
        if opcode == 0x07:  # RLCA
            c = (self.a >> 7) & 1
            self.a = ((self.a << 1) | c) & 0xFF
            self.flag_z = self.flag_n = self.flag_h = False
            self.flag_c = bool(c)
            return 4
        if opcode == 0x0F:  # RRCA
            c = self.a & 1
            self.a = ((self.a >> 1) | (c << 7)) & 0xFF
            self.flag_z = self.flag_n = self.flag_h = False
            self.flag_c = bool(c)
            return 4
        if opcode == 0x17:  # RLA
            c = 1 if self.flag_c else 0
            new_c = (self.a >> 7) & 1
            self.a = ((self.a << 1) | c) & 0xFF
            self.flag_z = self.flag_n = self.flag_h = False
            self.flag_c = bool(new_c)
            return 4
        if opcode == 0x1F:  # RRA
            c = 0x80 if self.flag_c else 0
            new_c = self.a & 1
            self.a = ((self.a >> 1) | c) & 0xFF
            self.flag_z = self.flag_n = self.flag_h = False
            self.flag_c = bool(new_c)
            return 4
        if opcode == 0xE8:
            offset = self.fetch_signed_byte()
            val = self.sp
            unsigned_offset = offset & 0xFF
            self.flag_z = self.flag_n = False
            self.flag_h = ((val & 0x0F) + (unsigned_offset & 0x0F)) > 0x0F
            self.flag_c = ((val & 0xFF) + (unsigned_offset & 0xFF)) > 0xFF
            self.sp = (val + offset) & 0xFFFF
            return 16
        if opcode == 0xF8:
            offset = self.fetch_signed_byte()
            val = self.sp
            unsigned_offset = offset & 0xFF
            self.flag_z = self.flag_n = False
            self.flag_h = ((val & 0x0F) + (unsigned_offset & 0x0F)) > 0x0F
            self.flag_c = ((val & 0xFF) + (unsigned_offset & 0xFF)) > 0xFF
            self.hl = (val + offset) & 0xFFFF
            return 12

        if opcode not in self._warned_opcodes:
            self._warned_opcodes.add(opcode)
            print(
                f"Warning: unimplemented opcode 0x{opcode:02X} at PC 0x{(self.pc - 1) & 0xFFFF:04X} "
                "(treated as NOP)"
            )
        return 4

    def get_state(self):
        return {
            "a": self.a, "f": self.f, "b": self.b, "c": self.c,
            "d": self.d, "e": self.e, "h": self.h, "l": self.l,
            "sp": self.sp, "pc": self.pc, "ime": self.ime, "halted": self.halted,
            "ei_delay": self.ei_delay, "halt_bug": self.halt_bug, "last_opcode": self.last_opcode,
        }

    def set_state(self, state):
        self.a = state["a"]
        self.f = state["f"]
        self.b = state["b"]
        self.c = state["c"]
        self.d = state["d"]
        self.e = state["e"]
        self.h = state["h"]
        self.l = state["l"]
        self.sp = state["sp"]
        self.pc = state["pc"]
        self.ime = state["ime"]
        self.halted = state["halted"]
        self.ei_delay = state["ei_delay"]
        self.halt_bug = state["halt_bug"]
        self.last_opcode = state.get("last_opcode", 0)


# ---------------------------------------------------------------------------
# PPU - tighter mode timings + cleaner code
# ---------------------------------------------------------------------------
class PPU:
    def __init__(self, mmu):
        self.mmu = mmu
        self.cycles = 0
        self.temp_surf = pygame.Surface((160, 144))
        self.colors = [(224, 248, 208), (136, 192, 112), (52, 104, 86), (8, 24, 32)]
        self.colors_np = np.array(self.colors, dtype=np.uint8)
        self.stat_signal = False
        self.frame_buffer = np.zeros((144, 160, 3), dtype=np.uint8)

        # 2bpp → pixel index lookup
        combined = np.arange(65536, dtype=np.uint16)
        b1 = (combined >> 8).astype(np.uint16)
        b2 = (combined & 0xFF).astype(np.uint16)
        bitpos = np.arange(7, -1, -1, dtype=np.uint16)
        b1_bits = (b1[:, None] >> bitpos[None, :]) & 1
        b2_bits = (b2[:, None] >> bitpos[None, :]) & 1
        self.tile_row_lut = (b1_bits | (b2_bits << 1)).astype(np.uint8)

        self._cgb_bg_table = None
        self._cgb_obj_table = None
        self._cgb_bg_dirty = True
        self._cgb_obj_dirty = True

    def set_palette(self, palette_idx):
        if palette_idx == 1:
            self.colors = [(255, 255, 255), (170, 170, 170), (85, 85, 85), (0, 0, 0)]
        elif palette_idx == 2 and CONFIG.get("custom_palette"):
            self.colors = [tuple(c) for c in CONFIG["custom_palette"]]
        else:
            self.colors = [(224, 248, 208), (136, 192, 112), (52, 104, 86), (8, 24, 32)]
        self.colors_np = np.array(self.colors, dtype=np.uint8)

    def _rebuild_cgb_table(self, palette_ram):
        # RGB555 → RGB888 with optional GBC correction
        raw = np.frombuffer(bytes(palette_ram), dtype=np.uint8).astype(np.uint16)
        low = raw[0::2]
        high = raw[1::2]
        val = (high << 8) | low
        r5 = val & 0x1F
        g5 = (val >> 5) & 0x1F
        b5 = (val >> 10) & 0x1F

        if CONFIG.get("cgb_color_correction", True):
            r = (r5 * 26 + g5 * 4 + b5 * 2) // 32
            g = (g5 * 24 + b5 * 8) // 32
            b = (r5 * 6 + g5 * 4 + b5 * 22) // 32
            r = np.minimum(r, 31) * 255 // 31
            g = np.minimum(g, 31) * 255 // 31
            b = np.minimum(b, 31) * 255 // 31
        else:
            r = r5 * 255 // 31
            g = g5 * 255 // 31
            b = b5 * 255 // 31
        return np.stack([r, g, b], axis=1).astype(np.uint8)

    def get_cgb_bg_table(self):
        if self._cgb_bg_dirty:
            self._cgb_bg_table = self._rebuild_cgb_table(self.mmu.bg_palette_ram)
            self._cgb_bg_dirty = False
        return self._cgb_bg_table

    def get_cgb_obj_table(self):
        if self._cgb_obj_dirty:
            self._cgb_obj_table = self._rebuild_cgb_table(self.mmu.obj_palette_ram)
            self._cgb_obj_dirty = False
        return self._cgb_obj_table

    def mark_palettes_dirty(self):
        self._cgb_bg_dirty = True
        self._cgb_obj_dirty = True

    def update_stat_interrupts(self):
        # Better edge-triggered STAT IRQ
        stat = self.mmu.io[0x41]
        ly = self.mmu.io[0x44]
        lyc = self.mmu.io[0x45]
        coincidence = ly == lyc
        self.mmu.io[0x41] = (self.mmu.io[0x41] & ~0x04) | (0x04 if coincidence else 0)
        mode = stat & 0x03
        hblank_int = bool(stat & 0x08) and mode == 0
        vblank_int = bool(stat & 0x10) and mode == 1
        oam_int = bool(stat & 0x20) and mode == 2
        lyc_int = bool(stat & 0x40) and coincidence
        stat_line = hblank_int or vblank_int or oam_int or lyc_int
        if stat_line and not self.stat_signal:
            self.mmu.io[0x0F] |= 0x02
        self.stat_signal = stat_line

    def step(self, cycles):
        lcdc = self.mmu.io[0x40]
        if not (lcdc & 0x80):
            self.mmu.io[0x44] = 0
            self.cycles = 0
            self.mmu.io[0x41] = (self.mmu.io[0x41] & 0xFC) | 0
            self.stat_signal = False
            return

        self.cycles += cycles
        ly = self.mmu.io[0x44]

        # Closer to real mode lengths
        if ly < 144:
            if self.cycles < 80:
                mode = 2          # OAM search
            elif self.cycles < 252:
                mode = 3          # Pixel transfer (approx)
            else:
                mode = 0          # HBlank
        else:
            mode = 1              # VBlank

        self.mmu.io[0x41] = (self.mmu.io[0x41] & 0xFC) | mode
        self.update_stat_interrupts()

        if self.cycles >= 456:
            self.cycles -= 456
            if ly < 144:
                self.render_scanline(ly)
            ly = (ly + 1) % 154
            self.mmu.io[0x44] = ly
            if ly == 144:
                self.mmu.io[0x0F] |= 0x01

    def render_scanline(self, ly):
        if ly >= 144:
            return
        lcdc = self.mmu.io[0x40]
        if not (lcdc & 0x80):
            return

        cgb = self.mmu.cgb_mode
        vram = self.mmu.vram
        lut = self.tile_row_lut

        bg_color_line = np.zeros(160, dtype=np.uint8)
        bg_priority_line = np.zeros(160, dtype=bool)
        bg_master_priority = (not cgb) or bool(lcdc & 0x01)

        line_colors = np.zeros((160, 3), dtype=np.uint8)
        line_colors[:] = self.colors_np[0]

        cgb_bg_table = self.get_cgb_bg_table() if cgb else None
        bgp = self.mmu.io[0x47]
        dmg_bg_lut = self.colors_np[np.array([(bgp >> (i * 2)) & 3 for i in range(4)])]

        # Background
        draw_bg = (lcdc & 0x01) or cgb
        if draw_bg:
            scx = self.mmu.io[0x43]
            scy = self.mmu.io[0x42]
            bg_map_addr = 0x9C00 if (lcdc & 0x08) else 0x9800
            use_signed_tiles = not (lcdc & 0x10)

            bg_y = (ly + scy) & 255
            tile_row = bg_y // 8
            pixel_row = bg_y % 8
            first_tile_col = scx // 8
            sub_offset = scx % 8
            n_tiles = 21

            strip_colors = np.empty(n_tiles * 8, dtype=np.uint8)
            strip_palnum = np.zeros(n_tiles * 8, dtype=np.uint8) if cgb else None
            strip_priority = np.zeros(n_tiles * 8, dtype=bool) if cgb else None

            for t in range(n_tiles):
                tile_col = (first_tile_col + t) & 31
                map_idx = bg_map_addr + (tile_row * 32) + tile_col
                vram_off = map_idx - 0x8000
                tile_num = vram[vram_off]

                if cgb:
                    attr = vram[0x2000 + vram_off]
                    pal_num = attr & 0x07
                    vbank = (attr >> 3) & 0x01
                    x_flip = bool(attr & 0x20)
                    y_flip = bool(attr & 0x40)
                    tile_priority = bool(attr & 0x80)
                else:
                    pal_num = vbank = 0
                    x_flip = y_flip = tile_priority = False

                row = (7 - pixel_row) if y_flip else pixel_row
                if use_signed_tiles:
                    offset = tile_num - 256 if tile_num >= 128 else tile_num
                    actual_tile = 256 + offset
                else:
                    actual_tile = tile_num

                tile_addr = vbank * 0x2000 + actual_tile * 16 + (row * 2)
                combined = (vram[tile_addr] << 8) | vram[tile_addr + 1]
                row8 = lut[combined]
                if x_flip:
                    row8 = row8[::-1]

                strip_colors[t * 8:t * 8 + 8] = row8
                if cgb:
                    strip_palnum[t * 8:t * 8 + 8] = pal_num
                    strip_priority[t * 8:t * 8 + 8] = tile_priority

            visible = strip_colors[sub_offset:sub_offset + 160]
            bg_color_line[:] = visible
            if cgb:
                visible_pal = strip_palnum[sub_offset:sub_offset + 160]
                bg_priority_line[:] = strip_priority[sub_offset:sub_offset + 160]
                flat_idx = (visible_pal.astype(np.int32) * 4 + visible.astype(np.int32))
                line_colors = cgb_bg_table[flat_idx]
            else:
                line_colors = dmg_bg_lut[visible]

        # Window
        wy = self.mmu.io[0x4A]
        wx = self.mmu.io[0x4B] - 7
        if (lcdc & 0x20) and wy <= ly and wx < 160 and (draw_bg or cgb):
            win_map_addr = 0x9C00 if (lcdc & 0x40) else 0x9800
            use_signed_tiles = not (lcdc & 0x10)
            win_y = ly - wy
            tile_row = win_y // 8
            pixel_row = win_y % 8
            win_width = 160 - max(wx, 0)
            n_tiles = (win_width // 8) + 2

            strip_colors = np.empty(n_tiles * 8, dtype=np.uint8)
            strip_palnum = np.zeros(n_tiles * 8, dtype=np.uint8) if cgb else None
            strip_priority = np.zeros(n_tiles * 8, dtype=bool) if cgb else None

            for t in range(n_tiles):
                map_idx = win_map_addr + (tile_row * 32) + t
                vram_off = map_idx - 0x8000
                tile_num = vram[vram_off]

                if cgb:
                    attr = vram[0x2000 + vram_off]
                    pal_num = attr & 0x07
                    vbank = (attr >> 3) & 0x01
                    x_flip = bool(attr & 0x20)
                    y_flip = bool(attr & 0x40)
                    tile_priority = bool(attr & 0x80)
                else:
                    pal_num = vbank = 0
                    x_flip = y_flip = tile_priority = False

                row = (7 - pixel_row) if y_flip else pixel_row
                if use_signed_tiles:
                    offset = tile_num - 256 if tile_num >= 128 else tile_num
                    actual_tile = 256 + offset
                else:
                    actual_tile = tile_num

                tile_addr = vbank * 0x2000 + actual_tile * 16 + (row * 2)
                combined = (vram[tile_addr] << 8) | vram[tile_addr + 1]
                row8 = lut[combined]
                if x_flip:
                    row8 = row8[::-1]

                strip_colors[t * 8:t * 8 + 8] = row8
                if cgb:
                    strip_palnum[t * 8:t * 8 + 8] = pal_num
                    strip_priority[t * 8:t * 8 + 8] = tile_priority

            dst_start = max(wx, 0)
            src_start = 0 if wx >= 0 else -wx
            span = 160 - dst_start
            visible = strip_colors[src_start:src_start + span]
            bg_color_line[dst_start:160] = visible
            if cgb:
                visible_pal = strip_palnum[src_start:src_start + span]
                bg_priority_line[dst_start:160] = strip_priority[src_start:src_start + span]
                flat_idx = (visible_pal.astype(np.int32) * 4 + visible.astype(np.int32))
                line_colors[dst_start:160] = cgb_bg_table[flat_idx]
            else:
                line_colors[dst_start:160] = dmg_bg_lut[visible]

        self.frame_buffer[ly] = line_colors

        # Sprites
        if lcdc & 0x02:
            sprite_size = 16 if (lcdc & 0x04) else 8
            obp0 = self.mmu.io[0x48]
            obp1 = self.mmu.io[0x49]
            sprites_to_render = []
            oam = self.mmu.oam
            for i in range(40):
                oam_addr = i * 4
                y = oam[oam_addr] - 16
                x = oam[oam_addr + 1] - 8
                tile_num = oam[oam_addr + 2]
                attr = oam[oam_addr + 3]
                if y <= ly < y + sprite_size:
                    sprites_to_render.append((x, y, tile_num, attr, i))
                    if len(sprites_to_render) == 10:
                        break

            if cgb:
                sprites_to_render.sort(key=lambda s: s[4], reverse=True)
            else:
                sprites_to_render.sort(key=lambda s: (s[0], s[4]), reverse=True)

            cgb_obj_table = self.get_cgb_obj_table() if cgb else None
            fb_row = self.frame_buffer[ly]

            for x, y, tile_num, attr, i in sprites_to_render:
                cgb_pal_num = attr & 0x07
                cgb_vbank = (attr >> 3) & 0x01
                palette = obp1 if (attr & 0x10) else obp0
                flip_y = bool(attr & 0x40)
                flip_x = bool(attr & 0x20)
                priority = bool(attr & 0x80)

                line = ly - y
                if flip_y:
                    line = sprite_size - 1 - line

                actual_tile = tile_num
                if sprite_size == 16:
                    actual_tile &= 0xFE
                    if line >= 8:
                        actual_tile |= 1
                        line -= 8

                tile_addr = (cgb_vbank * 0x2000 if cgb else 0) + actual_tile * 16 + (line * 2)
                combined = (vram[tile_addr] << 8) | vram[tile_addr + 1]
                row8 = lut[combined]
                if flip_x:
                    row8 = row8[::-1]

                for px in range(8):
                    pixel_x = x + px
                    if not (0 <= pixel_x < 160):
                        continue
                    color_idx = row8[px]
                    if color_idx == 0:
                        continue

                    if bg_master_priority:
                        bg_wins = (priority or bg_priority_line[pixel_x]) and bg_color_line[pixel_x] != 0
                        if bg_wins:
                            continue

                    if cgb:
                        color = cgb_obj_table[cgb_pal_num * 4 + color_idx]
                    else:
                        mapped_color = (palette >> (int(color_idx) * 2)) & 3
                        color = self.colors_np[mapped_color]
                    fb_row[pixel_x] = color

    def render(self, screen_surface):
        pygame.surfarray.blit_array(self.temp_surf, np.transpose(self.frame_buffer, (1, 0, 2)))
        screen_surface.blit(self.temp_surf, (0, 0))

    def get_state(self):
        return {
            "cycles": self.cycles,
            "stat_signal": self.stat_signal,
            "frame_buffer": self.frame_buffer.copy(),
        }

    def set_state(self, state):
        self.cycles = state["cycles"]
        self.stat_signal = state["stat_signal"]
        self.frame_buffer[:] = state["frame_buffer"]
        self.mark_palettes_dirty()


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self, mmu):
        self.mmu = mmu
        self.div_cycles = 0
        self.tima_cycles = 0

    def step(self, cycles):
        self.div_cycles += cycles
        while self.div_cycles >= 256:
            self.div_cycles -= 256
            self.mmu.io[0x04] = (self.mmu.io[0x04] + 1) & 0xFF

        tac = self.mmu.read_byte(0xFF07)
        if tac & 0x04:
            freqs = [1024, 16, 64, 256]
            limit = freqs[tac & 0x03]
            self.tima_cycles += cycles
            while self.tima_cycles >= limit:
                self.tima_cycles -= limit
                tima = self.mmu.read_byte(0xFF05)
                if tima == 0xFF:
                    self.mmu.io[0x05] = self.mmu.read_byte(0xFF06)
                    self.mmu.io[0x0F] |= 0x04
                else:
                    self.mmu.io[0x05] = (tima + 1) & 0xFF

    def get_state(self):
        return {"div_cycles": self.div_cycles, "tima_cycles": self.tima_cycles}

    def set_state(self, state):
        self.div_cycles = state["div_cycles"]
        self.tima_cycles = state["tima_cycles"]


# ---------------------------------------------------------------------------
# Main emulator
# ---------------------------------------------------------------------------
class SaffyBoy:
    def __init__(self):
        self.mmu = MMU()
        self.cpu = CPU(self.mmu)
        self.ppu = PPU(self.mmu)
        self.timer = Timer(self.mmu)
        self.apu = APU()
        self.mmu.ppu = self.ppu
        self.mmu.apu = self.apu
        self.mmu.timer = self.timer

        self.rom_path = None
        self.rom_title = ""
        self.paused = False
        self.turbo = False
        self.frame_advance_pending = False
        self.show_overlay = CONFIG.get("show_overlay", True)
        self.fullscreen = False
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()

        self.ppu.set_palette(CONFIG.get("dmg_palette", 0))

    def load_rom(self, filepath):
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            self.mmu.load_rom(f.read())
        self.rom_path = filepath
        self.rom_title = self.mmu.rom_title or filepath.stem
        self.cpu.init_registers_post_boot()
        save_path = SAVES_DIR / (filepath.stem + ".sav")
        self.mmu.load_save(str(save_path))
        self.mmu.print_rom_info()
        self.ppu.mark_palettes_dirty()

    def get_full_state(self):
        return {
            "version": 2,
            "rom_hash": self.mmu.rom_hash,
            "rom_title": self.mmu.rom_title,
            "cpu": self.cpu.get_state(),
            "mmu": self.mmu.get_state(),
            "ppu": self.ppu.get_state(),
            "timer": self.timer.get_state(),
            "apu": self.apu.get_state(),
        }

    def set_full_state(self, state):
        if state.get("rom_hash") != self.mmu.rom_hash:
            print("[state] Warning: ROM hash mismatch – state may be for a different ROM")
        self.cpu.set_state(state["cpu"])
        self.mmu.set_state(state["mmu"])
        self.ppu.set_state(state["ppu"])
        self.timer.set_state(state["timer"])
        self.apu.set_state(state["apu"])

    def save_state(self, slot):
        try:
            import pickle
            state = self.get_full_state()
            path = STATES_DIR / f"{self.rom_path.stem}_slot{slot}.state"
            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=4)
            print(f"State saved to slot {slot}")
        except Exception as e:
            print(f"[state] Save failed: {e}")
            traceback.print_exc()

    def load_state(self, slot):
        path = STATES_DIR / f"{self.rom_path.stem}_slot{slot}.state"
        if not path.exists():
            print(f"[state] Slot {slot} empty")
            return
        try:
            import pickle
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.set_full_state(state)
            print(f"State loaded from slot {slot}")
        except Exception as e:
            print(f"[state] Load failed: {e}")
            traceback.print_exc()

    def take_screenshot(self):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SCREENSHOTS_DIR / f"{self.rom_path.stem}_{ts}.png"
            pygame.image.save(self.ppu.temp_surf, str(path))
            print(f"Screenshot: {path}")
        except Exception as e:
            print(f"[screenshot] Failed: {e}")

    def reset(self):
        self.cpu.init_registers_post_boot()
        self.ppu.cycles = 0
        self.timer.div_cycles = 0
        self.timer.tima_cycles = 0
        self.apu.stop_all()
        self.ppu.mark_palettes_dirty()
        print("Reset")

    def run(self):
        pygame.init()
        pygame.joystick.init()
        caption = f"SaffyBoy — {self.rom_title}"
        if self.mmu.cgb_mode:
            caption += " [GBC]"
        pygame.display.set_caption(caption)

        controller = None
        if pygame.joystick.get_count() > 0:
            controller = pygame.joystick.Joystick(0)
            controller.init()
            print(f"Controller: {controller.get_name()} "
                  f"({controller.get_numaxes()} axes, {controller.get_numbuttons()} buttons, "
                  f"{controller.get_numhats()} hats)")

        screen_width = CONFIG.get("window_width", 480)
        screen_height = CONFIG.get("window_height", 432)
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        gb_surface = pygame.Surface((160, 144))

        target_fps = 59.7275
        cycles_per_frame = 70224

        running = True
        autosave_timer = 0.0
        AUTOSAVE_INTERVAL = CONFIG.get("autosave_interval", 5.0)
        font = pygame.font.SysFont("Consolas", 14)

        last_bg_cram = bytes(self.mmu.bg_palette_ram)
        last_obj_cram = bytes(self.mmu.obj_palette_ram)

        while running:
            frame_start = time.perf_counter()

            keys_pressed = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    if not self.fullscreen:
                        screen_width, screen_height = event.size
                        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                elif event.type == pygame.JOYDEVICEADDED:
                    if controller is None and pygame.joystick.get_count() > 0:
                        controller = pygame.joystick.Joystick(0)
                        controller.init()
                        print(f"Controller connected: {controller.get_name()}")
                elif event.type == pygame.JOYDEVICEREMOVED:
                    controller = None
                    print("Controller disconnected")
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == get_key("fullscreen") or k == pygame.K_F11:
                        self.fullscreen = not self.fullscreen
                        if self.fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                    elif k == get_key("pause"):
                        self.paused = not self.paused
                    elif k == get_key("frame_advance") and self.paused:
                        self.frame_advance_pending = True
                    elif k == get_key("screenshot"):
                        self.take_screenshot()
                    elif k == get_key("quick_save"):
                        self.save_state(0)
                    elif k == get_key("quick_load"):
                        self.load_state(0)
                    elif k == get_key("toggle_overlay"):
                        self.show_overlay = not self.show_overlay
                    elif k == get_key("reset"):
                        self.reset()
                    elif k == get_key("show_rom_info"):
                        self.mmu.print_rom_info()
                    elif pygame.K_F1 <= k <= pygame.K_F10:
                        slot = k - pygame.K_F1 + 1
                        mods = pygame.key.get_mods()
                        if mods & pygame.KMOD_SHIFT:
                            self.load_state(slot)
                        else:
                            self.save_state(slot)

            turbo_key = get_key("turbo")
            self.turbo = bool(turbo_key is not None and keys_pressed[turbo_key])
            if controller is not None:
                try:
                    if controller.get_numbuttons() > 4 and controller.get_button(4):
                        self.turbo = True
                except Exception:
                    pass

            if not self.paused or self.frame_advance_pending:
                self.mmu.update_input(keys_pressed, controller)

                speed = 1
                if self.turbo:
                    if CONFIG.get("unlimited_turbo"):
                        speed = 20
                    else:
                        speed = max(1, int(CONFIG.get("turbo_speed", 4)))

                for _ in range(speed):
                    cycles_run = 0
                    while cycles_run < cycles_per_frame:
                        cycles = self.cpu.step()
                        self.ppu.step(cycles)
                        self.timer.step(cycles)
                        cycles_run += cycles
                    self.mmu.tick_rtc(cycles_run)

                    if self.mmu.cgb_mode:
                        cur_bg = bytes(self.mmu.bg_palette_ram)
                        cur_obj = bytes(self.mmu.obj_palette_ram)
                        if cur_bg != last_bg_cram or cur_obj != last_obj_cram:
                            self.ppu.mark_palettes_dirty()
                            last_bg_cram = cur_bg
                            last_obj_cram = cur_obj

                    self.ppu.render(gb_surface)

                self.frame_advance_pending = False

            # Draw
            curr_w, curr_h = screen.get_size()
            if CONFIG.get("integer_scale", True):
                scale = max(1, min(curr_w // 160, curr_h // 144))
            else:
                scale = min(curr_w / 160.0, curr_h / 144.0)
            scaled_w = int(160 * scale)
            scaled_h = int(144 * scale)
            offset_x = (curr_w - scaled_w) // 2
            offset_y = (curr_h - scaled_h) // 2

            scaled_surf = pygame.transform.scale(gb_surface, (scaled_w, scaled_h))
            screen.fill((0, 0, 0))
            screen.blit(scaled_surf, (offset_x, offset_y))

            if self.show_overlay:
                self.frame_count += 1
                now = time.time()
                if now - self.fps_timer >= 1.0:
                    self.fps = self.frame_count / (now - self.fps_timer)
                    self.frame_count = 0
                    self.fps_timer = now

                lines = [
                    f"FPS: {self.fps:.1f}",
                    "PAUSED" if self.paused else "RUNNING",
                    "TURBO" if self.turbo else "",
                    f"{'CGB' if self.mmu.cgb_mode else 'DMG'} | {self.rom_title[:18]}",
                ]
                y = 4
                for line in lines:
                    if line:
                        txt = font.render(line, True, (255, 255, 0))
                        screen.blit(txt, (4, y))
                        y += 16

            pygame.display.flip()

            if not self.turbo:
                elapsed = time.perf_counter() - frame_start
                target = 1.0 / target_fps
                if elapsed < target:
                    time.sleep(target - elapsed)

            autosave_timer += 1.0 / target_fps
            if autosave_timer >= AUTOSAVE_INTERVAL:
                autosave_timer = 0.0
                self.mmu.write_save()

        self.mmu.write_save()
        save_config(CONFIG)
        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("SaffyBoy — Game Boy / Game Boy Color Emulator")
        print("Usage: python saffyboy.py <rom.gb / .gbc>")
        print()
        print("Keyboard (edit saffyboy_config.json to remap):")
        print("  Arrows / X / Z / Space / Enter  – D-pad / A / B / Select / Start")
        print("  Tab                            – Turbo (hold)")
        print("  P                              – Pause")
        print("  N                              – Frame advance (while paused)")
        print("  F5 / F8                        – Quick save / Quick load")
        print("  F1–F10 / Shift+F1–F10          – Save / Load state slots")
        print("  F12 / F11 / O / R / I          – Screenshot / Fullscreen / Overlay / Reset / Info")
        print()
        print("Xbox 360 / XInput controller:")
        print("  D-pad or Left stick            – D-pad")
        print("  A or X                         – A")
        print("  B or Y                         – B")
        print("  Back                           – Select")
        print("  Start                          – Start")
        print("  LB                             – Turbo (hold)")
        sys.exit(1)

    gb = SaffyBoy()
    try:
        gb.load_rom(sys.argv[1])
        gb.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
