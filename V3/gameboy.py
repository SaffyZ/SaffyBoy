import sys
import os
import time
import pygame
import array
import numpy as np


class APU:
    """Game Boy APU with real envelope, length counter, and frequency sweep.

    Notes render on-trigger as fully-synthesized (vectorized) buffers that bake
    in the envelope ramp / sweep glide / length cutoff up front, rather than a
    truly sample-clocked real-time engine. A cycle-accurate APU would need to
    tick at ~4.19MHz in Python, which isn't feasible at real speed - this is
    the standard practical compromise and gets very close to correct sound.
    """

    SAMPLE_RATE = 22050

    def __init__(self):
        try:
            #pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=2)
            self.ch1_channel = pygame.mixer.Channel(0)
            self.ch2_channel = pygame.mixer.Channel(1)
            self.ch3_channel = pygame.mixer.Channel(2)
            self.ch4_channel = pygame.mixer.Channel(3)
        except Exception:
            self.ch1_channel = None
            self.ch2_channel = None
            self.ch3_channel = None
            self.ch4_channel = None

        self.ch1_last_freq = 0
        self.ch2_last_freq = 0
        self.ch3_last_freq = 0
        self.ch1_last_time = 0
        self.ch2_last_time = 0
        self.ch3_last_time = 0
        self.ch4_last_time = 0
        self.ch1_end_time = 0
        self.ch2_end_time = 0
        self.ch3_end_time = 0
        self.ch4_end_time = 0
        self._lfsr15_bits = self._build_lfsr(step7=False, length=8192)
        self._lfsr7_bits = self._build_lfsr(step7=True, length=127)

    @staticmethod
    def _build_lfsr(step7, length):
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
                try: ch.stop()
                except Exception: pass
        self.ch1_end_time = self.ch2_end_time = self.ch3_end_time = self.ch4_end_time = 0

    def _compute_duration(self, length_enabled, length_load, max_len, vol_init=None,
                           vol_dir=None, vol_period=None, default=0.5, cap=3.0):
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
        if left_on: stereo[:, 0] = mono_i16.astype(np.float32) * left_vol
        if right_on: stereo[:, 1] = mono_i16.astype(np.float32) * right_vol
        return np.clip(stereo, -32767, 32767).astype(np.int16)

    def _fade_edges(self, wave, fade_ms=4):
        n = len(wave)
        fade_n = min(int(self.SAMPLE_RATE * fade_ms / 1000.0), n // 2)
        if fade_n <= 0: return wave
        ramp = np.linspace(0.0, 1.0, fade_n)
        wave = wave.astype(np.float32)
        wave[:fade_n] *= ramp
        wave[-fade_n:] *= ramp[::-1]
        return wave.astype(np.int16)

    def _square_wave(self, raw_freq_start, duty, vol_init, vol_dir, vol_period,
                      sweep_period, sweep_dir, sweep_shift, duration):
        n = int(self.SAMPLE_RATE * duration)
        if n <= 0: return None
        t = np.arange(n)

        if sweep_period > 0 and sweep_shift > 0:
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
        if freq <= 0: return None
        n = int(self.SAMPLE_RATE * duration)
        if n <= 0: return None
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
        if freq <= 0: freq = 1000.0

        n = int(self.SAMPLE_RATE * duration)
        if n <= 0: return None
        period_samples = max(1.0, self.SAMPLE_RATE / freq)

        lut = self._lfsr7_bits if step7 else self._lfsr15_bits
        idx = (np.arange(n) / period_samples).astype(int) % len(lut)
        square = lut[idx].astype(float)

        vol = self._envelope(n, vol_init, vol_dir, vol_period)
        amplitude = (vol / 15.0) * 0.055
        wave = (square * amplitude * 32767).astype(np.int16)
        return self._fade_edges(wave)

    def trigger_channel1(self, mmu):
        if not self.ch1_channel: return
        if not (mmu.read_byte(0xFF26) & 0x80): return
        low = mmu.read_byte(0xFF13)
        high = mmu.read_byte(0xFF14) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq >= 2048: return

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
        if vol_init == 0 and vol_dir == 0: return

        length_enabled = bool(mmu.read_byte(0xFF14) & 0x40)

        freq = 131072.0 / (2048 - raw_freq)
        now = pygame.time.get_ticks()
        if abs(freq - self.ch1_last_freq) < 3 and (now - self.ch1_last_time) < 30: return
        self.ch1_last_freq, self.ch1_last_time = freq, now

        duration = self._compute_duration(length_enabled, length_load, 64, vol_init, vol_dir, vol_period)
        self.ch1_end_time = now + duration * 1000
        mono = self._square_wave(raw_freq, duty, vol_init, vol_dir, vol_period,
                                  sweep_period, sweep_dir, sweep_shift, duration)
        if mono is None: return
        try:
            self.ch1_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 1).tobytes()))
        except Exception: pass

    def trigger_channel2(self, mmu):
        if not self.ch2_channel: return
        if not (mmu.read_byte(0xFF26) & 0x80): return
        low = mmu.read_byte(0xFF18)
        high = mmu.read_byte(0xFF19) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq >= 2048: return

        nr21 = mmu.read_byte(0xFF16)
        duty = [0.125, 0.25, 0.5, 0.75][(nr21 >> 6) & 0x03]
        length_load = nr21 & 0x3F

        nr22 = mmu.read_byte(0xFF17)
        vol_init = (nr22 >> 4) & 0x0F
        vol_dir = (nr22 >> 3) & 0x01
        vol_period = nr22 & 0x07
        if vol_init == 0 and vol_dir == 0: return

        length_enabled = bool(mmu.read_byte(0xFF19) & 0x40)

        freq = 131072.0 / (2048 - raw_freq)
        now = pygame.time.get_ticks()
        if abs(freq - self.ch2_last_freq) < 3 and (now - self.ch2_last_time) < 30: return
        self.ch2_last_freq, self.ch2_last_time = freq, now

        duration = self._compute_duration(length_enabled, length_load, 64, vol_init, vol_dir, vol_period)
        self.ch2_end_time = now + duration * 1000
        mono = self._square_wave(raw_freq, duty, vol_init, vol_dir, vol_period, 0, 0, 0, duration)
        if mono is None: return
        try:
            self.ch2_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 2).tobytes()))
        except Exception: pass

    def trigger_channel3(self, mmu):
        if not self.ch3_channel: return
        if not (mmu.read_byte(0xFF26) & 0x80): return
        if not (mmu.read_byte(0xFF1A) & 0x80): return
        low = mmu.read_byte(0xFF1D)
        high = mmu.read_byte(0xFF1E) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq >= 2048: return

        length_load = mmu.read_byte(0xFF1B)
        vol_reg = (mmu.read_byte(0xFF1C) >> 5) & 0x03
        length_enabled = bool(mmu.read_byte(0xFF1E) & 0x40)

        freq = 65536.0 / (2048 - raw_freq)
        now = pygame.time.get_ticks()
        if abs(freq - self.ch3_last_freq) < 3 and (now - self.ch3_last_time) < 30: return
        self.ch3_last_freq = freq
        self.ch3_last_time = now

        duration = self._compute_duration(length_enabled, length_load, 256, default=0.5, cap=3.0)
        self.ch3_end_time = now + duration * 1000
        wave_ram = bytearray(16)
        for i in range(16):
            wave_ram[i] = mmu.read_byte(0xFF30 + i)
        mono = self._wave_channel(freq, wave_ram, vol_reg, duration)
        if mono is None: return
        try:
            self.ch3_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 3).tobytes()))
        except Exception: pass

    def trigger_channel4(self, mmu):
        if not self.ch4_channel: return
        if not (mmu.read_byte(0xFF26) & 0x80): return
        nr41 = mmu.read_byte(0xFF20)
        length_load = nr41 & 0x3F

        nr42 = mmu.read_byte(0xFF21)
        vol_init = (nr42 >> 4) & 0x0F
        vol_dir = (nr42 >> 3) & 0x01
        vol_period = nr42 & 0x07
        if vol_init == 0 and vol_dir == 0: return

        nr43 = mmu.read_byte(0xFF22)
        length_enabled = bool(mmu.read_byte(0xFF23) & 0x40)

        now = pygame.time.get_ticks()
        if (now - self.ch4_last_time) < 30: return
        self.ch4_last_time = now

        duration = self._compute_duration(length_enabled, length_load, 64, vol_init, vol_dir, vol_period, default=0.3)
        self.ch4_end_time = now + duration * 1000
        mono = self._noise_channel(nr43, vol_init, vol_dir, vol_period, duration)
        if mono is None: return
        try:
            self.ch4_channel.play(pygame.mixer.Sound(buffer=self._apply_panning(mono, mmu, 4).tobytes()))
        except Exception: pass


class MMU:
    def __init__(self):
        self.rom = bytearray()
        self.cart_type = 0
        self.rom_banks = 2
        self.ram_size = 0
        self.rom_bank = 1
        self.ram_bank = 0
        self.vram = bytearray(0x4000)
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
        self.bg_palette_ram = bytearray(64)
        self.obj_palette_ram = bytearray(64)
        self.bgpi = 0
        self.obpi = 0
        self.bg_palette_used = False
        self.obj_palette_used = False

        self.has_battery = False
        self.save_path = None

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

        self.cgb_mode = len(self.rom) > 0x0143 and (self.rom[0x0143] & 0x80) != 0

        title = ""
        for i in range(0x0134, 0x0144):
            if i < len(self.rom) and self.rom[i] != 0:
                title += chr(self.rom[i])
        print(f"Executing ROM: {title.strip()} | Type: 0x{self.cart_type:02X} | Banks: {self.rom_banks} | "
              f"RAM Size: {self.ram_size} bytes | CGB: {self.cgb_mode} | Battery: {self.has_battery}")

    def load_save(self, path):
        self.save_path = path
        if self.has_battery and os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                n = min(len(data), len(self.eram))
                self.eram[:n] = data[:n]
                print(f"Loaded save: {path} ({n} bytes)")
            except Exception as e:
                print(f"Could not load save: {e}")

    def write_save(self):
        if self.has_battery and self.save_path:
            try:
                with open(self.save_path, 'wb') as f:
                    f.write(bytes(self.eram))
            except Exception as e:
                print(f"Could not write save: {e}")

    def tick_rtc(self, cycles):
        if self.cart_type not in (0x0F, 0x10, 0x11, 0x12, 0x13):
            return
        if self.rtc_regs[4] & 0x40:  # halt flag
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
                            self.rtc_regs[4] |= 0x80  # carry bit
                        self.rtc_regs[3] = day & 0xFF
                        self.rtc_regs[4] = (self.rtc_regs[4] & ~0x01) | ((day >> 8) & 0x01)

    def read_byte(self, addr):
        if 0x0000 <= addr < 0x4000:
            if self.cart_type in [1, 2, 3] and self.mbc1_mode == 1:
                bank = (self.mbc1_rom_bank_high << 5) % self.rom_banks
                rom_addr = (bank * 0x4000) + addr
                return self.rom[rom_addr] if rom_addr < len(self.rom) else 0xFF
            return self.rom[addr] if addr < len(self.rom) else 0xFF
            
        elif 0x4000 <= addr < 0x8000:
            bank = 1
            if self.cart_type in [1, 2, 3]:
                low = self.mbc1_rom_bank_low if self.mbc1_rom_bank_low != 0 else 1
                bank = ((self.mbc1_rom_bank_high << 5) | low) % self.rom_banks
            elif self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13]:
                bank = (self.rom_bank if self.rom_bank != 0 else 1) % self.rom_banks
            elif self.cart_type in [0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E]:
                bank = ((self.mbc5_rom_bank_high << 8) | self.mbc5_rom_bank_low) % self.rom_banks
            
            if len(self.rom) > 0:
                rom_addr = ((bank * 0x4000) + (addr - 0x4000))
                return self.rom[rom_addr] if rom_addr < len(self.rom) else 0xFF
            return 0xFF
            
        elif 0x8000 <= addr < 0xA000:
            return self.vram[(self.vram_bank * 0x2000) + (addr - 0x8000)]
            
        elif 0xA000 <= addr < 0xC000:
            if not self.ram_enabled: return 0xFF
            if self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13] and (0x08 <= self.ram_bank <= 0x0C):
                return self.rtc_latch[self.ram_bank - 0x08]
            
            bank = 0
            if self.cart_type in [1, 2, 3] and self.mbc1_mode == 1:
                bank = self.ram_bank
            elif self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13]:
                bank = self.ram_bank & 0x03
            elif self.cart_type in [0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E]:
                bank = self.ram_bank & 0x0F
                
            if self.ram_size > 0:
                ram_offset = ((bank * 0x2000) + (addr - 0xA000)) % self.ram_size
                return self.eram[ram_offset]
            return 0xFF
            
        elif 0xC000 <= addr < 0xD000:
            return self.wram_banks[0][addr - 0xC000]
        elif 0xD000 <= addr < 0xE000:
            bank = self.wram_bank if self.wram_bank != 0 else 1
            return self.wram_banks[bank][addr - 0xD000]
        elif 0xE000 <= addr < 0xFE00:
            return self.read_byte(addr - 0x2000)
        elif 0xFE00 <= addr < 0xFEA0:
            return self.oam[addr - 0xFE00]
        elif 0xFEA0 <= addr < 0xFF00:
            return 0x00
        elif 0xFF00 <= addr < 0xFF80:
            offset = addr - 0xFF00
            if offset == 0x00:
                val = self.io[0x00] | 0xC0
                joy_val = 0x0F
                if not (val & 0x10): joy_val &= self.joy_directions
                if not (val & 0x20): joy_val &= self.joy_buttons
                return (val & 0xF0) | joy_val
            elif offset == 0x0F:
                return self.io[0x0F] | 0xE0
            elif offset == 0x26:
                val = self.io[0x26] & 0x80
                if self.apu:
                    now = pygame.time.get_ticks()
                    if now < self.apu.ch1_end_time: val |= 0x01
                    if now < self.apu.ch2_end_time: val |= 0x02
                    if now < self.apu.ch3_end_time: val |= 0x04
                    if now < self.apu.ch4_end_time: val |= 0x08
                return val | 0x70
            elif offset == 0x69:
                return self.bg_palette_ram[self.bgpi & 0x3F]
            elif offset == 0x6B:
                return self.obj_palette_ram[self.obpi & 0x3F]
            return self.io[offset]
        elif 0xFF80 <= addr < 0xFFFF:
            return self.hram[addr - 0xFF80]
        elif addr == 0xFFFF:
            return self.ie | 0xE0
        return 0xFF

    def write_byte(self, addr, val):
        val &= 0xFF
        if 0x0000 <= addr < 0x2000:
            self.ram_enabled = (val & 0x0F) == 0x0A
        elif 0x2000 <= addr < 0x4000:
            if self.cart_type in [1, 2, 3]:
                self.mbc1_rom_bank_low = val & 0x1F
            elif self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13]:
                self.rom_bank = val & 0x7F
                if self.rom_bank == 0: self.rom_bank = 1
            elif self.cart_type in [0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E]:
                if addr < 0x3000:
                    self.mbc5_rom_bank_low = val
                else:
                    self.mbc5_rom_bank_high = val & 1
        elif 0x4000 <= addr < 0x6000:
            if self.cart_type in [1, 2, 3]:
                self.mbc1_rom_bank_high = val & 3
                self.ram_bank = val & 3
            elif self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13]:
                self.ram_bank = val
            elif self.cart_type in [0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E]:
                self.ram_bank = val & 0x0F
        elif 0x6000 <= addr < 0x8000:
            if self.cart_type in [1, 2, 3]:
                self.mbc1_mode = val & 1
            elif self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13]:
                if val == 0x00:
                    self._rtc_latch_pending = True
                elif val == 0x01 and self._rtc_latch_pending:
                    self.rtc_latch = list(self.rtc_regs)
                    self._rtc_latch_pending = False
                else:
                    self._rtc_latch_pending = False
        elif 0x8000 <= addr < 0xA000:
            offset = addr - 0x8000
            self.vram[(self.vram_bank * 0x2000) + offset] = val
        elif 0xA000 <= addr < 0xC000:
            if not self.ram_enabled: return
            if self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13] and (0x08 <= self.ram_bank <= 0x0C):
                self.rtc_regs[self.ram_bank - 0x08] = val
                return
            
            bank = 0
            if self.cart_type in [1, 2, 3] and self.mbc1_mode == 1:
                bank = self.ram_bank
            elif self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13]:
                bank = self.ram_bank & 0x03
            elif self.cart_type in [0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E]:
                bank = self.ram_bank & 0x0F
                
            if self.ram_size > 0:
                ram_offset = ((bank * 0x2000) + (addr - 0xA000)) % self.ram_size
                self.eram[ram_offset] = val
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
                if val & 0x80 and self.apu: self.apu.trigger_channel1(self)
            elif offset == 0x19:
                self.io[0x19] = val
                if val & 0x80 and self.apu: self.apu.trigger_channel2(self)
            elif offset == 0x1E:
                self.io[0x1E] = val
                if val & 0x80 and self.apu: self.apu.trigger_channel3(self)
            elif offset == 0x23:
                self.io[0x23] = val
                if val & 0x80 and self.apu: self.apu.trigger_channel4(self)
            elif offset == 0x26:
                self.io[0x26] = (self.io[0x26] & 0x0F) | (val & 0x80)
                if not (val & 0x80) and self.apu:
                    self.apu.stop_all()
            elif offset == 0x41:
                self.io[0x41] = (val & 0x78) | (self.io[0x41] & 0x07)
            elif offset == 0x44:
                pass
            elif offset == 0x46:
                self.io[0x46] = val
                src = val << 8
                for i in range(160): self.write_byte(0xFE00 + i, self.read_byte(src + i))
            elif offset == 0x4F:
                self.vram_bank = val & 0x01
                self.io[0x4F] = 0xFE | self.vram_bank
            elif offset == 0x70:
                bank = val & 0x07
                self.wram_bank = bank if bank != 0 else 1
                self.io[0x70] = val
            elif offset == 0x68:
                self.bgpi = val
                self.io[0x68] = val
            elif offset == 0x69:
                idx = self.bgpi & 0x3F
                self.bg_palette_ram[idx] = val
                self.bg_palette_used = True
                if self.bgpi & 0x80:
                    idx = (idx + 1) & 0x3F
                    self.bgpi = 0x80 | idx
            elif offset == 0x6A:
                self.obpi = val
                self.io[0x6A] = val
            elif offset == 0x6B:
                idx = self.obpi & 0x3F
                self.obj_palette_ram[idx] = val
                self.obj_palette_used = True
                if self.obpi & 0x80:
                    idx = (idx + 1) & 0x3F
                    self.obpi = 0x80 | idx
            elif offset == 0x55:
                self.io[0x55] = val
                src = ((self.io[0x51] << 8) | self.io[0x52]) & 0xFFF0
                dst = 0x8000 + (((self.io[0x53] << 8) | self.io[0x54]) & 0x1FF0)
                length = ((val & 0x7F) + 1) * 16
                for i in range(length):
                    self.write_byte(dst + i, self.read_byte(src + i))
                self.io[0x55] = 0xFF
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

    def update_input(self):
        keys = pygame.key.get_pressed()
        old_dir = self.joy_directions
        old_but = self.joy_buttons

        joy_dir = 0x0F
        if keys[pygame.K_RIGHT]: joy_dir &= ~0x01
        if keys[pygame.K_LEFT]:  joy_dir &= ~0x02
        if keys[pygame.K_UP]:    joy_dir &= ~0x04
        if keys[pygame.K_DOWN]:  joy_dir &= ~0x08
        self.joy_directions = joy_dir

        joy_but = 0x0F
        if keys[pygame.K_x]:      joy_but &= ~0x01
        if keys[pygame.K_z]:      joy_but &= ~0x02
        if keys[pygame.K_SPACE]:  joy_but &= ~0x04
        if keys[pygame.K_RETURN]: joy_but &= ~0x08
        self.joy_buttons = joy_but

        if (joy_dir != old_dir or joy_but != old_but):
            if ((old_dir & ~joy_dir) & 0x0F) or ((old_but & ~joy_but) & 0x0F):
                self.io[0x0F] |= 0x10


class CPU:
    def __init__(self, mmu):
        self.mmu = mmu
        self.a, self.f, self.b, self.c, self.d, self.e, self.h, self.l = 0, 0, 0, 0, 0, 0, 0, 0
        self.sp, self.pc = 0xFFFE, 0x0100
        self.ime = False
        self.halted = False
        self.ei_delay = 0
        self.halt_bug = False
        self._warned_opcodes = set()
        self._warned_cb_opcodes = set()
        self.init_registers_post_boot()

    def init_registers_post_boot(self):
        self.a, self.f, self.b, self.c, self.d, self.e, self.h, self.l = 0x01, 0xB0, 0x00, 0x13, 0x00, 0xD8, 0x01, 0x4D
        self.sp, self.pc = 0xFFFE, 0x0100
        
        self.mmu.write_byte(0xFF05, 0x00)
        self.mmu.write_byte(0xFF06, 0x00)
        self.mmu.write_byte(0xFF07, 0x00)
        
        self.mmu.write_byte(0xFF10, 0x80)
        self.mmu.write_byte(0xFF11, 0xBF)
        self.mmu.write_byte(0xFF12, 0xF3)
        self.mmu.write_byte(0xFF14, 0xBF)
        self.mmu.write_byte(0xFF16, 0x3F)
        self.mmu.write_byte(0xFF17, 0x00)
        self.mmu.write_byte(0xFF19, 0xBF)
        self.mmu.write_byte(0xFF1A, 0x7F)
        self.mmu.write_byte(0xFF1B, 0xFF)
        self.mmu.write_byte(0xFF1C, 0x9F)
        self.mmu.write_byte(0xFF1E, 0xBF)
        self.mmu.write_byte(0xFF20, 0xFF)
        self.mmu.write_byte(0xFF21, 0x00)
        self.mmu.write_byte(0xFF22, 0x00)
        self.mmu.write_byte(0xFF23, 0xBF)
        self.mmu.write_byte(0xFF24, 0x77)
        self.mmu.write_byte(0xFF25, 0xF3)
        self.mmu.write_byte(0xFF26, 0xF1)
        
        self.mmu.write_byte(0xFF40, 0x91)
        self.mmu.write_byte(0xFF42, 0x00)
        self.mmu.write_byte(0xFF43, 0x00)
        self.mmu.write_byte(0xFF45, 0x00)
        self.mmu.write_byte(0xFF47, 0xFC)
        self.mmu.write_byte(0xFF48, 0xFF)
        self.mmu.write_byte(0xFF49, 0xFF)
        self.mmu.write_byte(0xFF4A, 0x00)
        self.mmu.write_byte(0xFF4B, 0x00)
        self.mmu.write_byte(0xFFFF, 0x00)

    @property
    def bc(self): return (self.b << 8) | self.c
    @bc.setter
    def bc(self, val): self.b, self.c = (val >> 8) & 0xFF, val & 0xFF

    @property
    def de(self): return (self.d << 8) | self.e
    @de.setter
    def de(self, val): self.d, self.e = (val >> 8) & 0xFF, val & 0xFF

    @property
    def hl(self): return (self.h << 8) | self.l
    @hl.setter
    def hl(self, val): self.h, self.l = (val >> 8) & 0xFF, val & 0xFF

    @property
    def af(self): return (self.a << 8) | (self.f & 0xF0)
    @af.setter
    def af(self, val): self.a, self.f = (val >> 8) & 0xFF, val & 0xF0

    @property
    def flag_z(self): return bool(self.f & 0x80)
    @flag_z.setter
    def flag_z(self, b): self.f = (self.f & 0x7F) | (0x80 if b else 0)

    @property
    def flag_n(self): return bool(self.f & 0x40)
    @flag_n.setter
    def flag_n(self, b): self.f = (self.f & 0xBF) | (0x40 if b else 0)

    @property
    def flag_h(self): return bool(self.f & 0x20)
    @flag_h.setter
    def flag_h(self, b): self.f = (self.f & 0xDF) | (0x20 if b else 0)

    @property
    def flag_c(self): return bool(self.f & 0x10)
    @flag_c.setter
    def flag_c(self, b): self.f = (self.f & 0xEF) | (0x10 if b else 0)

    def get_reg8(self, reg):
        if reg == 0: return self.b
        if reg == 1: return self.c
        if reg == 2: return self.d
        if reg == 3: return self.e
        if reg == 4: return self.h
        if reg == 5: return self.l
        if reg == 6: return self.mmu.read_byte(self.hl)
        if reg == 7: return self.a

    def set_reg8(self, reg, val):
        val &= 0xFF
        if reg == 0: self.b = val
        elif reg == 1: self.c = val
        elif reg == 2: self.d = val
        elif reg == 3: self.e = val
        elif reg == 4: self.h = val
        elif reg == 5: self.l = val
        elif reg == 6: self.mmu.write_byte(self.hl, val)
        elif reg == 7: self.a = val

    def get_reg16(self, reg, stack=False):
        if reg == 0: return self.bc
        if reg == 1: return self.de
        if reg == 2: return self.hl
        if reg == 3: return self.af if stack else self.sp

    def set_reg16(self, reg, val, stack=False):
        val &= 0xFFFF
        if reg == 0: self.bc = val
        elif reg == 1: self.de = val
        elif reg == 2: self.hl = val
        elif reg == 3:
            if stack: self.af = val
            else: self.sp = val

    def check_condition(self, cond):
        if cond == 0: return not self.flag_z
        if cond == 1: return self.flag_z
        if cond == 2: return not self.flag_c
        if cond == 3: return self.flag_c

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
        if op == 0:
            res = self.a + val
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = False
            self.flag_h = ((self.a & 0x0F) + (val & 0x0F)) > 0x0F
            self.flag_c = res > 0xFF
            self.a = res & 0xFF
        elif op == 1:
            c = 1 if self.flag_c else 0
            res = self.a + val + c
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = False
            self.flag_h = ((self.a & 0x0F) + (val & 0x0F) + c) > 0x0F
            self.flag_c = res > 0xFF
            self.a = res & 0xFF
        elif op == 2:
            res = self.a - val
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = True
            self.flag_h = (self.a & 0x0F) < (val & 0x0F)
            self.flag_c = self.a < val
            self.a = res & 0xFF
        elif op == 3:
            c = 1 if self.flag_c else 0
            res = self.a - val - c
            self.flag_z = (res & 0xFF) == 0
            self.flag_n = True
            self.flag_h = (self.a & 0x0F) < ((val & 0x0F) + c)
            self.flag_c = self.a < (val + c)
            self.a = res & 0xFF
        elif op == 4:
            self.a &= val
            self.flag_z = self.a == 0
            self.flag_n, self.flag_h, self.flag_c = False, True, False
        elif op == 5:
            self.a ^= val
            self.flag_z = self.a == 0
            self.flag_n, self.flag_h, self.flag_c = False, False, False
        elif op == 6:
            self.a |= val
            self.flag_z = self.a == 0
            self.flag_n, self.flag_h, self.flag_c = False, False, False
        elif op == 7:
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
        
        if op_type == 1:
            val = self.get_reg8(reg)
            self.flag_z = (val & (1 << bit)) == 0
            self.flag_n, self.flag_h = False, True
        elif op_type == 2:
            val = self.get_reg8(reg) & ~(1 << bit)
            self.set_reg8(reg, val)
        elif op_type == 3:
            val = self.get_reg8(reg) | (1 << bit)
            self.set_reg8(reg, val)
        elif op_type == 0:
            val = self.get_reg8(reg)
            shift_op = bit
            if shift_op == 0:
                c = (val >> 7) & 1
                res = ((val << 1) | c) & 0xFF
                self.flag_c = bool(c)
            elif shift_op == 1:
                c = val & 1
                res = ((val >> 1) | (c << 7)) & 0xFF
                self.flag_c = bool(c)
            elif shift_op == 2:
                c = 1 if self.flag_c else 0
                res = ((val << 1) | c) & 0xFF
                self.flag_c = bool((val >> 7) & 1)
            elif shift_op == 3:
                c = 0x80 if self.flag_c else 0
                res = ((val >> 1) | c) & 0xFF
                self.flag_c = bool(val & 1)
            elif shift_op == 4:
                res = (val << 1) & 0xFF
                self.flag_c = bool((val >> 7) & 1)
            elif shift_op == 5:
                res = (val >> 1) | (val & 0x80)
                self.flag_c = bool(val & 1)
            elif shift_op == 6:
                res = ((val & 0x0F) << 4) | ((val & 0xF0) >> 4)
                self.flag_c = False
            elif shift_op == 7:
                res = (val >> 1) & 0xFF
                self.flag_c = bool(val & 1)
            
            self.flag_z = res == 0
            self.flag_n, self.flag_h = False, False
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
                        vectors = [0x0040, 0x0048, 0x0050, 0x0058, 0x0060]
                        self.pc = vectors[i]
                        return 20
        return 0

    def step(self):
        if self.ei_delay > 0:
            self.ei_delay -= 1
            if self.ei_delay == 0:
                self.ime = True

        cycles = self.handle_interrupts()
        if cycles > 0: return cycles
        
        if self.halted: return 4
        
        opcode = self.fetch_byte()
        
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
            
        elif 0x80 <= opcode <= 0xBF:
            op, reg = (opcode >> 3) & 0x07, opcode & 0x07
            self.execute_alu(op, self.get_reg8(reg))
            return 8 if reg == 6 else 4
            
        elif (opcode & 0xC7) == 0x06:
            y = (opcode >> 3) & 0x07
            self.set_reg8(y, self.fetch_byte())
            return 12 if y == 6 else 8
            
        elif (opcode & 0xC7) == 0x04:
            y = (opcode >> 3) & 0x07
            orig = self.get_reg8(y)
            val = (orig + 1) & 0xFF
            self.flag_z = val == 0
            self.flag_n, self.flag_h = False, (orig & 0x0F) == 0x0F
            self.set_reg8(y, val)
            return 12 if y == 6 else 4
            
        elif (opcode & 0xC7) == 0x05:
            y = (opcode >> 3) & 0x07
            orig = self.get_reg8(y)
            val = (orig - 1) & 0xFF
            self.flag_z = val == 0
            self.flag_n, self.flag_h = True, (orig & 0x0F) == 0x00
            self.set_reg8(y, val)
            return 12 if y == 6 else 4
            
        elif (opcode & 0xE7) == 0x20:
            cc = (opcode >> 3) & 0x03
            offset = self.fetch_signed_byte()
            if self.check_condition(cc):
                self.pc = (self.pc + offset) & 0xFFFF
                return 12
            return 8
            
        elif (opcode & 0xE7) == 0xC2:
            cc = (opcode >> 3) & 0x03
            addr = self.fetch_word()
            if self.check_condition(cc):
                self.pc = addr & 0xFFFF
                return 16
            return 12
            
        elif (opcode & 0xE7) == 0xC4:
            cc = (opcode >> 3) & 0x03
            addr = self.fetch_word()
            if self.check_condition(cc):
                self.push_word(self.pc)
                self.pc = addr & 0xFFFF
                return 24
            return 12
            
        elif (opcode & 0xE7) == 0xC0:
            cc = (opcode >> 3) & 0x03
            if self.check_condition(cc):
                self.pc = self.pop_word() & 0xFFFF
                return 20
            return 8
            
        elif (opcode & 0xCF) == 0x03:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, (self.get_reg16(q) + 1) & 0xFFFF)
            return 8
            
        elif (opcode & 0xCF) == 0x0B:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, (self.get_reg16(q) - 1) & 0xFFFF)
            return 8
            
        elif (opcode & 0xCF) == 0x09:
            q = (opcode >> 4) & 0x03
            val = self.get_reg16(q)
            hl_val = self.hl
            res = hl_val + val
            self.flag_n = False
            self.flag_h = ((hl_val & 0x0FFF) + (val & 0x0FFF)) > 0x0FFF
            self.flag_c = res > 0xFFFF
            self.hl = res & 0xFFFF
            return 8
            
        elif (opcode & 0xCF) == 0x01:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, self.fetch_word() & 0xFFFF)
            return 12
            
        elif (opcode & 0xCF) == 0xC1:
            q = (opcode >> 4) & 0x03
            self.set_reg16(q, self.pop_word() & 0xFFFF, stack=True)
            return 12
            
        elif (opcode & 0xCF) == 0xC5:
            q = (opcode >> 4) & 0x03
            self.push_word(self.get_reg16(q, stack=True))
            return 16
            
        elif (opcode & 0xC7) == 0xC7:
            self.push_word(self.pc)
            self.pc = (opcode & 0x38) & 0xFFFF
            return 16
            
        elif (opcode & 0xC7) == 0xC6:
            op = (opcode >> 3) & 0x07
            self.execute_alu(op, self.fetch_byte())
            return 8
            
        elif opcode == 0x00: return 4
        elif opcode == 0x10:
            self.fetch_byte()
            return 4
        elif opcode == 0x18:
            offset = self.fetch_signed_byte()
            self.pc = (self.pc + offset) & 0xFFFF
            return 12
        elif opcode == 0xCD:
            addr = self.fetch_word()
            self.push_word(self.pc)
            self.pc = addr & 0xFFFF
            return 24
        elif opcode == 0xC9:
            self.pc = self.pop_word() & 0xFFFF
            return 16
        elif opcode == 0xD9:
            self.pc = self.pop_word() & 0xFFFF
            self.ime = True
            return 16
        elif opcode == 0xC3:
            self.pc = self.fetch_word() & 0xFFFF
            return 16
        elif opcode == 0xE9:
            self.pc = self.hl
            return 4
        elif opcode == 0x22:
            self.mmu.write_byte(self.hl, self.a)
            self.hl = (self.hl + 1) & 0xFFFF
            return 8
        elif opcode == 0x32:
            self.mmu.write_byte(self.hl, self.a)
            self.hl = (self.hl - 1) & 0xFFFF
            return 8
        elif opcode == 0x2A:
            self.a = self.mmu.read_byte(self.hl)
            self.hl = (self.hl + 1) & 0xFFFF
            return 8
        elif opcode == 0x3A:
            self.a = self.mmu.read_byte(self.hl)
            self.hl = (self.hl - 1) & 0xFFFF
            return 8
        elif opcode == 0xE0:
            addr = 0xFF00 + self.fetch_byte()
            self.mmu.write_byte(addr, self.a)
            return 12
        elif opcode == 0xF0:
            addr = 0xFF00 + self.fetch_byte()
            self.a = self.mmu.read_byte(addr)
            return 12
        elif opcode == 0xE2:
            self.mmu.write_byte(0xFF00 + self.c, self.a)
            return 8
        elif opcode == 0xF2:
            self.a = self.mmu.read_byte(0xFF00 + self.c)
            return 8
        elif opcode == 0xEA:
            self.mmu.write_byte(self.fetch_word(), self.a)
            return 16
        elif opcode == 0xFA:
            self.a = self.mmu.read_byte(self.fetch_word())
            return 16
        elif opcode == 0x08:
            self.mmu.write_word(self.fetch_word(), self.sp)
            return 20
        elif opcode == 0xF9:
            self.sp = self.hl
            return 8
        elif opcode == 0x02:
            self.mmu.write_byte(self.bc, self.a)
            return 8
        elif opcode == 0x12:
            self.mmu.write_byte(self.de, self.a)
            return 8
        elif opcode == 0x0A:
            self.a = self.mmu.read_byte(self.bc)
            return 8
        elif opcode == 0x1A:
            self.a = self.mmu.read_byte(self.de)
            return 8
        elif opcode == 0xCB:
            cb_op = self.execute_cb()
            reg = cb_op & 0x07
            op_type = (cb_op >> 6) & 0x03
            if reg == 6:
                if op_type == 1: return 12
                else: return 16
            return 8
        elif opcode == 0xF3:
            self.ime = False
            self.ei_delay = 0
            return 4
        elif opcode == 0xFB:
            self.ei_delay = 2
            return 4
        elif opcode == 0x2F:
            self.a ^= 0xFF
            self.flag_n, self.flag_h = True, True
            return 4
        elif opcode == 0x37:
            self.flag_n, self.flag_h, self.flag_c = False, False, True
            return 4
        elif opcode == 0x3F:
            self.flag_n, self.flag_h = False, False
            self.flag_c = not self.flag_c
            return 4
        elif opcode == 0x27:
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
            self.flag_h, self.flag_c = False, set_carry
            return 4
        elif opcode == 0x07:
            c = (self.a >> 7) & 1
            self.a = ((self.a << 1) | c) & 0xFF
            self.flag_z, self.flag_n, self.flag_h, self.flag_c = False, False, False, bool(c)
            return 4
        elif opcode == 0x0F:
            c = self.a & 1
            self.a = ((self.a >> 1) | (c << 7)) & 0xFF
            self.flag_z, self.flag_n, self.flag_h, self.flag_c = False, False, False, bool(c)
            return 4
        elif opcode == 0x17:
            c = 1 if self.flag_c else 0
            new_c = (self.a >> 7) & 1
            self.a = ((self.a << 1) | c) & 0xFF
            self.flag_z, self.flag_n, self.flag_h, self.flag_c = False, False, False, bool(new_c)
            return 4
        elif opcode == 0x1F:
            c = 0x80 if self.flag_c else 0
            new_c = self.a & 1
            self.a = ((self.a >> 1) | c) & 0xFF
            self.flag_z, self.flag_n, self.flag_h, self.flag_c = False, False, False, bool(new_c)
            return 4
        elif opcode == 0xE8:
            offset = self.fetch_signed_byte()
            val = self.sp
            unsigned_offset = offset & 0xFF
            self.flag_z, self.flag_n = False, False
            self.flag_h = ((val & 0x0F) + (unsigned_offset & 0x0F)) > 0x0F
            self.flag_c = ((val & 0xFF) + (unsigned_offset & 0xFF)) > 0xFF
            self.sp = (val + offset) & 0xFFFF
            return 16
        elif opcode == 0xF8:
            offset = self.fetch_signed_byte()
            val = self.sp
            unsigned_offset = offset & 0xFF
            self.flag_z, self.flag_n = False, False
            self.flag_h = ((val & 0x0F) + (unsigned_offset & 0x0F)) > 0x0F
            self.flag_c = ((val & 0xFF) + (unsigned_offset & 0xFF)) > 0xFF
            self.hl = (val + offset) & 0xFFFF
            return 12
        else:
            if opcode not in self._warned_opcodes:
                self._warned_opcodes.add(opcode)
                print(f"Warning: unimplemented opcode 0x{opcode:02X} at PC: 0x{self.pc - 1:04X} "
                      f"(treating as NOP; further occurrences of this opcode won't be logged)")
            return 4


class PPU:
    def __init__(self, mmu):
        self.mmu = mmu
        self.cycles = 0
        self.temp_surf = pygame.Surface((160, 144))
        self.colors = [(224, 248, 208), (136, 192, 112), (52, 104, 86), (8, 24, 32)]
        self.colors_np = np.array(self.colors, dtype=np.uint8)
        self.stat_signal = False
        self.frame_buffer = np.zeros((144, 160, 3), dtype=np.uint8)
        combined = np.arange(65536, dtype=np.uint16)
        b1 = (combined >> 8).astype(np.uint16)
        b2 = (combined & 0xFF).astype(np.uint16)
        bitpos = np.arange(7, -1, -1, dtype=np.uint16)
        b1_bits = (b1[:, None] >> bitpos[None, :]) & 1
        b2_bits = (b2[:, None] >> bitpos[None, :]) & 1
        self.tile_row_lut = (b1_bits | (b2_bits << 1)).astype(np.uint8)

    def cgb_color(self, palette_ram, palette_num, color_idx):
        base = palette_num * 8 + color_idx * 2
        low = palette_ram[base]
        high = palette_ram[base + 1]
        val = (high << 8) | low
        r = val & 0x1F
        g = (val >> 5) & 0x1F
        b = (val >> 10) & 0x1F
        return (r * 255 // 31, g * 255 // 31, b * 255 // 31)

    def _cgb_palette_table(self, palette_ram):
        raw = np.frombuffer(bytes(palette_ram), dtype=np.uint8).astype(np.uint16)
        low = raw[0::2]
        high = raw[1::2]
        val = (high << 8) | low
        r = (val & 0x1F) * 255 // 31
        g = ((val >> 5) & 0x1F) * 255 // 31
        b = ((val >> 10) & 0x1F) * 255 // 31
        return np.stack([r, g, b], axis=1).astype(np.uint8)

    def update_stat_interrupts(self):
        stat = self.mmu.io[0x41]
        ly = self.mmu.io[0x44]
        lyc = self.mmu.io[0x45]
        
        coincidence = (ly == lyc)
        if coincidence:
            stat |= 0x04
        else:
            stat &= ~0x04
            
        self.mmu.io[0x41] = (self.mmu.io[0x41] & ~0x04) | (0x04 if coincidence else 0)
        
        mode = stat & 0x03
        hblank_int = bool(stat & 0x08) and (mode == 0)
        vblank_int = bool(stat & 0x10) and (mode == 1)
        oam_int    = bool(stat & 0x20) and (mode == 2)
        lyc_int    = bool(stat & 0x40) and coincidence
        
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
        
        if ly < 144:
            if self.cycles < 80: mode = 2
            elif self.cycles < 252: mode = 3
            else: mode = 0
        else:
            mode = 1

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
        if ly >= 144: return
        lcdc = self.mmu.io[0x40]
        if not (lcdc & 0x80): return

        cgb = self.mmu.cgb_mode
        cgb_bg = cgb and self.mmu.bg_palette_used
        cgb_obj = cgb and self.mmu.obj_palette_used
        vram = self.mmu.vram
        lut = self.tile_row_lut
        bg_color_line = np.zeros(160, dtype=np.uint8)
        bg_priority_line = np.zeros(160, dtype=bool)
        bg_master_priority = (not cgb_bg) or bool(lcdc & 0x01)

        line_colors = np.zeros((160, 3), dtype=np.uint8)
        line_colors[:] = self.colors_np[0]

        cgb_bg_table = self._cgb_palette_table(self.mmu.bg_palette_ram) if cgb_bg else None
        bgp = self.mmu.io[0x47]
        dmg_bg_lut = self.colors_np[np.array([(bgp >> (i * 2)) & 3 for i in range(4)])]

        if lcdc & 0x01 or cgb_bg:
            scx = self.mmu.io[0x43]
            scy = self.mmu.io[0x42]
            bg_map_addr = 0x9C00 if (lcdc & 0x08) else 0x9800
            use_signed_tiles = not (lcdc & 0x10)

            bg_y = (ly + scy) & 255
            tile_row = bg_y // 8
            pixel_row = bg_y % 8

            first_tile_col = scx // 8
            sub_offset = scx % 8
            n_tiles = 21  # enough to cover 160px + partial edge tiles

            strip_colors = np.empty(n_tiles * 8, dtype=np.uint8)
            strip_palnum = np.zeros(n_tiles * 8, dtype=np.uint8) if cgb_bg else None
            strip_priority = np.zeros(n_tiles * 8, dtype=bool) if cgb_bg else None

            for t in range(n_tiles):
                tile_col = (first_tile_col + t) & 31
                map_idx = bg_map_addr + (tile_row * 32) + tile_col
                vram_off = map_idx - 0x8000
                tile_num = vram[vram_off]

                if cgb_bg:
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
                if cgb_bg:
                    strip_palnum[t * 8:t * 8 + 8] = pal_num
                    strip_priority[t * 8:t * 8 + 8] = tile_priority

            visible = strip_colors[sub_offset:sub_offset + 160]
            bg_color_line[:] = visible
            if cgb_bg:
                visible_pal = strip_palnum[sub_offset:sub_offset + 160]
                bg_priority_line[:] = strip_priority[sub_offset:sub_offset + 160]
                flat_idx = (visible_pal.astype(np.int32) * 4 + visible.astype(np.int32))
                line_colors = cgb_bg_table[flat_idx]
            else:
                line_colors = dmg_bg_lut[visible]

        wy = self.mmu.io[0x4A]
        wx = self.mmu.io[0x4B] - 7
        if (lcdc & 0x20) and wy <= ly and wx < 160:
            win_map_addr = 0x9C00 if (lcdc & 0x40) else 0x9800
            use_signed_tiles = not (lcdc & 0x10)

            win_y = ly - wy
            tile_row = win_y // 8
            pixel_row = win_y % 8

            win_width = 160 - max(wx, 0)
            n_tiles = (win_width // 8) + 2

            strip_colors = np.empty(n_tiles * 8, dtype=np.uint8)
            strip_palnum = np.zeros(n_tiles * 8, dtype=np.uint8) if cgb_bg else None
            strip_priority = np.zeros(n_tiles * 8, dtype=bool) if cgb_bg else None

            for t in range(n_tiles):
                map_idx = win_map_addr + (tile_row * 32) + t
                vram_off = map_idx - 0x8000
                tile_num = vram[vram_off]

                if cgb_bg:
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
                if cgb_bg:
                    strip_palnum[t * 8:t * 8 + 8] = pal_num
                    strip_priority[t * 8:t * 8 + 8] = tile_priority

            dst_start = max(wx, 0)
            src_start = 0 if wx >= 0 else -wx
            span = 160 - dst_start
            visible = strip_colors[src_start:src_start + span]
            bg_color_line[dst_start:160] = visible
            if cgb_bg:
                visible_pal = strip_palnum[src_start:src_start + span]
                bg_priority_line[dst_start:160] = strip_priority[src_start:src_start + span]
                flat_idx = (visible_pal.astype(np.int32) * 4 + visible.astype(np.int32))
                line_colors[dst_start:160] = cgb_bg_table[flat_idx]
            else:
                line_colors[dst_start:160] = dmg_bg_lut[visible]

        self.frame_buffer[ly] = line_colors

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

            if cgb_obj:
                sprites_to_render.sort(key=lambda s: s[4], reverse=True)
            else:
                sprites_to_render.sort(key=lambda s: (s[0], s[4]), reverse=True)

            cgb_obj_table = self._cgb_palette_table(self.mmu.obj_palette_ram) if cgb_obj else None
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

                tile_addr = (cgb_vbank * 0x2000 if cgb_obj else 0) + actual_tile * 16 + (line * 2)
                combined = (vram[tile_addr] << 8) | vram[tile_addr + 1]
                row8 = lut[combined]
                if flip_x:
                    row8 = row8[::-1]

                for px in range(8):
                    pixel_x = x + px
                    if not (0 <= pixel_x < 160): continue

                    color_idx = row8[px]
                    if color_idx == 0: continue

                    if bg_master_priority:
                        bg_wins = (priority or bg_priority_line[pixel_x]) and bg_color_line[pixel_x] != 0
                        if bg_wins: continue

                    if cgb_obj:
                        color = cgb_obj_table[cgb_pal_num * 4 + color_idx]
                    else:
                        mapped_color = (palette >> (int(color_idx) * 2)) & 3
                        color = self.colors_np[mapped_color]
                    fb_row[pixel_x] = color

    def render(self, screen_surface):
        pygame.surfarray.blit_array(self.temp_surf, np.transpose(self.frame_buffer, (1, 0, 2)))
        screen_surface.blit(self.temp_surf, (0, 0))


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


class GameBoy:
    def __init__(self):
        self.mmu = MMU()
        self.cpu = CPU(self.mmu)
        self.ppu = PPU(self.mmu)
        self.timer = Timer(self.mmu)
        self.apu = APU()
        self.mmu.ppu = self.ppu
        self.mmu.apu = self.apu
        self.mmu.timer = self.timer

    def load_rom(self, filepath):
        with open(filepath, 'rb') as f:
            self.mmu.load_rom(f.read())
        save_path = os.path.splitext(filepath)[0] + '.sav'
        self.mmu.load_save(save_path)
        self.rom_title = filepath

    def run(self):
        pygame.init()
        screen_width, screen_height = 480, 432
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption(f"PyGB Emulator - {os.path.basename(self.rom_title)}"
                                    f"{' [GBC]' if self.mmu.cgb_mode else ''}")
        gb_surface = pygame.Surface((160, 144))
        fullscreen = False
        paused = False

        cpu_step = self.cpu.step
        ppu_step = self.ppu.step
        timer_step = self.timer.step

        target_frame_time = 1.0 / 59.7275
        running = True
        autosave_timer = 0.0
        AUTOSAVE_INTERVAL = 5.0

        while running:
            start_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    if not fullscreen:
                        screen_width, screen_height = event.size
                        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f or event.key == pygame.K_F11:
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_F5:
                        self.mmu.write_save()
                        print("Saved.")

            if not paused:
                self.mmu.update_input()

                cycles_run = 0
                while cycles_run < 70224:
                    cycles = cpu_step()
                    ppu_step(cycles)
                    timer_step(cycles)
                    cycles_run += cycles

                self.mmu.tick_rtc(cycles_run)
                self.ppu.render(gb_surface)

            curr_w, curr_h = screen.get_size()
            scale = min(curr_w / 160.0, curr_h / 144.0)
            scaled_w = int(160 * scale)
            scaled_h = int(144 * scale)
            offset_x = (curr_w - scaled_w) // 2
            offset_y = (curr_h - scaled_h) // 2

            scaled_surf = pygame.transform.scale(gb_surface, (scaled_w, scaled_h))
            screen.fill((0, 0, 0))
            screen.blit(scaled_surf, (offset_x, offset_y))
            pygame.display.flip()
            
            elapsed = (pygame.time.get_ticks() - start_time) / 1000.0
            remaining = target_frame_time - elapsed
            if remaining > 0:
                pygame.time.delay(int(remaining * 1000))

            autosave_timer += max(elapsed, target_frame_time)
            if autosave_timer >= AUTOSAVE_INTERVAL:
                autosave_timer = 0.0
                self.mmu.write_save()

        self.mmu.write_save()
        pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gameboy.py <rom.gb>")
        sys.exit(1)
    gb = GameBoy()
    gb.load_rom(sys.argv[1])
    gb.run()