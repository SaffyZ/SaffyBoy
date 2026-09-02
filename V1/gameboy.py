import sys
import pygame
import array

class APU:
    def __init__(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2)
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
        self.ch4_last_freq = 0
        self.ch1_last_time = 0
        self.ch2_last_time = 0
        self.ch3_last_time = 0
        self.ch4_last_time = 0

    def trigger_channel1(self, mmu):
        if not self.ch1_channel: return
        low = mmu.read_byte(0xFF13)
        high = mmu.read_byte(0xFF14) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq < 2048:
            freq = 131072 / (2048 - raw_freq)
            now = pygame.time.get_ticks()
            if abs(freq - self.ch1_last_freq) < 10 and (now - self.ch1_last_time) < 100: return
            self.ch1_last_freq = freq
            self.ch1_last_time = now
            vol_reg = mmu.read_byte(0xFF12)
            vol = (vol_reg >> 4) / 15.0 * 0.05
            duty_idx = (mmu.read_byte(0xFF11) >> 6) & 0x03
            duty = [0.125, 0.25, 0.5, 0.75][duty_idx]
            sound = self.generate_square_wave(freq, duty_cycle=duty, volume=vol)
            if sound: self.ch1_channel.play(sound)

    def trigger_channel2(self, mmu):
        if not self.ch2_channel: return
        low = mmu.read_byte(0xFF18)
        high = mmu.read_byte(0xFF19) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq < 2048:
            freq = 131072 / (2048 - raw_freq)
            now = pygame.time.get_ticks()
            if abs(freq - self.ch2_last_freq) < 10 and (now - self.ch2_last_time) < 100: return
            self.ch2_last_freq = freq
            self.ch2_last_time = now
            vol_reg = mmu.read_byte(0xFF17)
            vol = (vol_reg >> 4) / 15.0 * 0.05
            duty_idx = (mmu.read_byte(0xFF16) >> 6) & 0x03
            duty = [0.125, 0.25, 0.5, 0.75][duty_idx]
            sound = self.generate_square_wave(freq, duty_cycle=duty, volume=vol)
            if sound: self.ch2_channel.play(sound)

    def trigger_channel3(self, mmu):
        if not self.ch3_channel: return
        if not (mmu.read_byte(0xFF1A) & 0x80): return
        low = mmu.read_byte(0xFF1D)
        high = mmu.read_byte(0xFF1E) & 0x07
        raw_freq = (high << 8) | low
        if raw_freq < 2048:
            freq = 65536 / (2048 - raw_freq)
            now = pygame.time.get_ticks()
            if abs(freq - self.ch3_last_freq) < 10 and (now - self.ch3_last_time) < 100: return
            self.ch3_last_freq = freq
            self.ch3_last_time = now
            vol_reg = (mmu.read_byte(0xFF1C) >> 5) & 0x03
            vol_scale = [0.0, 1.0, 0.5, 0.25][vol_reg] * 0.05
            wave_ram = bytearray(16)
            for i in range(16):
                wave_ram[i] = mmu.read_byte(0xFF30 + i)
            sound = self.generate_custom_wave(freq, wave_ram, volume=vol_scale)
            if sound: self.ch3_channel.play(sound)

    def trigger_channel4(self, mmu):
        if not self.ch4_channel: return
        nr43 = mmu.read_byte(0xFF22)
        now = pygame.time.get_ticks()
        if (now - self.ch4_last_time) < 100: return
        self.ch4_last_time = now
        vol_reg = mmu.read_byte(0xFF21)
        vol = (vol_reg >> 4) / 15.0 * 0.05
        sound = self.generate_noise_wave(nr43, volume=vol)
        if sound: self.ch4_channel.play(sound)

    def generate_square_wave(self, frequency, duty_cycle=0.5, duration=0.2, volume=0.05, sample_rate=22050):
        if frequency <= 0 or frequency > 8000: return None
        num_samples = int(sample_rate * duration)
        period = sample_rate / frequency
        data = array.array('h')
        for i in range(num_samples):
            val = 16384 if (i % period) < (period * duty_cycle) else -16384
            data.append(int(val * volume))
        try:
            return pygame.mixer.Sound(buffer=data)
        except Exception:
            return None

    def generate_custom_wave(self, frequency, wave_ram, duration=0.2, volume=0.05, sample_rate=22050):
        if frequency <= 0 or frequency > 8000: return None
        num_samples = int(sample_rate * duration)
        period = sample_rate / frequency
        samples = []
        for b in wave_ram:
            samples.append((b >> 4) & 0x0F)
            samples.append(b & 0x0F)
        data = array.array('h')
        for i in range(num_samples):
            sample_idx = int((i % period) * 32 / period) % 32
            val = int((samples[sample_idx] - 8) * 4096)
            data.append(int(val * volume))
        try:
            return pygame.mixer.Sound(buffer=data)
        except Exception:
            return None

    def generate_noise_wave(self, nr43, duration=0.2, volume=0.05, sample_rate=22050):
        shift = (nr43 >> 4) & 0x0F
        counter_step = bool(nr43 & 0x08)
        div = nr43 & 0x07
        divisor = [8, 16, 32, 48, 64, 80, 96, 112][div]
        freq = 524288 / (divisor * (1 << shift))
        if freq <= 0 or freq > 22050: freq = 1000
        num_samples = int(sample_rate * duration)
        period = sample_rate / freq
        lfsr = 0x7FFF
        data = array.array('h')
        val = 16384
        for i in range(num_samples):
            if i % int(max(1, period)) == 0:
                bit = (lfsr ^ (lfsr >> 1)) & 1
                lfsr = (lfsr >> 1) | (bit << 14)
                if counter_step:
                    lfsr = (lfsr & ~0x40) | (bit << 6)
                val = 16384 if (lfsr & 1) else -16384
            data.append(int(val * volume))
        try:
            return pygame.mixer.Sound(buffer=data)
        except Exception:
            return None


class MMU:
    def __init__(self):
        self.rom = bytearray()
        self.cart_type = 0
        self.rom_banks = 2
        self.ram_size = 0
        self.rom_bank = 1
        self.ram_bank = 0
        self.vram = bytearray(0x2000)
        self.eram = bytearray()
        self.wram = bytearray(0x2000)
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

    def load_rom(self, data):
        self.rom = bytearray(data)
        self.cart_type = self.rom[0x0147] if len(self.rom) > 0x0147 else 0
        
        rom_size_code = self.rom[0x0148] if len(self.rom) > 0x0148 else 0
        self.rom_banks = 2 << rom_size_code
        
        ram_code = self.rom[0x0149] if len(self.rom) > 0x0149 else 0
        ram_sizes = {0: 0, 1: 2048, 2: 8192, 3: 32768, 4: 131072, 5: 65536}
        self.ram_size = ram_sizes.get(ram_code, 0)
        self.eram = bytearray(self.ram_size if self.ram_size > 0 else 8192)
        
        title = ""
        for i in range(0x0134, 0x0144):
            if i < len(self.rom) and self.rom[i] != 0:
                title += chr(self.rom[i])
        print(f"Executing ROM: {title.strip()} | Type: 0x{self.cart_type:02X} | Banks: {self.rom_banks} | RAM Size: {self.ram_size} bytes")

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
            return self.vram[addr - 0x8000]
            
        elif 0xA000 <= addr < 0xC000:
            if not self.ram_enabled: return 0xFF
            if self.cart_type in [0x0F, 0x10, 0x11, 0x12, 0x13] and (0x08 <= self.ram_bank <= 0x0C):
                return self.rtc_regs[self.ram_bank - 0x08]
            
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
            
        elif 0xC000 <= addr < 0xE000:
            return self.wram[addr - 0xC000]
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
        elif 0x8000 <= addr < 0xA000:
            offset = addr - 0x8000
            self.vram[offset] = val
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
        elif 0xC000 <= addr < 0xE000:
            self.wram[addr - 0xC000] = val
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
            elif offset == 0x41:
                self.io[0x41] = (val & 0x78) | (self.io[0x41] & 0x07)
            elif offset == 0x44:
                pass
            elif offset == 0x46:
                self.io[0x46] = val
                src = val << 8
                for i in range(160): self.write_byte(0xFE00 + i, self.read_byte(src + i))
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
        if_reg = self.mmu.read_byte(0xFF0F)
        ie_reg = self.mmu.read_byte(0xFFFF)
        fired = if_reg & ie_reg & 0x1F
        if fired:
            self.halted = False
            if self.ime:
                for i in range(5):
                    if fired & (1 << i):
                        self.ime = False
                        self.mmu.write_byte(0xFF0F, if_reg & ~(1 << i))
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
            self.pc = (self.pc + self.fetch_signed_byte()) & 0xFFFF
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
            print(f"Unknown Opcode: 0x{opcode:02X} at PC: 0x{self.pc-1:04X}")
            print(f"Registers: A: 0x{self.a:02X} F: 0x{self.f:02X} B: 0x{self.b:02X} C: 0x{self.c:02X}")
            print(f"D: 0x{self.d:02X} E: 0x{self.e:02X} H: 0x{self.h:02X} L: 0x{self.l:02X}")
            print(f"SP: 0x{self.sp:04X} PC: 0x{self.pc:04X}")
            sys.exit(1)


class PPU:
    def __init__(self, mmu):
        self.mmu = mmu
        self.cycles = 0
        self.temp_surf = pygame.Surface((160, 144))
        self.colors = [(224, 248, 208), (136, 192, 112), (52, 104, 86), (8, 24, 32)]
        self.stat_signal = False

    def update_stat_interrupts(self):
        stat = self.mmu.read_byte(0xFF41)
        ly = self.mmu.read_byte(0xFF44)
        lyc = self.mmu.read_byte(0xFF45)
        
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
        lcdc = self.mmu.read_byte(0xFF40)
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
        lcdc = self.mmu.read_byte(0xFF40)
        if not (lcdc & 0x80): return

        bg_color_line = [0] * 160

        if lcdc & 0x01:
            scx = self.mmu.read_byte(0xFF43)
            scy = self.mmu.read_byte(0xFF42)
            bg_map_addr = 0x9C00 if (lcdc & 0x08) else 0x9800
            use_signed_tiles = not (lcdc & 0x10)
            
            bg_y = (ly + scy) & 255
            tile_row = bg_y // 8
            pixel_row = bg_y % 8
            
            bgp = self.mmu.read_byte(0xFF47)
            
            for x in range(160):
                bg_x = (x + scx) & 255
                tile_col = bg_x // 8
                pixel_col = bg_x % 8
                
                map_idx = bg_map_addr + (tile_row * 32) + tile_col
                tile_num = self.mmu.vram[map_idx - 0x8000]
                
                if use_signed_tiles:
                    offset = tile_num - 256 if tile_num >= 128 else tile_num
                    actual_tile = 256 + offset
                else:
                    actual_tile = tile_num
                    
                tile_addr = actual_tile * 16 + (pixel_row * 2)
                byte1 = self.mmu.vram[tile_addr]
                byte2 = self.mmu.vram[tile_addr + 1]
                
                bit_idx = 7 - pixel_col
                color_idx = (((byte2 >> bit_idx) & 1) << 1) | ((byte1 >> bit_idx) & 1)
                bg_color_line[x] = color_idx
                mapped_color = (bgp >> (color_idx * 2)) & 3
                
                self.temp_surf.set_at((x, ly), self.colors[mapped_color])

        if (lcdc & 0x20) and (self.mmu.read_byte(0xFF4A) <= ly):
            wy = self.mmu.read_byte(0xFF4A)
            wx = self.mmu.read_byte(0xFF4B) - 7
            win_map_addr = 0x9C00 if (lcdc & 0x40) else 0x9800
            use_signed_tiles = not (lcdc & 0x10)
            
            win_y = ly - wy
            tile_row = win_y // 8
            pixel_row = win_y % 8
            
            bgp = self.mmu.read_byte(0xFF47)
            
            for x in range(160):
                if x < wx: continue
                win_x = x - wx
                tile_col = win_x // 8
                pixel_col = win_x % 8
                
                map_idx = win_map_addr + (tile_row * 32) + tile_col
                tile_num = self.mmu.vram[map_idx - 0x8000]
                
                if use_signed_tiles:
                    offset = tile_num - 256 if tile_num >= 128 else tile_num
                    actual_tile = 256 + offset
                else:
                    actual_tile = tile_num
                    
                tile_addr = actual_tile * 16 + (pixel_row * 2)
                byte1 = self.mmu.vram[tile_addr]
                byte2 = self.mmu.vram[tile_addr + 1]
                
                bit_idx = 7 - pixel_col
                color_idx = (((byte2 >> bit_idx) & 1) << 1) | ((byte1 >> bit_idx) & 1)
                bg_color_line[x] = color_idx
                mapped_color = (bgp >> (color_idx * 2)) & 3
                
                self.temp_surf.set_at((x, ly), self.colors[mapped_color])

        if lcdc & 0x02:
            sprite_size = 16 if (lcdc & 0x04) else 8
            obp0 = self.mmu.read_byte(0xFF48)
            obp1 = self.mmu.read_byte(0xFF49)
            
            sprites_to_render = []
            for i in range(40):
                oam_addr = i * 4
                y = self.mmu.oam[oam_addr] - 16
                x = self.mmu.oam[oam_addr + 1] - 8
                tile_num = self.mmu.oam[oam_addr + 2]
                attr = self.mmu.oam[oam_addr + 3]
                
                if y <= ly < y + sprite_size:
                    sprites_to_render.append((x, y, tile_num, attr, i))
                    if len(sprites_to_render) == 10:
                        break

            sprites_to_render.sort(key=lambda s: (s[0], s[4]), reverse=True)
            
            for x, y, tile_num, attr, i in sprites_to_render:
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
                    
                tile_addr = actual_tile * 16 + (line * 2)
                byte1 = self.mmu.vram[tile_addr]
                byte2 = self.mmu.vram[tile_addr + 1]
                
                for px in range(8):
                    pixel_x = x + px
                    if not (0 <= pixel_x < 160): continue
                    
                    bit_idx = px if flip_x else 7 - px
                    color_idx = (((byte2 >> bit_idx) & 1) << 1) | ((byte1 >> bit_idx) & 1)
                    if color_idx == 0: continue
                    if priority and bg_color_line[pixel_x] != 0: continue
                    
                    mapped_color = (palette >> (color_idx * 2)) & 3
                    self.temp_surf.set_at((pixel_x, ly), self.colors[mapped_color])

    def render(self, screen_surface):
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

    def run(self):
        pygame.init()
        screen_width, screen_height = 480, 432
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("PyGB Emulator")
        gb_surface = pygame.Surface((160, 144))
        fullscreen = False

        cpu_step = self.cpu.step
        ppu_step = self.ppu.step
        timer_step = self.timer.step

        target_frame_time = 1.0 / 59.7275
        running = True
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

            self.mmu.update_input()

            cycles_run = 0
            while cycles_run < 70224:
                cycles = cpu_step()
                ppu_step(cycles)
                timer_step(cycles)
                cycles_run += cycles

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

        pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gameboy.py <rom.gb>")
        sys.exit(1)
    gb = GameBoy()
    gb.load_rom(sys.argv[1])
    gb.run()