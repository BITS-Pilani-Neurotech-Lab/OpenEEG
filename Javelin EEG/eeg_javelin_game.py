import pygame
import math
import sys
import serial
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, iirnotch
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

pygame.init()

SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 810
GROUND_LEVEL = SCREEN_HEIGHT - 50
GRAVITY = 9.81
SCALE = 10
SPEED_MULTIPLIER = 5.0
DEBUG_MODE = True

WHITE = (255, 255, 255)
BLUE = (135, 206, 235)
GREEN = (34, 139, 34)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 100, 0)
SILVER = (192, 192, 192)

@dataclass
class EEGConfig:
    port: str = "COM3"
    baud_rate: int = 115200
    sampling_rate: int = 250
    buffer_size: int = 1000
    lowpass_cutoff_frequency: float = 50.0
    notch_frequency: float = 60.0
    filter_order: int = 4
    Q_fac: int = 30
    use_csv: bool = False  # Set True to use CSV mode, False for Serial mode
    csv_file: str = "eeg_data.csv"

class EEGMonitor:
    def __init__(self, config: EEGConfig = None):
        self.config = config if config else EEGConfig()
        self.data_buffer = np.zeros(self.config.buffer_size)

        # CSV mode variables
        self.csv_data = None
        self.csv_index = 0
        self.use_csv_mode = self.config.use_csv  # Boolean, set at init only

        # Serial connection setup
        if not self.use_csv_mode:
            try:
                self.serial = serial.Serial(self.config.port, self.config.baud_rate)
                self.serial.reset_input_buffer()
                self.serial_connected = True
            except serial.SerialException:
                print(f"Warning: Could not connect to serial port {self.config.port}")
                self.serial = None
                self.serial_connected = False
        else:
            self.serial = None
            self.serial_connected = False
            # Always use synthetic data instead of loading from CSV
            self.csv_data = self.generate_synthetic_data()

        self.ratio = []
        self.ratio_history = []
        self.debug_ratio_history = []
        self.running = True
        self.calibration_active = False
        self.fft_signal = np.zeros(self.config.buffer_size // 2 + 1)
        self.freq = np.fft.rfftfreq(self.config.buffer_size, d=1/self.config.sampling_rate)
        self.eeg_thread = threading.Thread(target=self.monitor_eeg, daemon=True)
        self.eeg_thread.start()
    
    def apply_notch_filter(self, signal):
        try:
            b, a = iirnotch(self.config.notch_frequency, self.config.Q_fac, self.config.sampling_rate)
            filtered_signal = filtfilt(b, a, signal)
            return filtered_signal
        except:
            return signal
    
    def load_csv_data(self):
        """Always generate synthetic EEG data instead of loading from CSV"""
        self.csv_data = self.generate_synthetic_data()
        print(f"Generated {len(self.csv_data)} synthetic EEG data points.")
    
    def generate_synthetic_data(self):
        """Generate synthetic EEG data for testing"""
        n_samples = 10000
        t = np.linspace(0, n_samples / self.config.sampling_rate, n_samples)
        # Create synthetic EEG-like signal with multiple frequency components
        signal = (
            100 * np.sin(2 * np.pi * 10 * t) +  # Alpha waves
            50 * np.sin(2 * np.pi * 20 * t) +   # Beta waves
            30 * np.random.normal(0, 1, len(t))  # Noise
        )
        return signal + 512  # Add offset to make values positive
    
    def get_csv_value(self):
        """Get next value from CSV data, looping over if at end"""
        if self.csv_data is None or len(self.csv_data) == 0:
            return 512  # Default value
        if self.csv_index >= len(self.csv_data):
            self.csv_index = 0  # Loop back to start
        value = self.csv_data[self.csv_index]
        self.csv_index += 1
        return int(value)
    
    def set_csv_mode(self, use_csv: bool):
        """Set CSV mode to True (CSV) or False (Serial)"""
        if self.use_csv_mode == use_csv:
            return
        self.use_csv_mode = use_csv
        if self.use_csv_mode:
            if self.serial_connected and self.serial:
                self.serial.close()
                self.serial_connected = False
            self.load_csv_data()
            self.csv_index = 0
            print("Switched to CSV mode")
        else:
            try:
                self.serial = serial.Serial(self.config.port, self.config.baud_rate)
                self.serial.reset_input_buffer()
                self.serial_connected = True
                print("Switched to Serial mode")
            except serial.SerialException:
                print(f"Warning: Could not connect to serial port {self.config.port}")
                self.serial = None
                self.serial_connected = False
    
    def monitor_eeg(self):
        while self.running:
            new_value = None
            
            # Get data based on current mode
            if self.use_csv_mode:
                # CSV mode - get value from loaded data
                new_value = self.get_csv_value()
            elif self.serial_connected and self.serial and self.serial.in_waiting >= 2:
                # Serial mode - read from serial port
                try:
                    new_value = int(self.serial.readline())
                except Exception as e:
                    print(f"Serial read error: {e}")
                    new_value = None
            
            # Process the new value if we have one
            if new_value is not None:
                try:
                    self.data_buffer = np.roll(self.data_buffer, -1)
                    self.data_buffer[-1] = new_value
                    filtered_signal = self.apply_notch_filter(self.data_buffer)
                    N = len(filtered_signal)
                    hamming_window = np.hamming(N)
                    self.fft_signal = np.abs(np.fft.rfft(filtered_signal * hamming_window))
                    current_ratio = np.mean(self.fft_signal[15:25]) / np.mean(self.fft_signal[9:15])
                    self.ratio.append(current_ratio)
                    if DEBUG_MODE:
                        self.debug_ratio_history.append(current_ratio)
                        if len(self.debug_ratio_history) > 200:
                            self.debug_ratio_history = self.debug_ratio_history[-200:]
                    if self.calibration_active:
                        current_time = time.time()
                        self.ratio_history.append((current_time, current_ratio))
                        cutoff_time = current_time - 15
                        self.ratio_history = [(t, r) for t, r in self.ratio_history if t > cutoff_time]
                except Exception as e:
                    print(f"EEG processing error: {e}")
            else:
                # Fallback to debug mode simulation if no data source available
                if DEBUG_MODE and not self.serial_connected and not self.use_csv_mode:
                    current_ratio = 1.0 + 0.5 * np.sin(time.time() * 2) + np.random.normal(0, 0.1)
                    current_ratio = max(0.5, min(3.0, current_ratio))
                    self.debug_ratio_history.append(current_ratio)
                    if len(self.debug_ratio_history) > 200:
                        self.debug_ratio_history = self.debug_ratio_history[-200:]
            
            # Sleep to maintain proper sampling rate
            time.sleep(1.0 / self.config.sampling_rate)
    
    def get_max_ratio_15s(self):
        if not self.ratio_history:
            return 1.0
        ratios = [r for t, r in self.ratio_history]
        if not ratios:
            return 1.0
        sorted_ratios = sorted(ratios, reverse=True)
        top_5_ratios = sorted_ratios[:5]
        return sum(top_5_ratios) / len(top_5_ratios)
    
    def get_current_ratio(self):
        if not self.ratio_history:
            return 1.0
        return self.ratio_history[-1][1] if self.ratio_history else 1.0
    
    def set_calibration_active(self, active):
        self.calibration_active = active
    
    def clear_history(self):
        self.ratio_history = []
    
    def stop(self):
        self.running = False
        if self.serial_connected and self.serial:
            self.serial.close()

class Javelin:
    def __init__(self, x, y, eeg_monitor):
        self.initial_x = x
        self.initial_y = y
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.angle = 45
        self.base_force = 75
        self.eeg_force_multiplier = 1.0
        self.eeg_monitor = eeg_monitor
        self.trail = []
        self.flying = False
        self.landed = False
        self.distance = 0
        self.calibration_start_time = None
        self.calibrating = False
        self.calibration_complete_time = None
        self.freeze_meter = False
        self.final_max_ratio = 0.0
        
    def start_calibration(self):
        self.calibration_start_time = time.time()
        self.calibrating = True
        self.eeg_monitor.set_calibration_active(True)
    
    def get_calibration_time_remaining(self):
        if not self.calibrating or not self.calibration_start_time:
            return 0
        elapsed = time.time() - self.calibration_start_time
        remaining = max(0, 15 - elapsed)
        if remaining <= 0:
            self.calibrating = False
            self.eeg_monitor.set_calibration_active(False)
            max_ratio = self.eeg_monitor.get_max_ratio_15s()
            self.eeg_force_multiplier = min(2.0, max(0.5, max_ratio))
            self.final_max_ratio = max_ratio
            self.calibration_complete_time = time.time()
            self.freeze_meter = True
        return remaining
    
    def check_auto_throw(self):
        if self.freeze_meter and self.calibration_complete_time:
            elapsed = time.time() - self.calibration_complete_time
            if elapsed >= 2.0:
                self.freeze_meter = False
                self.calibration_complete_time = None
                if not self.flying and not self.landed:
                    self.throw()
                return True
        return False
    
    def get_effective_force(self):
        return self.base_force * self.eeg_force_multiplier
        
    def throw(self):
        if not self.flying and not self.landed and not self.calibrating:
            angle_rad = math.radians(self.angle)
            effective_force = self.get_effective_force()
            self.vx = effective_force * math.cos(angle_rad)
            self.vy = -effective_force * math.sin(angle_rad)
            self.flying = True
            self.trail = [(self.x, self.y)]
    
    def update(self, dt):
        if self.flying and not self.landed:
            dt *= SPEED_MULTIPLIER
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.vy += GRAVITY * dt
            if len(self.trail) == 0 or len(self.trail) % 3 == 0:
                self.trail.append((self.x, self.y))
            if self.y >= GROUND_LEVEL - 10:
                self.y = GROUND_LEVEL - 10
                self.flying = False
                self.landed = True
                self.distance = (self.x - self.initial_x) / SCALE
    
    def reset(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.vx = 0
        self.vy = 0
        self.trail = []
        self.flying = False
        self.landed = False
        self.distance = 0
        self.eeg_force_multiplier = 1.0
        self.calibrating = False
        self.calibration_start_time = None
        self.calibration_complete_time = None
        self.freeze_meter = False
        self.final_max_ratio = 0.0
        self.eeg_monitor.set_calibration_active(False)
        self.eeg_monitor.clear_history()
    
    def draw(self, screen):
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                alpha = min(255, int(255 * (i / len(self.trail))))
                trail_color = (255, alpha // 2, 0)
                if i < len(self.trail) - 2:
                    pygame.draw.line(screen, trail_color, self.trail[i], self.trail[i + 1], 3)
        if not self.landed:
            if self.vx != 0 or self.vy != 0:
                javelin_angle = math.atan2(self.vy, self.vx)
            else:
                javelin_angle = math.radians(-self.angle)
        else:
            javelin_angle = 0
        length = 35
        tip_length = 8
        end_x = self.x + length * math.cos(javelin_angle)
        end_y = self.y + length * math.sin(javelin_angle)
        pygame.draw.line(screen, BLACK, (self.x, self.y), (end_x, end_y), 5)
        tip_x = end_x + tip_length * math.cos(javelin_angle)
        tip_y = end_y + tip_length * math.sin(javelin_angle)
        tip_points = [
            (tip_x, tip_y),
            (end_x + 3 * math.cos(javelin_angle + math.pi/2), end_y + 3 * math.sin(javelin_angle + math.pi/2)),
            (end_x + 3 * math.cos(javelin_angle - math.pi/2), end_y + 3 * math.sin(javelin_angle - math.pi/2))
        ]
        pygame.draw.polygon(screen, SILVER, tip_points)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("EEG-Controlled Javelin Throw")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.eeg_monitor = EEGMonitor()
        self.javelin = Javelin(100, GROUND_LEVEL - 50, self.eeg_monitor)
        self.debug_graph_surface = None
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.eeg_monitor.stop()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.javelin.flying and not self.javelin.landed and not self.javelin.calibrating:
                        self.javelin.throw()
                elif event.key == pygame.K_r:
                    self.javelin.reset()
                elif event.key == pygame.K_c:
                    if not self.javelin.flying and not self.javelin.landed:
                        self.javelin.start_calibration()
                elif event.key == pygame.K_d:
                    global DEBUG_MODE
                    DEBUG_MODE = not DEBUG_MODE
                    if not DEBUG_MODE:
                        self.debug_graph_surface = None
                # Remove T toggle for mode
        return True
    
    def update(self, dt):
        self.javelin.update(dt)
        self.javelin.check_auto_throw()
    
    def draw_background(self):
        for y in range(GROUND_LEVEL):
            color_ratio = y / GROUND_LEVEL
            r = int(135 + (255 - 135) * color_ratio)
            g = int(206 + (255 - 206) * color_ratio)
            b = int(235 + (255 - 235) * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
        self.draw_clouds()
        pygame.draw.rect(self.screen, GREEN, (0, GROUND_LEVEL, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_LEVEL))
        for i in range(0, SCREEN_WIDTH, 5):
            grass_height = GROUND_LEVEL + 5 + (i % 3)
            pygame.draw.line(self.screen, DARK_GREEN, (i, GROUND_LEVEL), (i, grass_height), 1)
        track_y = GROUND_LEVEL - 5
        pygame.draw.rect(self.screen, (180, 100, 50), (0, track_y, 200, 5))
        for i in range(0, 200, 20):
            pygame.draw.line(self.screen, WHITE, (i, track_y + 1), (i + 10, track_y + 1), 1)
        for i in range(0, SCREEN_WIDTH, 50):
            if i > 100:
                distance_m = (i - 100) / SCALE
                pygame.draw.line(self.screen, BLACK, (i, GROUND_LEVEL), (i, GROUND_LEVEL + 15), 3)
                pygame.draw.circle(self.screen, RED, (i, GROUND_LEVEL + 20), 5)
                text = self.small_font.render(f"{distance_m:.0f}m", True, BLACK)
                text_rect = text.get_rect()
                text_rect.center = (i, GROUND_LEVEL + 35)
                pygame.draw.rect(self.screen, WHITE, text_rect.inflate(4, 2))
                self.screen.blit(text, text_rect)
    
    def draw_clouds(self):
        cloud_positions = [(200, 80), (400, 60), (600, 100), (800, 70)]
        for x, y in cloud_positions:
            if x < SCREEN_WIDTH - 100:
                for i, (dx, dy, radius) in enumerate([(0, 0, 25), (-15, 5, 20), (15, 5, 20), (-25, 0, 15), (25, 0, 15)]):
                    pygame.draw.circle(self.screen, WHITE, (x + dx, y + dy), radius)
    
    def draw_ui(self):
        eeg_panel = pygame.Rect(10, 10, 330, 120)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), eeg_panel)
        pygame.draw.rect(self.screen, BLUE, eeg_panel, 2)
        current_ratio = self.eeg_monitor.get_current_ratio()
        max_ratio_15s = self.eeg_monitor.get_max_ratio_15s()
        effective_force = self.javelin.get_effective_force()
        
        eeg_info = [
            "EEG STATUS",
            ""
        ]
        if self.javelin.calibrating:
            remaining = self.javelin.get_calibration_time_remaining()
            eeg_info.append(f"Calibrating: {remaining:.1f}s")
        y_offset = 25
        for info in eeg_info:
            if info == "EEG STATUS":
                text = self.font.render(info, True, BLUE)
            elif info.startswith("Calibrating:"):
                text = self.small_font.render(info, True, YELLOW)
            else:
                text = self.small_font.render(info, True, WHITE)
            self.screen.blit(text, (20, y_offset))
            y_offset += 20
        
    # Controls info removed
            
        if self.javelin.landed:
            distance = self.javelin.distance
            dist_panel = pygame.Rect(SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT - 90, 300, 80)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), dist_panel)
            pygame.draw.rect(self.screen, WHITE, dist_panel, 3)
            distance_text = self.font.render(f"Distance: {distance:.2f}m", True, WHITE)
            text_rect = distance_text.get_rect()
            text_rect.center = (SCREEN_WIDTH//2, SCREEN_HEIGHT - 50)
            self.screen.blit(distance_text, text_rect)
    
    def draw_ratio_meter(self):
        center_x = SCREEN_WIDTH // 2
        center_y = 140
        radius = 104
        if self.javelin.freeze_meter:
            current_ratio = self.javelin.final_max_ratio
            max_ratio = self.javelin.final_max_ratio
        elif self.eeg_monitor.calibration_active or self.eeg_monitor.ratio_history:
            current_ratio = self.eeg_monitor.get_current_ratio()
            max_ratio = self.eeg_monitor.get_max_ratio_15s()
        else:
            current_ratio = 0.0
            max_ratio = 0.0
        percentage = min(100, (current_ratio / 3.0) * 100)
        ratio_normalized = min(1.0, current_ratio / 3.0)
        max_percentage = min(100, (max_ratio / 3.0) * 100)
        needle_angle = 180 - (ratio_normalized * 180)
        needle_rad = math.radians(needle_angle)
        pygame.draw.arc(self.screen, (60, 60, 60), (center_x - radius - 8, center_y - radius - 8, (radius + 8) * 2, (radius + 8) * 2), 0, math.pi, 6)
        pygame.draw.arc(self.screen, (30, 30, 40), (center_x - radius, center_y - radius, radius * 2, radius * 2), 0, math.pi, radius)
        zone_thickness = 12
        zone_radius = radius - 15
        green_end = math.pi / 3
        pygame.draw.arc(self.screen, (0, 200, 0), (center_x - zone_radius, center_y - zone_radius, zone_radius * 2, zone_radius * 2), 0, green_end, zone_thickness)
        yellow_start = green_end
        yellow_end = 2 * math.pi / 3
        pygame.draw.arc(self.screen, (255, 200, 0), (center_x - zone_radius, center_y - zone_radius, zone_radius * 2, zone_radius * 2), yellow_start, yellow_end, zone_thickness)
        red_start = yellow_end
        pygame.draw.arc(self.screen, (255, 50, 50), (center_x - zone_radius, center_y - zone_radius, zone_radius * 2, zone_radius * 2), red_start, math.pi, zone_thickness)
        for i in range(5):
            tick_angle = 180 - (i * 45)
            tick_rad = math.radians(tick_angle)
            inner_x = center_x + (radius - 20) * math.cos(tick_rad)
            inner_y = center_y - (radius - 20) * math.sin(tick_rad)
            outer_x = center_x + (radius - 5) * math.cos(tick_rad)
            outer_y = center_y - (radius - 5) * math.sin(tick_rad)
            pygame.draw.line(self.screen, WHITE, (inner_x, inner_y), (outer_x, outer_y), 2)
            label = f""
            text = self.small_font.render(label, True, WHITE)
            text_rect = text.get_rect()
            label_x = center_x + (radius - 30) * math.cos(tick_rad)
            label_y = center_y - (radius - 30) * math.sin(tick_rad)
            text_rect.center = (label_x, label_y)
            self.screen.blit(text, text_rect)
        needle_length = radius - 25
        needle_tip_x = center_x + needle_length * math.cos(needle_rad)
        needle_tip_y = center_y - needle_length * math.sin(needle_rad)
        pygame.draw.line(self.screen, (255, 100, 100), (center_x, center_y), (needle_tip_x, needle_tip_y), 4)
        pygame.draw.line(self.screen, (255, 150, 150), (center_x, center_y), (needle_tip_x, needle_tip_y), 2)
        pygame.draw.circle(self.screen, (80, 80, 80), (center_x, center_y), 8)
        pygame.draw.circle(self.screen, WHITE, (center_x, center_y), 5)
        pygame.draw.line(self.screen, (60, 60, 60), (center_x - radius, center_y), (center_x + radius, center_y), 3)
        percent_text = self.font.render(f"{max_percentage:.1f}%", True, BLACK)
        percent_rect = percent_text.get_rect()
        percent_rect.center = (center_x, center_y + 20)
        self.screen.blit(percent_text, percent_rect)
    
    def create_debug_graph(self):
        if not DEBUG_MODE or len(self.eeg_monitor.debug_ratio_history) < 5:
            return
        try:
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ratios = self.eeg_monitor.debug_ratio_history
            n = len(ratios)
            time_points = list(range(n))
            # Ensure both arrays are the same length
            if len(time_points) > len(ratios):
                time_points = time_points[:len(ratios)]
            elif len(ratios) > len(time_points):
                ratios = ratios[:len(time_points)]
            ax.plot(time_points, ratios, 'cyan', linewidth=2)
            ax.fill_between(time_points, ratios, alpha=0.3, color='cyan')
            ax.set_title('', color='white', fontsize=10)
            ax.set_xlabel('Time', color='white', fontsize=8)
            ax.set_ylabel('Ratio', color='white', fontsize=8)
            ax.tick_params(colors='white', labelsize=6)
            ax.grid(True, alpha=0.3, color='white')
            ax.set_ylim(0, max(3, max(ratios) if ratios else 3))
            plt.tight_layout()
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size = canvas.get_width_height()
            plt.close(fig)
            surf = pygame.image.fromstring(raw_data, size, 'RGB')
            self.debug_graph_surface = surf
        except Exception as e:
            print(f"Debug graph error: {e}")
            self.debug_graph_surface = None
    
    def draw_debug_graph(self):
        if DEBUG_MODE:
            self.create_debug_graph()
            if self.debug_graph_surface:
                graph_width = 300
                graph_height = 225
                scaled_surface = pygame.transform.scale(self.debug_graph_surface, (graph_width, graph_height))
                graph_x = SCREEN_WIDTH - graph_width - 10
                graph_y = 10
                self.screen.blit(scaled_surface, (graph_x, graph_y))
                pygame.draw.rect(self.screen, WHITE, (graph_x, graph_y, graph_width, graph_height), 2)
    
    def run(self):
        running = True
        try:
            while running:
                dt = self.clock.tick(60) / 1000.0
                running = self.handle_events()
                self.update(dt)
                self.draw_background()
                self.javelin.draw(self.screen)
                self.draw_ui()
                self.draw_ratio_meter()
                self.draw_debug_graph()
                pygame.display.flip()
        finally:
            self.eeg_monitor.stop()
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
