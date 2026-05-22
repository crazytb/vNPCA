import copy

# === Core Simulation Parameters ===
simulation_time = 5_000  # Total simulation time in us
simulation_slot = simulation_time // 9  # Slot duration in us (9us for 802.11ax)

# === Frame and Transmission Parameters ===
PPDU_DURATION = 33  # PPDU duration in slots (default frame size)
RADIO_TRANSITION_TIME = 1  # Radio transition time in slots
ENERGY_COST = 15.0  # Fixed energy cost per transmission attempt (increased for better differentiation)

# PPDU Duration Variants for controlled experiments
PPDU_DURATION_VARIANTS = {
    'short': 20,       # Short frame (20 slots)
    'medium': 33,      # Default frame (33 slots)  
    'long': 50,        # Long frame (50 slots)
    'extra_long': 80,  # Extra long frame (80 slots)
}

# === OBSS Parameters ===
OBSS_GENERATION_RATE = {
    'primary': 0.01,    # OBSS generation rate for primary channel
    'secondary': 0.0,   # OBSS generation rate for secondary/NPCA channel
    'high': 0.05,       # High OBSS rate for stress testing
    'medium': 0.03,     # Medium OBSS rate
    'low': 0.01,        # Low OBSS rate
}

OBSS_DURATION_RANGE = {
    'short': (10, 30),       # Short OBSS duration
    'medium': (50, 100),     # Medium OBSS duration  
    'long': (150, 250),      # Long OBSS duration (increased)
    'extreme': (300, 500),   # Extreme OBSS duration
    'fixed_20': (20, 20),    # Fixed 20 slots
    'fixed_50': (50, 50),    # Fixed 50 slots
    'fixed_100': (100, 100), # Fixed 100 slots
    'fixed_150': (150, 150), # Fixed 150 slots
    'fixed_200': (200, 200), # Fixed 200 slots (new)
    'fixed_300': (300, 300), # Fixed 300 slots (new)
}

# === Training Parameters ===
DEFAULT_NUM_EPISODES = 1000  # Increased for better learning and random env convergence
DEFAULT_NUM_SLOTS_PER_EPISODE = int(100_000/9)  # Increased for more decisions per episode
DEFAULT_NUM_STAS_CH0 = 10  # Default number of STAs in secondary channel
DEFAULT_NUM_STAS_CH1 = 10  # Default number of STAs in primary channel

# === Random Environment Parameters ===
RANDOM_OBSS_DURATION_RANGE = (20, 200)  # Random OBSS duration range for robust training
RANDOM_PPDU_VARIANTS = ['short', 'medium', 'long', 'extra_long']  # Random PPDU variants
RANDOM_OBSS_GENERATION_RATE_RANGE = (0.02, 0.08)  # Random OBSS generation rate range

# 공통 설정
base_config = {
    "num_channels": 2,
    "simulation_time": simulation_time,
    "obss_enabled_per_channel": [False, True],
    # "npca_enabled": [False, True],
    "obss_generation_rate": OBSS_GENERATION_RATE['high'],  # 기존 호환성 유지
    "obss_frame_size_range": (20, 201),  # 범위로 설정
    "ppdu_duration": PPDU_DURATION,
    "radio_transition_time": RADIO_TRANSITION_TIME,
    "energy_cost": ENERGY_COST,
}

# 후보 값 (기존 호환성 유지)
sta_values = [2, 6, 10]  # 각 채널의 STA 수
# sta_values = [2]  # 각 채널의 STA 수
frame_sizes = [PPDU_DURATION, PPDU_DURATION * 5]
# frame_sizes = [PPDU_DURATION]
frame_labels = {33: "fshort", 33*5: "flong"}
# npca_options = [[False, True]]  # 추가된 부분
npca_options = [[False, True], [False, False]]  # 추가된 부분

# === LLM Reward Designer Parameters ===
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_UPDATE_INTERVAL = 50   # episodes between LLM reward-design calls
LLM_USE_MOCK = False        # set True to skip API calls during unit tests

# === Energy Model (IEEE 802.11ax TG, 11-14-0980-16-00ax, 20MHz, V=1.1V, NSS=1) ===
SLOT_DURATION_US = 9.0             # μs per slot (802.11ax)
ENERGY_TX_PER_SLOT_UJ = 2.772      # 280mA × 1.1V × 9μs = 308mW × 9μs
ENERGY_LISTEN_PER_SLOT_UJ = 0.495  # 50mA × 1.1V × 9μs = 55mW × 9μs (backoff/frozen/CCA)
ENERGY_NPCA_TRANSITION_UJ = 0.75   # TX↔Listen @ 75mW × 0.01ms per NPCA switch event

# 시뮬레이션 설정 생성
simulation_configs = []
for ch0 in sta_values:
    for ch1 in sta_values:
        for fs in frame_sizes:
            for npca_enabled in npca_options:
                config = copy.deepcopy(base_config)
                config["stas_per_channel"] = [ch0, ch1]
                config["frame_size"] = fs
                config["npca_enabled"] = npca_enabled
                config["label"] = f"s{ch0}_{ch1}_{frame_labels[fs]}_npca_{int(npca_enabled[1])}"
                simulation_configs.append(config)