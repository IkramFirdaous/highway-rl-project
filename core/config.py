# ─── Config (NE PAS MODIFIER) ────────────────────────────────────────

SHARED_CORE_ENV_ID = "highway-v0"

SHARED_CORE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "observe_intentions": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [20, 25, 30],
    },
    "lanes_count": 4,
    "vehicles_count": 45,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 30,
    "ego_spacing": 2,
    "vehicles_density": 1.0,
    "collision_reward": -1.5,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.7,
    "lane_change_reward": -0.02,
    "reward_speed_range": [22, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}

# ─── Hyperparamètres partagés (DQN scratch + SB3 utilisent les mêmes) ────────

LR            = 5e-4
GAMMA         = 0.99
BUFFER_SIZE   = 50_000
BATCH_SIZE    = 128
TARGET_UPDATE = 500

# ─── Hyperparamètres DQN scratch uniquement ──────────────────────────────────

EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.99
N_EPISODES    = 2000

# ─── Hyperparamètres SB3 uniquement ──────────────────────────────────────────

SB3_TIMESTEPS          = 40_000
SB3_LEARNING_STARTS    = 1_000    # steps avant le premier update
SB3_EXPLORATION_FRAC   = 0.3      # fraction des timesteps pour décroître epsilon

# ─── Évaluation (partagée DQN scratch + SB3) ─────────────────────────────────

EVAL_EPISODES = 50
EVAL_SEEDS    = [42, 43, 44]

# ─── Dérivé de la config  ─────────────────

N_FEATURES    = len(SHARED_CORE_CONFIG["observation"]["features"])
N_VEHICLES    = SHARED_CORE_CONFIG["observation"]["vehicles_count"]
OBS_SHAPE     = (N_VEHICLES, N_FEATURES)
N_ACTIONS     = 5  # DiscreteMetaAction : LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER