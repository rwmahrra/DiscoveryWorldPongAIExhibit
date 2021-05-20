
class Config:
    PADDING = 10  # Distance between screen edge and player paddles (px)
    MAX_SCORE = 21  # Points one side must win to finish game
    WIDTH = 192  # Game window width (px)
    HEIGHT = 160  # Game window height (px)
    SPEEDUP = 1  # Flat multiplier to game movement speeds
    ACTIONS = ["LEFT", "RIGHT", "NONE"]
    GAME_FPS = 60
    AI_FRAME_INTERVAL = 5
    BALL_MARKER_SIZE = 4  # Pixel height and width of experimental position markers
    CUSTOM = 0
    HIT_PRACTICE = 2
    CUSTOM_ACTION_SIZE = 3
    CUSTOM_STATE_SHAPE = HEIGHT // 2, WIDTH // 2
    CUSTOM_STATE_SIZE = HEIGHT // 2 * WIDTH // 2
    BALL_DIAMETER = 6
    PADDLE_HEIGHT = 2
    PADDLE_WIDTH = 20
    BOTTOM_PADDLE_Y = HEIGHT - PADDING
    TOP_PADDLE_Y = PADDING
    ENV_TYPE = CUSTOM
    DEBUG = True
    ENABLE_AUDIO = False
    VOLLEY_SPEEDUP = 0.2
    AUDIO_DIR = "C:\\dev\\DiscoveryWorld\\exhibit\\game\\resources\\"
