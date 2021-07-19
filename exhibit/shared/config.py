
class Config:
    PADDING = 10  # Distance between screen edge and player paddles (px)
    MAX_SCORE = 7  # Points one side must win to finish game, 21
    WIDTH = 160  # Game window width (px)
    HEIGHT = 192  # Game window height (px)
    SPEEDUP = 1  # Flat multiplier to game movement speeds
    ACTIONS = ["UP", "DOWN", "NONE", "DEPTH"]
    GAME_FPS = 80 # 60
    AI_FRAME_INTERVAL = 5  # AI will publish inference every n frames
    BALL_MARKER_SIZE = 4  # Pixel height and width of experimental position markers
    CUSTOM = 0
    HIT_PRACTICE = 2
    CUSTOM_ACTION_SIZE = 3
    CUSTOM_STATE_SHAPE = HEIGHT // 2, WIDTH // 2
    CUSTOM_STATE_SIZE = HEIGHT // 2 * WIDTH // 2
    BALL_DIAMETER = 6
    PADDLE_HEIGHT = 20
    PADDLE_WIDTH = 2
    LEFT_PADDLE_X = PADDING
    RIGHT_PADDLE_X = WIDTH - PADDING
    ENV_TYPE = CUSTOM
    DEBUG = True
    ENABLE_AUDIO = True
    VOLLEY_SPEEDUP = 0.2  # Multiplier to ball speed after each paddle hit
    #AUDIO_DIR = "C:\\dev\\DiscoveryWorld\\exhibit\\game\\resources\\"
    AUDIO_DIR = "C:\\Users\\DW Pong\\Downloads\\DiscoveryWorldPongAIExhibit-master\\DiscoveryWorldPongAIExhibit-master\\exhibit\\game\\resources\\"
