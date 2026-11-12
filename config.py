# ── 경로 ──
DATA_DIR = "data"
DB_PATH = "data/visitor.db"
FACES_DIR = "data/faces"
CAPTURES_DIR = "data/captures"
MODELS_DIR = "models"

# ── 카메라·YOLO (B) ──
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
WINDOW_NAME = "visitor"
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.5
YOLO_IMGSZ = 480
MIN_PERSON_HEIGHT = 120        # 픽셀. 이보다 작은(멀리 있는) 사람은 건너뜀
MIRROR = True                  # 화면을 거울처럼 좌우 반전 (보이는 화면만. 인식·저장은 원본)

# ── 얼굴 인식·판정 (A) ──
FACE_INTERVAL = 5              # 몇 프레임마다 얼굴 인식을 할지
MATCH_THRESHOLD = 0.5          # 임시값. 라이브러리를 정하면 A가 바꾸고 3주차 실험으로 확정
USE_CLAHE = False              # 3주차 실험 결과로 결정. 등록·인식에 똑같이 적용
VOTE_WINDOW = 5                # 최근 몇 번의 결과로 판정할지
VOTE_MIN = 4                   # 그중 몇 번 이상 같아야 확정할지
FACE_HINT_SEC = 2              # 얼굴을 못 찾은 지 몇 초부터 안내할지
UNKNOWN_TIMEOUT_SEC = 5        # 얼굴을 못 찾은 지 몇 초면 "인식 불가"로 확정할지

# ── 안내 (B) ──
COOLDOWN_SEC = 60              # 같은 사람 재안내·재기록 금지 시간
UNKNOWN_DEDUP_MIN = 10         # 미등록자 중복 확인 범위 (분)
MESSAGE_SEC = 5                # 안내 배너 표시 시간
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
MSG_WELCOME = "{name}님 환영합니다."
MSG_COUNTER = "카운터에 가서 문의하세요."
MSG_FACE_HINT = "얼굴이 보이도록 카메라를 바라봐 주세요."

# ── 아두이노 (B·C) ──
USE_ARDUINO = False            # 부품이 오기 전에는 False → FakeDoor 사용
SERIAL_PORT = "COM3"           # 장치 관리자에서 확인
SERIAL_BAUD = 9600

# ── 대시보드 (B) ──
DASHBOARD_HOST = "0.0.0.0"     # 같은 와이파이의 휴대폰에서 접속
DASHBOARD_PORT = 5000
