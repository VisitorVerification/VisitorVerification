# 모든 숫자·경로·문구는 여기에만 둔다. 각 값은 주석의 담당자가 정한다. (공통 명세 3.3)

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

# ── 얼굴 인식·판정 (A) ──
FACE_INTERVAL = 5              # 몇 프레임마다 얼굴 인식을 할지
MATCH_THRESHOLD = 0.55         # SFace 거리(1 - 코사인 유사도). 문서 기준 0.637보다 엄격한 임시값, 3주차 실험으로 확정
USE_CLAHE = False              # 3주차 실험 결과로 결정. 등록·인식에 똑같이 적용
VOTE_WINDOW = 5                # 최근 몇 번의 결과로 판정할지
VOTE_MIN = 4                   # 그중 몇 번 이상 같아야 확정할지
FACE_HINT_SEC = 2              # 얼굴을 못 찾은 지 몇 초부터 안내할지
UNKNOWN_TIMEOUT_SEC = 5        # 얼굴을 못 찾은 지 몇 초면 "인식 불가"로 확정할지
# A 추가값 (팀에 공유 필요)
FACE_MODEL_KEY = "sface_2021dec"   # C의 DB에 벡터를 저장할 때 쓰는 모델 이름. USE_CLAHE면 "_clahe"가 자동으로 붙음
FACE_DET_MODEL = "face_detection_yunet_2023mar.onnx"   # MODELS_DIR 안의 파일 이름
FACE_REC_MODEL = "face_recognition_sface_2021dec.onnx"
FACE_DET_SCORE = 0.8           # YuNet 얼굴 검출 신뢰도 기준
MIN_FACE_WIDTH = 40            # 픽셀. 이보다 작은 얼굴은 "얼굴 못 찾음"으로 처리
PERSON_BOX_MARGIN = 0.1        # 사람 박스를 넓혀 자르는 비율
TRACK_EXPIRE_SEC = 10          # 이 시간 넘게 갱신되지 않은 추적 번호는 판정 상태를 지움

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
