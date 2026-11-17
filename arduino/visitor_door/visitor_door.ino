/*
 * 방문자 확인 시스템 / Arduino UNO R3
 * 9600bps, 한 줄에 한 명령: OPEN, DENY, IDLE, WAIT, RETRY
 * 송신: READY, OPENED, CLOSED, BUSY, ERROR ...
 * 통과 센서 없이 문을 연 뒤 OPEN_HOLD_MS가 지나면 자동으로 닫습니다.
 * 실제 출입문이 아닌 폼보드 모형용 코드입니다.
 */
#include <Arduino.h>
#include <Servo.h>
#include <string.h>

// ===== 배선과 모형에 맞춰 이 구역을 조정하세요. =====
const uint8_t SERVO_PIN = 9;
const uint8_t RED_PIN = 4;
const uint8_t YELLOW_PIN = 5;
const uint8_t GREEN_PIN = 6;
const uint8_t LED_ON_LEVEL = HIGH;  // LOW로 켜지는 LED 모듈은 LOW로 변경

const int CLOSED_ANGLE = 0;
const int OPEN_ANGLE = 90;
const unsigned long SERVO_STEP_MS = 20UL;  // 20ms마다 1도씩 이동
const unsigned long SERVO_SETTLE_MS = 250UL;
const unsigned long OPEN_HOLD_MS = 30000UL;  // 열림 완료 처리 후 유지 시간 (기본 30초)
const unsigned long DENY_DISPLAY_MS = 5000UL;
const unsigned long YELLOW_BLINK_MS = 250UL;
const unsigned long SERIAL_PARTIAL_TIMEOUT_MS = 1000UL;

static_assert(CLOSED_ANGLE >= 0 && CLOSED_ANGLE <= 180, "Invalid closed angle");
static_assert(OPEN_ANGLE >= 0 && OPEN_ANGLE <= 180, "Invalid open angle");
static_assert(CLOSED_ANGLE != OPEN_ANGLE, "Door angles must differ");

enum DoorState : uint8_t {
  IDLE_STATE,
  OPENING_STATE,
  OPEN_STATE,
  CLOSING_STATE,
  DENIED_STATE,
  WAITING_STATE,
  RETRY_STATE
};

enum LineError : uint8_t {
  LINE_OK,
  LINE_TOO_LONG,
  LINE_INVALID,
  LINE_TIMED_OUT
};

Servo doorServo;
DoorState state = CLOSING_STATE;
bool startingUp = true;
int servoAngle = CLOSED_ANGLE;
int targetAngle = CLOSED_ANGLE;
unsigned long lastServoStepAt = 0;
unsigned long stateStartedAt = 0;
unsigned long openedAt = 0;

char commandBuffer[24];
uint8_t commandLength = 0;
LineError lineError = LINE_OK;
unsigned long lastSerialByteAt = 0;

// unsigned long 시간 차이를 사용해 millis()가 한 바퀴 돌아도 동작합니다.
bool elapsed(unsigned long now, unsigned long since, unsigned long interval) {
  return static_cast<unsigned long>(now - since) >= interval;
}

void setLeds(bool red, bool yellow, bool green) {
  const uint8_t offLevel = (LED_ON_LEVEL == HIGH) ? LOW : HIGH;
  digitalWrite(RED_PIN, red ? LED_ON_LEVEL : offLevel);
  digitalWrite(YELLOW_PIN, yellow ? LED_ON_LEVEL : offLevel);
  digitalWrite(GREEN_PIN, green ? LED_ON_LEVEL : offLevel);
}

bool cycleInProgress() {
  return state == OPENING_STATE || state == OPEN_STATE || state == CLOSING_STATE;
}

void beginOpen(unsigned long now) {
  state = OPENING_STATE;
  stateStartedAt = now;
  targetAngle = OPEN_ANGLE;
  lastServoStepAt = now;
}

void beginClose(unsigned long now) {
  state = CLOSING_STATE;
  stateStartedAt = now;
  targetAngle = CLOSED_ANGLE;
  lastServoStepAt = now;
}

void tickServo(unsigned long now) {
  if (state != OPENING_STATE && state != CLOSING_STATE) {
    return;
  }
  if (servoAngle != targetAngle) {
    if (elapsed(now, lastServoStepAt, SERVO_STEP_MS)) {
      servoAngle += (servoAngle < targetAngle) ? 1 : -1;
      doorServo.write(servoAngle);
      lastServoStepAt = now;
    }
    return;
  }
  // 목표 각도 명령 후 여유 시간을 둡니다. 실제 위치를 측정하는 센서는 없습니다.
  if (!elapsed(now, lastServoStepAt, SERVO_SETTLE_MS)) {
    return;
  }
  if (state == OPENING_STATE) {
    state = OPEN_STATE;
    openedAt = now;
    Serial.println(F("OPENED"));
  } else {
    state = IDLE_STATE;
    stateStartedAt = now;
    Serial.println(F("CLOSED"));
    if (startingUp) {
      startingUp = false;
      Serial.println(F("READY"));
    }
  }
}

void tickTimers(unsigned long now) {
  if (state == OPEN_STATE) {
    if (elapsed(now, openedAt, OPEN_HOLD_MS)) {
      beginClose(now);
    }
  } else if (state == DENIED_STATE && elapsed(now, stateStartedAt, DENY_DISPLAY_MS)) {
    state = IDLE_STATE;
    stateStartedAt = now;
  }
}

void tickLeds(unsigned long now) {
  if (state == OPENING_STATE || state == OPEN_STATE ||
      (state == CLOSING_STATE && !startingUp)) {
    setLeds(false, false, true);
  } else if (state == DENIED_STATE) {
    setLeds(true, false, false);
  } else if (state == WAITING_STATE) {
    setLeds(false, true, false);
  } else if (state == RETRY_STATE) {
    const bool blinkOn = ((now - stateStartedAt) / YELLOW_BLINK_MS) % 2UL == 0;
    setLeds(false, blinkOn, false);
  } else {
    setLeds(false, false, false);
  }
}

void handleCommand(char *command, unsigned long now) {
  // 앞뒤 공백/탭은 허용하지만 명령은 대문자만 허용합니다.
  while (*command == ' ' || *command == '\t') {
    ++command;
  }
  size_t length = strlen(command);
  while (length > 0 && (command[length - 1] == ' ' || command[length - 1] == '\t')) {
    command[--length] = '\0';
  }
  if (length == 0) {
    return;
  }
  const bool isOpen = strcmp(command, "OPEN") == 0;
  const bool isDeny = strcmp(command, "DENY") == 0;
  const bool isIdle = strcmp(command, "IDLE") == 0;
  const bool isWait = strcmp(command, "WAIT") == 0;
  const bool isRetry = strcmp(command, "RETRY") == 0;
  if (!isOpen && !isDeny && !isIdle && !isWait && !isRetry) {
    Serial.println(F("ERROR UNKNOWN_COMMAND"));
    return;
  }
  if (cycleInProgress()) {
    // 중복 OPEN은 타이머를 초기화하지 않습니다. 다른 방문 명령은 대기열에 넣지 않습니다.
    if (!(isOpen && (state == OPENING_STATE || state == OPEN_STATE))) {
      Serial.println(F("BUSY"));
    }
    return;
  }
  if (isOpen) {
    beginOpen(now);
  } else {
    stateStartedAt = now;
    if (isDeny) {
      state = DENIED_STATE;
    } else if (isWait) {
      state = WAITING_STATE;
    } else if (isRetry) {
      state = RETRY_STATE;
    } else {
      state = IDLE_STATE;
    }
  }
}

void finishLine(unsigned long now) {
  if (lineError == LINE_TOO_LONG) {
    Serial.println(F("ERROR LINE_TOO_LONG"));
  } else if (lineError == LINE_INVALID) {
    Serial.println(F("ERROR INVALID_CHARACTER"));
  } else if (lineError == LINE_OK) {
    commandBuffer[commandLength] = '\0';
    handleCommand(commandBuffer, now);
  }
  // TIMEOUT은 이미 안내했으므로 줄바꿈 시 중복 출력하지 않습니다.
  commandLength = 0;
  lineError = LINE_OK;
}

void readSerial(unsigned long now) {
  if (commandLength > 0 && elapsed(now, lastSerialByteAt, SERIAL_PARTIAL_TIMEOUT_MS)) {
    commandLength = 0;
    lineError = LINE_TIMED_OUT;
    Serial.println(F("ERROR TIMEOUT"));
  }
  // 입력이 계속 와도 모터·자동 닫힘 처리가 돌아가도록 읽기 양을 제한합니다.
  uint8_t budget = 16;
  while (budget > 0 && Serial.available() > 0) {
    --budget;
    const char incoming = static_cast<char>(Serial.read());
    lastSerialByteAt = now;
    if (incoming == '\r' || incoming == '\n') {
      finishLine(now);
    } else if (lineError == LINE_OK) {
      if ((incoming < ' ' || incoming > '~') && incoming != '\t') {
        commandLength = 0;
        lineError = LINE_INVALID;
      } else if (commandLength >= sizeof(commandBuffer) - 1) {
        commandLength = 0;
        lineError = LINE_TOO_LONG;
      } else {
        commandBuffer[commandLength++] = incoming;
      }
    }
  }
}

void setup() {
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  setLeds(false, false, false);
  Serial.begin(9600);
  doorServo.attach(SERVO_PIN);
  doorServo.write(CLOSED_ANGLE);
  const unsigned long now = millis();
  lastServoStepAt = now;
  stateStartedAt = now;
}

void loop() {
  const unsigned long now = millis();
  readSerial(now);
  tickServo(now);
  tickTimers(now);
  tickLeds(now);
}
