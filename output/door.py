import logging
import threading

import serial

log = logging.getLogger("door")

# FakeDoor가 흉내 내는 시간 (5.4절): open 후 통과까지, 닫힘까지 (초)
FAKE_PASS_SEC = 2
FAKE_CLOSE_SEC = 5

# Door (3.6절, 5.4절)
READY_TIMEOUT_SEC = 3      # 포트를 연 뒤 아두이노 READY를 기다리는 최대 시간
RECONNECT_SEC = 5          # 연결이 끊기면 다시 연결을 시도하는 간격
READ_TIMEOUT_SEC = 0.2     # 수신 스레드가 한 줄을 기다리는 시간 (종료 신호를 확인하는 간격)


class FakeDoor:
    """아두이노 없이 문 동작을 흉내 낸다. 명령은 콘솔에 출력한다. Door와 같은 메서드를 가진다.

    문은 하나다. 열린 동안 OPEN이 다시 오면 남은 시간을 취소하고 새 방문 번호로 처음부터 다시 잰다
    (계획서 5.5절, 마지막 OPEN의 방문 번호로 연결: 3.6절).
    """

    def __init__(self) -> None:
        self._callback = None
        self._timers: list[threading.Timer] = []
        self._lock = threading.Lock()
        self._closed = False
        self._generation = 0   # 마지막 OPEN 번호. 이전 OPEN의 타이머는 이벤트를 보내지 않음

    def open(self, visit_id: int) -> None:
        """OPEN: 바로 open, 2초 뒤 pass, 5초 뒤 close 이벤트를 보낸다."""
        print(f"[FakeDoor] OPEN (visit {visit_id})")
        with self._lock:
            if self._closed:
                return
            self._cancel_timers()
            self._generation += 1
            generation = self._generation
        self._emit("open", visit_id)
        self._schedule(FAKE_PASS_SEC, "pass", visit_id, generation)
        self._schedule(FAKE_CLOSE_SEC, "close", visit_id, generation)

    def deny(self) -> None:
        print("[FakeDoor] DENY")

    def idle(self) -> None:
        print("[FakeDoor] IDLE")

    def wait(self) -> None:
        print("[FakeDoor] WAIT")

    def hint(self) -> None:
        print("[FakeDoor] HINT")

    def on_event(self, callback) -> None:
        """callback(event: str, visit_id: int | None)을 등록한다."""
        self._callback = callback

    def close(self) -> None:
        """남은 타이머를 모두 취소한다."""
        with self._lock:
            self._closed = True
            self._cancel_timers()

    def _cancel_timers(self) -> None:
        for t in self._timers:
            t.cancel()
        self._timers.clear()

    def _schedule(self, delay: float, event: str, visit_id: int, generation: int) -> None:
        timer = threading.Timer(delay, self._fire, args=(event, visit_id, generation))
        timer.daemon = True   # 프로그램이 끝날 때 기다리지 않음
        with self._lock:
            if self._closed or generation != self._generation:
                return
            self._timers.append(timer)
        timer.start()

    def _fire(self, event: str, visit_id: int, generation: int) -> None:
        with self._lock:
            if self._closed or generation != self._generation:
                return
        self._emit(event, visit_id)

    def _emit(self, event: str, visit_id: int | None) -> None:
        print(f"[FakeDoor] -> {event} (visit {visit_id})")
        if self._callback is not None:
            self._callback(event, visit_id)


class Door:
    """노트북 쪽 아두이노 통신 (USB 시리얼). FakeDoor와 같은 메서드를 가진다.

    - 포트를 열면 아두이노가 리셋되므로 READY를 최대 3초 기다린 뒤 명령을 보낸다.
    - 수신 스레드가 PASS / CLOSED를 받아 마지막 OPEN의 방문 번호로 콜백을 부른다.
    - 연결이 없거나 끊기면 로그를 남기고 5초마다 다시 연결한다. 그동안 온 명령은 버린다.
    - 어떤 오류도 메인 루프로 올려 보내지 않는다. 콜백은 수신 스레드에서 실행된다 (DB 함수만 부를 것).
    """

    def __init__(self, port: str, baud: int) -> None:
        self._port = port
        self._baud = baud
        self._ser = None
        self._write_lock = threading.Lock()
        self._callback = None
        self._current_visit: int | None = None
        self._stop = threading.Event()

        if not self._connect():
            log.warning("아두이노 포트 %s를 열 수 없습니다. %d초마다 다시 연결합니다. "
                        "(USB 연결, COM 번호, Arduino IDE 시리얼 모니터가 열려 있는지 확인)",
                        port, RECONNECT_SEC)
        self._thread = threading.Thread(target=self._reader, name="door-reader", daemon=True)
        self._thread.start()

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def open(self, visit_id: int) -> None:
        """OPEN 전송, 현재 방문 번호를 기억, 콜백 ("open", visit_id). 전송하지 못하면 콜백도 하지 않는다."""
        self._current_visit = visit_id
        if self._send("OPEN"):
            self._notify("open", visit_id)

    def deny(self) -> None:
        self._send("DENY")

    def idle(self) -> None:
        self._send("IDLE")

    def wait(self) -> None:
        self._send("WAIT")

    def hint(self) -> None:
        self._send("HINT")

    def on_event(self, callback) -> None:
        """callback(event: str, visit_id: int | None)을 등록한다."""
        self._callback = callback

    def close(self) -> None:
        """수신 스레드를 멈추고 포트를 닫는다."""
        self._stop.set()
        self._thread.join(timeout=READ_TIMEOUT_SEC * 5)
        self._disconnect(reason=None)

    # ── 내부 ──

    def _connect(self) -> bool:
        """포트를 열고 READY를 기다린다. 성공하면 True."""
        try:
            ser = serial.Serial(self._port, self._baud, timeout=READ_TIMEOUT_SEC)
        except (serial.SerialException, OSError, ValueError):
            return False

        ready = False
        waited = 0.0
        try:
            while waited < READY_TIMEOUT_SEC and not self._stop.is_set():
                line = ser.readline()
                waited += READ_TIMEOUT_SEC
                if line.decode("ascii", errors="ignore").strip().upper() == "READY":
                    ready = True
                    break
        except (serial.SerialException, OSError):
            ser.close()
            return False

        with self._write_lock:
            self._ser = ser
        if ready:
            log.info("아두이노 연결됨 (%s, READY 받음)", self._port)
        else:
            log.warning("아두이노 연결됨 (%s), %d초 안에 READY가 오지 않아 그대로 진행합니다",
                        self._port, READY_TIMEOUT_SEC)
        return True

    def _disconnect(self, reason: str | None) -> None:
        with self._write_lock:
            ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except (serial.SerialException, OSError):
                pass
            if reason:
                log.warning("아두이노 연결이 끊겼습니다 (%s). %d초마다 다시 연결합니다.", reason, RECONNECT_SEC)

    def _send(self, command: str) -> bool:
        with self._write_lock:
            ser = self._ser
            if ser is None:
                log.warning("아두이노 연결 안 됨: %s 명령을 버립니다", command)
                return False
            try:
                ser.write((command + "\n").encode("ascii"))
                return True
            except (serial.SerialException, OSError) as e:
                error = e
        self._disconnect(reason=f"{command} 전송 실패: {error}")
        return False

    def _reader(self) -> None:
        while not self._stop.is_set():
            ser = self._ser
            if ser is None:
                if self._stop.wait(RECONNECT_SEC):
                    break
                self._connect()
                continue
            try:
                line = ser.readline()
            except (serial.SerialException, OSError, TypeError, AttributeError) as e:
                if self._stop.is_set():
                    break
                self._disconnect(reason=f"수신 실패: {e}")
                continue
            if not line:
                continue
            text = line.decode("ascii", errors="ignore").strip().upper()
            if text == "PASS":
                self._notify("pass", self._current_visit)
            elif text == "CLOSED":
                self._notify("close", self._current_visit)
                self._current_visit = None
            elif text == "READY":
                log.info("아두이노가 다시 시작되었습니다 (READY)")
            else:
                log.debug("아두이노: %s", text)

    def _notify(self, event: str, visit_id: int | None) -> None:
        if self._callback is None:
            return
        try:
            self._callback(event, visit_id)
        except Exception:
            log.exception("문 이벤트 처리(콜백) 중 오류: %s (visit %s)", event, visit_id)