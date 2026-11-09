import threading

# FakeDoor가 흉내 내는 시간 (5.4절): open 후 통과까지, 닫힘까지 (초)
FAKE_PASS_SEC = 2
FAKE_CLOSE_SEC = 5


class FakeDoor:
    """아두이노 없이 문 동작을 흉내 낸다. 명령은 콘솔에 출력한다. Door와 같은 메서드를 가진다."""

    def __init__(self) -> None:
        self._callback = None
        self._timers: list[threading.Timer] = []
        self._lock = threading.Lock()
        self._closed = False

    def open(self, visit_id: int) -> None:
        """OPEN: 바로 open, 2초 뒤 pass, 5초 뒤 close 이벤트를 보낸다."""
        print(f"[FakeDoor] OPEN (visit {visit_id})")
        self._emit("open", visit_id)
        self._schedule(FAKE_PASS_SEC, "pass", visit_id)
        self._schedule(FAKE_CLOSE_SEC, "close", visit_id)

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
            for t in self._timers:
                t.cancel()
            self._timers.clear()

    def _schedule(self, delay: float, event: str, visit_id: int) -> None:
        timer = threading.Timer(delay, self._fire, args=(event, visit_id))
        timer.daemon = True   # 프로그램이 끝날 때 기다리지 않음
        with self._lock:
            if self._closed:
                return
            self._timers = [t for t in self._timers if t.is_alive()]
            self._timers.append(timer)
        timer.start()

    def _fire(self, event: str, visit_id: int) -> None:
        with self._lock:
            if self._closed:
                return
        self._emit(event, visit_id)

    def _emit(self, event: str, visit_id: int | None) -> None:
        print(f"[FakeDoor] -> {event} (visit {visit_id})")
        if self._callback is not None:
            self._callback(event, visit_id)
