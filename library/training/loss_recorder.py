from typing import List
import time

class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        losses = len(self.loss_list)
        if losses == 0:
            return 0
        return self.loss_total / losses

class RateTracker:
    """
    Tracks recent iterations per second using EMA of step durations.

    A larger smoothing value gives more weight to recent steps.
    """

    def __init__(self, smoothing: float = 0.2, skip_first: bool = True):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1].")

        self.smoothing = smoothing
        self.beta = 1.0 - smoothing

        self._mean_step_time: float = 0.0
        self._last_time: float | None = None
        self._count: int = 0

        self._skip_first = skip_first
        self._first_skipped = False

    def tick(self) -> None:
        now = time.perf_counter()

        if self._last_time is not None:
            elapsed = now - self._last_time

            if self._skip_first and not self._first_skipped:
                self._first_skipped = True
            else:
                self._add(elapsed)

        self._last_time = now

    def _add(self, value: float) -> None:
        self._count += 1

        if self._count == 1:
            self._mean_step_time = value
        else:
            self._mean_step_time = (
                self.beta * self._mean_step_time
                + self.smoothing * value
            )

    @property
    def it_per_sec(self) -> float:
        if self._mean_step_time <= 0.0:
            return 0.0
        return 1.0 / self._mean_step_time

    @property
    def mean_step_time(self) -> float:
        return self._mean_step_time

    @property
    def display_rate(self) -> str:
        rate = self.it_per_sec

        if rate >= 1.0:
            return f"{rate:.2f}it/s"
        elif rate > 0.0:
            return f"{1.0 / rate:.2f}s/it"
        else:
            return "0.0it/s"

    @property
    def step_count(self) -> int:
        return self._count