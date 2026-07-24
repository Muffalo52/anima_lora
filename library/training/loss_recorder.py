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

class EMARecorder:
    """
    Calculates a bias-corrected Exponential Moving Average (EMA) with
    optional outlier-robust statistics.

    This is the preferred method for smoothing noisy data in real-time,
    such as mini-batch losses during model training. It gives more weight
    to recent values, making it responsive to trends.

    Uses Welford's online algorithm for running variance to enable
    automatic outlier clipping. Supports both the old LossRecorder
    keyword-argument calling convention (epoch, step, loss) and the
    simpler positional value call.
    """
    def __init__(self, smoothing: float = 0.25, outlier_sigma: float = 0.0):
        """
        Initializes the EMA recorder.

        Args:
            smoothing (float): The smoothing factor, typically between 0 and 1.
                A smaller value (e.g., 0.01) results in a smoother, less responsive average.
                A larger value (e.g., 0.25) results in a more responsive average.
                Default: 0.25 for fast adaptation while still filtering noise.
            outlier_sigma (float): Number of standard deviations beyond which values
                are clipped. Set to 0 to disable outlier clipping.
                Recommended: 3.0--5.0 for typical training. Default: 0 (disabled).
        """
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("Smoothing factor must be between 0 and 1.")
            
        self.smoothing = smoothing
        self.beta = 1 - self.smoothing  # The decay factor
        
        self.ema: float = 0.0
        self.num_updates: int = 0

        # Welford's online algorithm state for running mean and variance
        self._welford_mean: float = 0.0
        self._welford_m2: float = 0.0  # Sum of squared differences
        self.outlier_sigma: float = outlier_sigma

    def add(self, value: float = None, *, epoch: int = None, step: int = None, loss: float = None) -> None:
        """
        Updates the EMA with a new value.

        Accepts both the simple positional form `add(value)` and the
        legacy LossRecorder keyword form `add(epoch=epoch, step=step, loss=loss)`.
        """
        # Support legacy LossRecorder calling convention
        if value is None:
            if loss is not None:
                value = loss
            else:
                return  # Nothing to add

        self.num_updates += 1

        # Outlier clipping: check against current statistics BEFORE updating them.
        # This prevents a single extreme outlier from contaminating the running variance.
        if self.outlier_sigma > 0 and self.num_updates > 5:
            variance = self._welford_m2 / (self.num_updates - 1) if self.num_updates > 1 else 0.0
            std = variance ** 0.5
            # Fallback: if all values have been nearly identical (std ~ 0),
            # use a minimum std of 20% of the current EMA to still catch outliers
            min_std = 0.2 * abs(self.ema) if self.ema != 0 else 0.01
            effective_std = max(std, min_std)
            if effective_std > 0:
                lower_bound = self.ema - self.outlier_sigma * effective_std
                upper_bound = self.ema + self.outlier_sigma * effective_std
                value = max(lower_bound, min(value, upper_bound))

        # Update running variance via Welford's algorithm (with possibly clipped value)
        delta = value - self._welford_mean
        self._welford_mean += delta / self.num_updates
        delta2 = value - self._welford_mean
        self._welford_m2 += delta * delta2

        # Standard EMA update rule
        self.ema = self.beta * self.ema + self.smoothing * value

    @property
    def average(self) -> float:
        """
        Returns the bias-corrected moving average.

        Bias correction is important at the beginning of the series, as it
        corrects for the fact that the EMA is initialized at zero.
        """
        if self.num_updates == 0:
            return 0.0
            
        # Bias correction warms up the average faster
        # As num_updates -> infinity, the correction factor -> 1
        correction_factor = 1 - (self.beta ** self.num_updates)
        return self.ema / correction_factor

    @property
    def moving_average(self) -> float:
        """Alias for backward compatibility with LossRecorder."""
        return self.average

    @property
    def std(self) -> float:
        """Returns the running standard deviation (for diagnostics)."""
        if self.num_updates < 2:
            return 0.0
        return (self._welford_m2 / (self.num_updates - 1)) ** 0.5


class RateTracker:
    """
    Tracks iterations per second using Welford's online algorithm for
    running mean and variance of step durations.

    Unlike tqdm's built-in EMA-based rate smoothing, Welford provides
    both the running mean step time AND its variance, enabling
    statistically robust rate display that is resistant to I/O hiccups
    and other transient timing outliers.

    Usage:
        rate_tracker = RateTracker()
        # ... inside the training loop, at each optimizer step boundary:
        rate_tracker.tick()
        # Display in tqdm postfix:
        logs = {"avr_loss": ..., "it/s": rate_tracker.display_rate}
    """
    def __init__(self, skip_first: bool = True):
        """
        Args:
            skip_first: If True, discards the first recorded interval (step 0→1),
                which typically includes torch.compile, CUDA init, and other
                one-time overhead that would otherwise appear as an outlier
                and corrupt the running statistics. Default: True.
        """
        self._welford_mean: float = 0.0   # Running mean step time (seconds)
        self._welford_m2: float = 0.0     # Sum of squared differences
        self._count: int = 0               # Number of steps recorded
        self._last_time: float | None = None  # perf_counter of last tick
        self._skip_first: bool = skip_first
        self._first_skipped: bool = False

    def tick(self) -> None:
        """
        Record the end of one optimizer step and begin timing the next.

        Call this once per optimizer step (i.e., inside the
        ``if accelerator.sync_gradients:`` block, before
        ``progress_bar.update(1)``).
        """
        now = time.perf_counter()
        if self._last_time is not None:
            elapsed = now - self._last_time
            if self._skip_first and not self._first_skipped:
                self._first_skipped = True
                # Discard this interval—it includes torch.compile / init overhead
            else:
                self._add(elapsed)
        self._last_time = now

    def _add(self, value: float) -> None:
        """Update Welford statistics with a new step duration."""
        self._count += 1
        delta = value - self._welford_mean
        self._welford_mean += delta / self._count
        delta2 = value - self._welford_mean
        self._welford_m2 += delta * delta2

    @property
    def it_per_sec(self) -> float:
        """Running mean iterations per second (1 / mean step time)."""
        if self._welford_mean == 0.0:
            return 0.0
        return 1.0 / self._welford_mean

    @property
    def mean_step_time(self) -> float:
        """Running mean step time in seconds."""
        return self._welford_mean

    @property
    def display_rate(self) -> str:
        """Formatted rate string for tqdm postfix display.

        Shows ``it/s`` with two decimals when rate >= 1.0 (e.g. ``"12.34it/s"``),
        and ``s/it`` with two decimals when slower (e.g. ``"1.25s/it"``).
        """
        rate = self.it_per_sec
        if rate >= 1.0:
            return f"{rate:.2f}it/s"
        elif rate > 0.0:
            return f"{1.0 / rate:.2f}s/it"
        else:
            return "0.0it/s"

    @property
    def step_time_std(self) -> float:
        """Running standard deviation of step durations in seconds."""
        if self._count < 2:
            return 0.0
        return (self._welford_m2 / (self._count - 1)) ** 0.5

    @property
    def step_count(self) -> int:
        """Number of timed steps recorded."""
        return self._count