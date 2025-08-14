from contextlib import contextmanager
from time import perf_counter

class StepTimer:
    def __init__(self):
        self.durations = {}   # {label: seconds}
        self._stack = []      # for nested timing if needed

    @contextmanager
    def timeit(self, label: str):
        start = perf_counter()
        self._stack.append(label)
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.durations[label] = self.durations.get(label, 0.0) + elapsed
            self._stack.pop()

    def print_summary(self, title: str = "⏱️ Runtime summary"):
        width = max((len(k) for k in self.durations), default=10)
        print("\n" + title)
        print("-" * (width + 14))
        for k, v in self.durations.items():
            print(f"{k.ljust(width)} : {v:8.3f}s")
