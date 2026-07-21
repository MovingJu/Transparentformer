from collections.abc import Callable
from typing import Protocol


class BenchmarkFixture(Protocol):
    def __call__[**P, R](
        self, func: Callable[..., R], *args: object, **kwargs: object
    ) -> R: ...
