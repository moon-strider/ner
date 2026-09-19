from enum import Enum


class SpanPipelinePolicyOnUnavailable(str, Enum):
    DEGRADE = "degrade"
    FAIL = "fail"

    def __str__(self) -> str:
        return str(self.value)
