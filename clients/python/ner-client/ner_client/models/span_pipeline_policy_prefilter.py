from enum import Enum


class SpanPipelinePolicyPrefilter(str, Enum):
    AUTO = "auto"
    CAPITALIZED = "capitalized"
    NAME_LIKE = "name_like"

    def __str__(self) -> str:
        return str(self.value)
