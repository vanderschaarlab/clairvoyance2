from typing import NoReturn, Optional

import numpy as np

from . import DEFAULT_PADDING_INDICATOR


class ToNumpyMixin:
    @staticmethod
    def raise_arg_error(additional_text: str) -> NoReturn:
        raise ValueError(f"Incorrect arguments / keyword arguments passed to `to_numpy`. {additional_text}.")

    def to_numpy(self, *args, **kwargs) -> np.ndarray:
        padding_indicator: Optional[float] = None
        max_len: Optional[int] = None
        if args:
            if len(args) > 2:
                self.raise_arg_error("No more than two arguments can be provided")
            else:
                if len(args) == 2:
                    padding_indicator, max_len = args
                else:
                    padding_indicator = args[0]
        if kwargs:
            if "padding_indicator" in kwargs:
                if padding_indicator is None:
                    padding_indicator = kwargs.pop("padding_indicator")
                else:
                    self.raise_arg_error("`padding_indicator` provided via both arguments and keyword arguments")
            if "max_len" in kwargs:
                if max_len is None:
                    max_len = kwargs.pop("max_len")
                else:
                    self.raise_arg_error("`max_len` provided via both arguments and keyword arguments")
            if len(kwargs) > 0:
                self.raise_arg_error("Too many or unknown keyword arguments provided")
        no_args = padding_indicator is None and max_len is None
        at_least_padding_indicator_defined = padding_indicator is not None
        if no_args:
            try:
                return self._to_numpy_static()
            except NotImplementedError:
                return self._to_numpy_time_series(padding_indicator=DEFAULT_PADDING_INDICATOR, max_len=None)
                # ^ Call with default arguments.
        elif at_least_padding_indicator_defined:
            if not isinstance(padding_indicator, float):
                raise TypeError("`padding_indicator` must be a float")
            if not (isinstance(max_len, int) or max_len is None):
                raise TypeError("`max_len` must be an int or None")
            return self._to_numpy_time_series(padding_indicator=padding_indicator, max_len=max_len)
        else:  # pragma: no cover (this should not be reached)
            self.raise_arg_error("Expected either no arguments, or: padding_indicator[, max_len]")

    def _to_numpy_time_series(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        raise NotImplementedError("`_to_numpy_time_series` method not implemented")

    def _to_numpy_static(self) -> np.ndarray:
        raise NotImplementedError("`_to_numpy_static` method not implemented")
