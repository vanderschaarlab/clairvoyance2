from typing import NoReturn

NOT_IMPLEMENTED_MESSAGE = "Feature not yet implemented: $FEATURE"


def raise_not_implemented(feature: str) -> NoReturn:
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.replace("$FEATURE", feature))
