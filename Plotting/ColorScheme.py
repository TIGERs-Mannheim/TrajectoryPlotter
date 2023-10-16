from abc import ABC, abstractmethod
from typing import Dict


class ColorScheme(ABC):
    @property
    @abstractmethod
    def a_abs(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def a_x(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def a_y(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def a_optimal(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def a_chosen(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def total(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def passed(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def limit(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def pos(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def vel(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def acc(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def target(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def marking(self) -> Dict[str, str]:
        raise NotImplementedError


class CSNormal(ColorScheme):

    @property
    def a_abs(self) -> Dict[str, str]:
        return {"color": "orange", "ls": "-"}

    @property
    def a_x(self) -> Dict[str, str]:
        return {"color": "blue", "ls": "-"}

    @property
    def a_y(self) -> Dict[str, str]:
        return {"color": "cyan", "ls": "-"}

    @property
    def a_optimal(self) -> Dict[str, str]:
        return {"color": "green", "ls": "-"}

    @property
    def a_chosen(self) -> Dict[str, str]:
        return {"color": "red", "ls": "--"}

    @property
    def total(self) -> Dict[str, str]:
        return {"color": "gray", "ls": "-"}

    @property
    def passed(self) -> Dict[str, str]:
        return {"color": "blue", "ls": "-"}

    @property
    def limit(self) -> Dict[str, str]:
        return {"color": "black", "ls": "-"}

    @property
    def pos(self) -> Dict[str, str]:
        return {"color": "green", "ls": "-"}

    @property
    def vel(self) -> Dict[str, str]:
        return {"color": "red", "ls": "-"}

    @property
    def acc(self) -> Dict[str, str]:
        return {"color": "blue", "ls": "-"}

    @property
    def target(self) -> Dict[str, str]:
        return {"color": "red", "ls": "-"}

    @property
    def marking(self) -> Dict[str, str]:
        return {"color": "gray", "ls": "-"}


class CSBlackAndWhite(ColorScheme):
    @property
    def a_abs(self) -> Dict[str, str]:
        return {"color": "0", "ls": "-"}

    @property
    def a_x(self) -> Dict[str, str]:
        return {"color": "0", "ls": "--"}

    @property
    def a_y(self) -> Dict[str, str]:
        return {"color": "0", "ls": "-."}

    @property
    def a_optimal(self) -> Dict[str, str]:
        return {"color": "0.5", "ls": "-"}

    @property
    def a_chosen(self) -> Dict[str, str]:
        return {"color": "0", "ls": ":"}

    @property
    def total(self) -> Dict[str, str]:
        return {"color": "0.5", "ls": "-"}

    @property
    def passed(self) -> Dict[str, str]:
        return {"color": "0", "ls": "-"}

    @property
    def limit(self) -> Dict[str, str]:
        return {"color": "0", "ls": "--"}

    @property
    def pos(self) -> Dict[str, str]:
        return {"color": "0.3", "ls": "-"}

    @property
    def vel(self) -> Dict[str, str]:
        return {"color": "0.3", "ls": "-"}

    @property
    def acc(self) -> Dict[str, str]:
        return {"color": "0.3", "ls": "-"}

    @property
    def target(self) -> Dict[str, str]:
        return {"color": "0", "ls": "-"}

    @property
    def marking(self) -> Dict[str, str]:
        return {"color": "0.5", "ls": "--"}
