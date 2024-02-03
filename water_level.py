# Water Level Model
import copy
import numpy as np
from typing import Any, List


class Series:
    def __init__(self, name: str, data: list) -> None:
        self.__name = name
        self.__data = copy.deepcopy(data)
        self.__next_id = 0
        self.__max_len = len(self.__data)

    def next(self) -> float or None:
        if self.__next_id < self.__max_len:
            result = self.__data[self.__next_id]
            self.__next_id += 1
            return result
        return None

    def name(self) -> str:
        return self.__name


class Pool:
    def __init__(self) -> None:
        self.__series_dict = {}

    def register_series(self, series: Series) -> None:
        if series.name() in self.__series_dict:
            raise ValueError(f"Series(name={series.name()}) is duplicate")
        self.__series_dict[series.name()] = series

    def next_from_series(self, name: str) -> float or None:
        if name not in self.__series_dict:
            print(f"Warning: Series(name={name}) does not exist")
            return None
        return self.__series_dict[name].next()


class Factor:
    def __init__(self, model: Any, data_pool: Pool, deps: List[str]) -> None:
        assert model is not None, 'trained model must not be None'

        self.__model = model
        self.__data_pool = data_pool
        self.__deps = deps

    def __call__(self, *args, **kwargs) -> Any:
        ins = np.array([self.__data_pool.next_from_series(name) for name in self.__deps])
        return self.__model.predict(ins)


# (inflow - outflow) / surface_area => delta_level
class LakeModel:
    def __init__(self, inflows: List[Factor], outflows: List[Factor], initial_level: float, surface_area: float):
        # assume lake has vertical walls, keeping surface area the same for certain height
        self.__inflow_factors = inflows
        self.__outflow_factors = outflows
        self.__initial_level = initial_level
        self.__surface_area = surface_area

    def get_water_level(self) -> float:
        delta = 0.0
        for model in self.__inflow_factors:
            delta += model()
        for model in self.__outflow_factors:
            delta += model()
        return self.__initial_level + delta / self.__surface_area
