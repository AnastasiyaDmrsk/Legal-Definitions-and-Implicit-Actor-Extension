from typing import Iterable, Tuple, List


class TokenList(Iterable[str]):
    """
    Simply list proxy for recording the position in which items were inserted
    """

    def __init__(self, items: Iterable[str]):
        self._items = list(items)
        self._positions = []

    def __iter__(self):
        return self._items.__iter__()

    def __setitem__(self, key, new_value):
        self._items.__setitem__(key, new_value)
        self._positions.append(key)

    def join(self) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Joins the items and records the position of insertions
        """

        spans = [
            (start := sum(len(x) for x in self._items[:p]), start + len(self._items[p]))
            for p in self._positions
        ]
        return "".join(self._items), spans

    def positions(self):
        """
        Returns the positions of the all insertions
        """
        return self._positions
