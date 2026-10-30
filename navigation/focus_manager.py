"""
Focus Manager tracking selected items across mouse and gesture input methods.
Maintains focus state independently of activation.
"""

from typing import List, Optional, Union, Callable


class FocusManager:
    """Tracks and updates the focused element in active menus."""

    def __init__(self, items: Optional[List[str]] = None):
        self._items: List[str] = items or []
        self._current_index: int = 0
        self._listeners: List[Callable[[int, str], None]] = []

    @property
    def items(self) -> List[str]:
        return list(self._items)

    def set_items(self, items: List[str], reset_index: bool = True):
        """Sets the active navigation items."""
        self._items = list(items)
        if reset_index or not self._items:
            self._current_index = 0
        else:
            self._current_index = max(0, min(self._current_index, len(self._items) - 1))
        self._notify()

    def get_focused_index(self) -> int:
        return self._current_index

    def get_focused_item(self) -> Optional[str]:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index]
        return None

    def move_next(self) -> Optional[str]:
        """Moves focus to the next item (cycling). Does NOT activate."""
        if not self._items:
            return None
        self._current_index = (self._current_index + 1) % len(self._items)
        self._notify()
        return self.get_focused_item()

    def move_prev(self) -> Optional[str]:
        """Moves focus to the previous item (cycling). Does NOT activate."""
        if not self._items:
            return None
        self._current_index = (self._current_index - 1) % len(self._items)
        self._notify()
        return self.get_focused_item()

    def set_focus(self, target: Union[int, str]) -> Optional[str]:
        """Sets focus by index or item name without activating."""
        if isinstance(target, int):
            if 0 <= target < len(self._items):
                self._current_index = target
                self._notify()
        elif isinstance(target, str):
            if target in self._items:
                self._current_index = self._items.index(target)
                self._notify()
        return self.get_focused_item()

    def add_listener(self, callback: Callable[[int, str], None]):
        """Registers a callback(index, item_name) on focus change."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[int, str], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self):
        item = self.get_focused_item() or ""
        for listener in self._listeners:
            try:
                listener(self._current_index, item)
            except Exception as e:
                print(f"[FocusManager] Listener error: {e}")
