"""
Wrapper giữ API cũ cho CLI/GP:
- run_episode: mô phỏng event-driven (định nghĩa trong sim_core).
- validate_routes: kiểm tra ràng buộc tuyến (định nghĩa trong sim_validate).

Logic chính được tách ở sim_core.py và sim_validate.py để dễ bảo trì.
"""

from .sim_core import run_episode
from .sim_validate import validate_routes

__all__ = ["run_episode", "validate_routes"]
