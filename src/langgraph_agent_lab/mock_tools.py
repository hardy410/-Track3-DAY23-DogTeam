"""Deterministic mock integrations for an offline classroom demo."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any


def _load_orders() -> dict[str, dict[str, Any]]:
    """Load the packaged order fixtures."""
    resource = files("langgraph_agent_lab.data").joinpath("mock_orders.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def extract_order_id(query: str) -> str | None:
    """Extract and normalize the first explicit order number in a query."""
    match = re.search(r"#\s*(\d{5,})", query)
    return match.group(1) if match else None


def lookup_order(query: str) -> dict[str, Any] | None:
    """Return only the record whose key exactly matches the requested order number."""
    order_id = extract_order_id(query)
    return _load_orders().get(order_id) if order_id else None


def is_order_lookup_without_id(query: str) -> bool:
    """Detect an order-tracking request that cannot yet be executed precisely."""
    normalized = query.casefold()
    order_markers = ("đơn", "order", "giao hàng", "delivery")
    lookup_markers = (
        "kiểm tra",
        "tra cứu",
        "theo dõi",
        "đang ở",
        "giao tới",
        "chưa thấy giao",
        "track",
        "where",
    )
    return (
        extract_order_id(query) is None
        and any(marker in normalized for marker in order_markers)
        and any(marker in normalized for marker in lookup_markers)
    )


def format_tracking_result(order: dict[str, Any]) -> str:
    """Serialize a fixture into explicit, LLM-groundable Vietnamese context."""
    history = "; ".join(str(item) for item in order.get("tracking_history", []))
    support_flags = "; ".join(str(item) for item in order.get("support_flags", []))
    return (
        f"SUCCESS: Đã khớp chính xác khóa {order['order_id']} với bản ghi "
        f"{order['record_id']} trong mock_orders.json. "
        f"Người bán: {order['merchant']}. Tạo lúc: {order['created_at']}. "
        f"Trạng thái: {order['status']}. "
        f"Đơn vị vận chuyển: {order['carrier']}. "
        f"Dịch vụ: {order['service_level']}. "
        f"Tuyến giao: {order['origin']} → {order['destination']}. "
        f"Vị trí hiện tại: {order['current_location']}. "
        f"Cập nhật lúc: {order['last_updated']}. "
        f"Dự kiến giao: {order['estimated_delivery']}. "
        f"Kiện hàng: {order['package']}. "
        f"Thanh toán: {order['payment']}. "
        f"Số lần giao không thành công: {order['delivery_attempts']}. "
        f"Lưu ý giao hàng: {order['delivery_note']}. "
        f"Cờ hỗ trợ: {support_flags}. "
        f"Lịch sử: {history}."
    )
