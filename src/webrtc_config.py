"""WebRTC ICE server configuration for hosted camera streaming."""

from collections.abc import Mapping
from typing import Any


def build_rtc_configuration(
    settings: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    """Build ICE settings and report whether TURN settings are ready."""
    ice_servers: list[dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]

    turn_urls = [
        url.strip()
        for url in settings.get("turn_server_url", "").split(",")
        if url.strip()
    ]
    username = settings.get("turn_username", "").strip()
    credential = settings.get("turn_credential", "").strip()

    has_turn_settings = bool(turn_urls or username or credential)
    if not has_turn_settings:
        return {"iceServers": ice_servers}, "missing"

    if not turn_urls or not username or not credential:
        return {"iceServers": ice_servers}, "incomplete"

    if any(not url.startswith(("turn:", "turns:")) for url in turn_urls):
        return {"iceServers": ice_servers}, "invalid"

    ice_servers.append({
        "urls": turn_urls,
        "username": username,
        "credential": credential,
    })
    return {"iceServers": ice_servers}, "configured"
