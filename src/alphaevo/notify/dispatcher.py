"""Stdlib webhook/bot push notification dispatcher for Feishu, WeCom, DingTalk, Telegram, Discord, Slack & custom webhooks."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from alphaevo.core.config import NotifyConfig

NotifyChannel = Literal[
    "feishu",
    "wecom",
    "dingtalk",
    "telegram",
    "discord",
    "slack",
    "webhook",
]


class NotificationMessage(BaseModel):
    """Structured notification payload rendered to channel-specific formats."""

    title: str
    summary: str
    markdown_body: str
    severity: Literal["info", "success", "warning", "alert"] = "info"
    url: str | None = None
    tags: list[str] = Field(default_factory=list)


class NotificationDeliveryResult(BaseModel):
    """Outcome of delivering a notification to one channel."""

    channel: NotifyChannel
    success: bool
    status_code: int | None = None
    error: str | None = None
    dry_run: bool = False


_SEVERITY_PREFIX = {
    "info": "🧬",
    "success": "🚀",
    "warning": "⚠️",
    "alert": "🛑",
}


class NotificationDispatcher:
    """Zero-dependency stdlib dispatcher across 7 team & research channels."""

    def __init__(self, notify_config: NotifyConfig | None = None) -> None:
        self._cfg = notify_config

    def _get(self, field: str, env_var: str) -> str | None:
        val = getattr(self._cfg, field, None) if self._cfg is not None else None
        if isinstance(val, str) and val.strip():
            return val.strip()
        env_val = os.environ.get(env_var, "").strip()
        return env_val or None

    def configured_channels(self) -> list[NotifyChannel]:
        """Return ordered list of channels with credentials configured."""
        channels: list[NotifyChannel] = []
        if self._get("feishu_webhook", "ALPHAEVO_FEISHU_WEBHOOK"):
            channels.append("feishu")
        if self._get("wecom_webhook", "ALPHAEVO_WECOM_WEBHOOK"):
            channels.append("wecom")
        if self._get("dingtalk_webhook", "ALPHAEVO_DINGTALK_WEBHOOK"):
            channels.append("dingtalk")
        if self._get("telegram_bot_token", "ALPHAEVO_TELEGRAM_BOT_TOKEN") and self._get(
            "telegram_chat_id", "ALPHAEVO_TELEGRAM_CHAT_ID"
        ):
            channels.append("telegram")
        if self._get("discord_webhook", "ALPHAEVO_DISCORD_WEBHOOK"):
            channels.append("discord")
        if self._get("slack_webhook", "ALPHAEVO_SLACK_WEBHOOK"):
            channels.append("slack")
        if self._get("webhook_url", "ALPHAEVO_WEBHOOK_URL"):
            channels.append("webhook")
        return channels

    def build_payload(
        self,
        channel: NotifyChannel,
        message: NotificationMessage,
    ) -> tuple[str, dict[str, Any]]:
        """Build target URL and JSON body for *channel* without sending."""
        icon = _SEVERITY_PREFIX.get(message.severity, "🧬")
        header = f"{icon} {message.title}"
        body_text = message.markdown_body.strip() or message.summary
        if message.url:
            body_text = f"{body_text}\n\nDashboard: {message.url}"

        if channel == "feishu":
            url = self._require("feishu_webhook", "ALPHAEVO_FEISHU_WEBHOOK", channel)
            return url, {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {"tag": "plain_text", "content": header},
                        "template": "red" if message.severity == "alert" else "blue",
                    },
                    "elements": [
                        {"tag": "markdown", "content": body_text[:4000]},
                    ],
                },
            }

        if channel == "wecom":
            url = self._require("wecom_webhook", "ALPHAEVO_WECOM_WEBHOOK", channel)
            return url, {
                "msgtype": "markdown",
                "markdown": {"content": f"### {header}\n{body_text[:4000]}"},
            }

        if channel == "dingtalk":
            url = self._require("dingtalk_webhook", "ALPHAEVO_DINGTALK_WEBHOOK", channel)
            return url, {
                "msgtype": "markdown",
                "markdown": {"title": header, "text": f"### {header}\n\n{body_text[:4000]}"},
            }

        if channel == "telegram":
            token = self._require("telegram_bot_token", "ALPHAEVO_TELEGRAM_BOT_TOKEN", channel)
            chat_id = self._require("telegram_chat_id", "ALPHAEVO_TELEGRAM_CHAT_ID", channel)
            url = f"https://api.telegram.org/bot{quote(token, safe=':')}/sendMessage"
            return url, {
                "chat_id": chat_id,
                "text": f"{header}\n{message.summary}\n\n{body_text}"[:3900],
                "disable_web_page_preview": True,
            }

        if channel == "discord":
            url = self._require("discord_webhook", "ALPHAEVO_DISCORD_WEBHOOK", channel)
            return url, {
                "username": "AlphaEvo",
                "content": f"**{header}**\n{message.summary}",
                "embeds": [
                    {
                        "title": message.title[:250],
                        "description": body_text[:3800],
                        "color": 0xEF4444 if message.severity == "alert" else 0x10B981,
                    }
                ],
            }

        if channel == "slack":
            url = self._require("slack_webhook", "ALPHAEVO_SLACK_WEBHOOK", channel)
            return url, {
                "text": f"*{header}*\n{message.summary}\n\n{body_text[:3500]}",
            }

        url = self._require("webhook_url", "ALPHAEVO_WEBHOOK_URL", channel)
        return url, {
            "source": "alphaevo",
            "severity": message.severity,
            "title": message.title,
            "summary": message.summary,
            "markdown": body_text,
            "url": message.url,
            "tags": message.tags,
        }

    def send(
        self,
        message: NotificationMessage,
        *,
        channels: Sequence[NotifyChannel] | None = None,
        dry_run: bool = False,
        timeout: float = 10.0,
    ) -> list[NotificationDeliveryResult]:
        """Deliver *message* to selected or all configured channels."""
        targets = list(channels) if channels is not None else self.configured_channels()
        results: list[NotificationDeliveryResult] = []
        for channel in targets:
            try:
                url, payload = self.build_payload(channel, message)
            except ValueError as exc:
                results.append(
                    NotificationDeliveryResult(
                        channel=channel,
                        success=False,
                        error=str(exc),
                        dry_run=dry_run,
                    )
                )
                continue

            if dry_run:
                results.append(
                    NotificationDeliveryResult(
                        channel=channel,
                        success=True,
                        status_code=200,
                        dry_run=True,
                    )
                )
                continue

            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                    code = int(getattr(resp, "status", 200) or 200)
                results.append(
                    NotificationDeliveryResult(
                        channel=channel,
                        success=200 <= code < 300,
                        status_code=code,
                        dry_run=False,
                    )
                )
            except urllib.error.HTTPError as exc:
                results.append(
                    NotificationDeliveryResult(
                        channel=channel,
                        success=False,
                        status_code=int(exc.code),
                        error=str(exc),
                        dry_run=False,
                    )
                )
            except Exception as exc:
                results.append(
                    NotificationDeliveryResult(
                        channel=channel,
                        success=False,
                        error=str(exc),
                        dry_run=False,
                    )
                )
        return results

    def _require(self, field: str, env_var: str, channel: NotifyChannel) -> str:
        value = self._get(field, env_var)
        if not value:
            raise ValueError(f"Channel '{channel}' is not configured ({env_var} missing)")
        return value


__all__ = [
    "NotificationDeliveryResult",
    "NotificationDispatcher",
    "NotificationMessage",
    "NotifyChannel",
]
