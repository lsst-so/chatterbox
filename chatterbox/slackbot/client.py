"""Slack delivery, with an offline mode that keeps the pipeline testable.

A bot token is required rather than an incoming webhook because incoming
webhooks cannot upload files and every ToO post carries plots. When no token is
configured, `SlackPoster` writes the payload and keeps the plot paths locally
instead of failing, so a replay can be checked end to end without Slack.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config

__all__ = ["SlackPoster", "PostedMessage"]

logger = logging.getLogger(__name__)


@dataclass
class PostedMessage:
    """Identifies a posted message so replies can be threaded onto it."""

    channel: str
    #: Slack message timestamp, used as ``thread_ts``. None in offline mode.
    ts: str | None = None
    #: True when nothing was actually sent to Slack.
    offline: bool = False
    uploaded: list[str] = field(default_factory=list)

    @property
    def can_thread(self) -> bool:
        """True when replies can be threaded onto this message."""
        return self.ts is not None


class SlackPoster:
    """Post ToO messages, files and threaded replies.

    Parameters
    ----------
    config : `Config`
        Configuration supplying the channel and the token environment variable.
    dry_run : `bool`
        Force offline mode even when a token is available.
    output_dir : `str` or `pathlib.Path`, optional
        Where offline payloads are written. Defaults to the work directory.
    """

    def __init__(self, config: Config, dry_run: bool = False, output_dir: str | Path | None = None) -> None:
        self.config = config
        self.dry_run = dry_run
        self.output_dir = (
            Path(output_dir).expanduser()
            if output_dir is not None
            else Path(config.paths.work_dir).expanduser() / "posts"
        )
        self._client = None
        self._crew_ids: dict[str, str] | None = None
        self._token = None if dry_run else config.slack_token
        if self._token is None and not dry_run:
            logger.warning(
                "%s is not set; running in offline mode and writing payloads to %s",
                config.slack.bot_token_env,
                self.output_dir,
            )

    @property
    def offline(self) -> bool:
        """True when no message will actually reach Slack."""
        return self._token is None

    def _get_client(self):
        if self._client is None:
            try:
                from slack_sdk import WebClient
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Posting to Slack requires slack-sdk (pip install slack-sdk)") from exc
            self._client = WebClient(token=self._token)
        return self._client

    def crew_ids(self) -> dict[str, str] | None:
        """Slack user group IDs for the shift crew, by handle.

        A mention only notifies when it carries the group's ID, and the
        configuration names groups by the handle people actually type. IDs
        given in ``slack.crew.ids`` are used as they are; the rest are looked
        up once with ``usergroups.list`` and kept for the life of the poster,
        so only the first alert pays for the call.

        Returns
        -------
        ids : `dict` [`str`, `str`] or None
            Handle to group ID. A group that could not be resolved is simply
            absent, which `chatterbox.slackbot.blocks.crew_blocks` reports in
            the post rather than letting an untagged message look tagged.
            None when nothing could be looked up because the poster is
            offline: the answer is then unknown rather than "no such group",
            and a dry run must not claim the real post would not notify.

        Notes
        -----
        Never raises: failing to find a group must not cost the alert.
        """
        if self._crew_ids is not None:
            return self._crew_ids

        crew = self.config.slack.crew
        ids = {h: i for h, i in crew.ids.items() if i}
        missing = [h for h in crew.handles() if h not in ids]
        if crew.enabled and missing and self.offline:
            return None
        if crew.enabled and missing:
            try:
                response = self._get_client().usergroups_list()
                by_handle = {g.get("handle"): g.get("id") for g in response.get("usergroups", [])}
                for handle in missing:
                    if by_handle.get(handle):
                        ids[handle] = by_handle[handle]
            except Exception as exc:
                logger.error(
                    "Could not look up the user groups %s (%s); the post will not notify them. "
                    "Give the bot the usergroups:read scope or set slack.crew.ids.",
                    ", ".join(missing),
                    exc,
                )
            still = [h for h in missing if h not in ids]
            if still:
                logger.warning("No Slack user group found for handle(s): %s", ", ".join(still))
        self._crew_ids = ids
        return ids

    def channel_for(self, is_test: bool) -> str:
        """Channel a message should go to.

        Test alerts go to ``slack.test_channel`` when one is configured, and
        otherwise to the main channel where they are still visibly marked.
        """
        if is_test and self.config.slack.test_channel:
            return self.config.slack.test_channel
        return self.config.slack.channel

    # ------------------------------------------------------------------ post

    def post(
        self,
        blocks: list[dict[str, Any]],
        text: str,
        is_test: bool = False,
        files: list[Path | str] | None = None,
        label: str = "post",
    ) -> PostedMessage:
        """Post a message and optionally attach files in its thread.

        Parameters
        ----------
        blocks : `list` [`dict`]
            Block Kit payload.
        text : `str`
            Fallback text, used for notifications.
        is_test : `bool`
            Route to the test channel when one is configured.
        files : `list`, optional
            Image paths to upload into the message's thread.
        label : `str`
            Name used for the offline payload file.

        Returns
        -------
        posted : `PostedMessage`
        """
        channel = self.channel_for(is_test)
        mention = " ".join(f"<{m}>" for m in self.config.slack.mention)
        if mention and not is_test:
            text = f"{mention} {text}"

        if self.offline:
            return self._write_offline(channel, blocks, text, files, label)

        client = self._get_client()
        response = client.chat_postMessage(
            channel=channel,
            text=text,
            blocks=blocks,
            username=self.config.slack.username,
            icon_emoji=self.config.slack.icon_emoji,
            unfurl_links=False,
            unfurl_media=False,
        )
        ts = response.get("ts")
        posted = PostedMessage(channel=response.get("channel", channel), ts=ts)
        logger.info("Posted to %s (ts=%s)", posted.channel, ts)

        for path in files or []:
            uploaded = self.upload(path, channel=posted.channel, thread_ts=ts)
            if uploaded:
                posted.uploaded.append(str(path))
        return posted

    def reply(
        self,
        parent: PostedMessage,
        blocks: list[dict[str, Any]],
        text: str,
        files: list[Path | str] | None = None,
        label: str = "reply",
    ) -> PostedMessage | None:
        """Reply in a message's thread.

        Returns
        -------
        posted : `PostedMessage` or None
            None when the parent cannot be threaded onto and offline output was
            written instead.
        """
        if self.offline or not parent.can_thread:
            self._write_offline(parent.channel, blocks, text, files, label)
            return None

        client = self._get_client()
        response = client.chat_postMessage(
            channel=parent.channel,
            thread_ts=parent.ts,
            text=text,
            blocks=blocks,
            username=self.config.slack.username,
            icon_emoji=self.config.slack.icon_emoji,
            unfurl_links=False,
        )
        posted = PostedMessage(channel=parent.channel, ts=response.get("ts"))
        for path in files or []:
            self.upload(path, channel=parent.channel, thread_ts=parent.ts)
        return posted

    def upload(self, path: Path | str, channel: str, thread_ts: str | None = None) -> bool:
        """Upload a file with ``files_upload_v2``.

        Returns
        -------
        ok : `bool`
            False when the file was missing or the upload failed. An upload
            failure is logged and swallowed: losing a plot should not lose the
            whole alert.
        """
        path = Path(path)
        if not path.is_file():
            logger.warning("Not uploading %s: file does not exist", path)
            return False
        if self.offline:
            logger.info("Offline: would upload %s to %s", path, channel)
            return False
        try:
            client = self._get_client()
            client.files_upload_v2(
                channel=channel,
                thread_ts=thread_ts,
                file=str(path),
                filename=path.name,
                title=path.stem.replace("_", " "),
            )
            logger.info("Uploaded %s", path.name)
            return True
        except Exception as exc:
            logger.error("Could not upload %s: %s", path, exc)
            return False

    # --------------------------------------------------------------- offline

    def _write_offline(
        self,
        channel: str,
        blocks: list[dict[str, Any]],
        text: str,
        files: list[Path | str] | None,
        label: str,
    ) -> PostedMessage:
        """Write what would have been posted, for inspection."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "channel": channel,
            "text": text,
            "blocks": blocks,
            "files": [str(f) for f in files or []],
        }
        path = self.output_dir / f"{label}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Offline: wrote %s (%d blocks, %d files)", path, len(blocks), len(files or []))
        return PostedMessage(channel=channel, ts=None, offline=True)


def render_blocks_as_text(blocks: list[dict[str, Any]]) -> str:
    """Flatten a Block Kit payload to readable text.

    Used by the CLI's dry run so a reviewer can read the message in a terminal
    without pasting JSON into Block Kit Builder.
    """
    lines: list[str] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "divider":
            lines.append("-" * 72)
        elif kind == "header":
            lines.append("")
            lines.append(f"=== {block['text']['text']} ===")
        elif kind == "section":
            lines.append(block["text"]["text"])
        elif kind == "context":
            for element in block.get("elements", []):
                lines.append(f"  ({element.get('text', '')})")
        lines.append("")
    return "\n".join(lines)
