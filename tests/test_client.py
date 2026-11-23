"""Tests for `llama_index_meetstream.client.MeetStreamClient` against a mocked MeetStream API."""

from __future__ import annotations

import json

import httpx
import pytest

from llama_index_meetstream.client import MeetStreamClient, parse_transcript
from llama_index_meetstream.exceptions import MeetStreamAPIError, MeetStreamConnectionError

from .conftest import API_URL

BOT = "bot_1"


def _sent(api_mock) -> dict:
    return json.loads(api_mock.calls.last.request.content)


def _detail(api_mock, bot: str = BOT, **fields):
    details = {"BotID": bot, "Status": "Stopped", "transcript_id": "tr_1", **fields}
    return api_mock.get(f"/bots/{bot}/detail").mock(return_value=httpx.Response(200, json={"bot_details": details}))


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("MEETSTREAM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        MeetStreamClient(base_url=API_URL)


def test_reads_key_and_base_url_from_environment(monkeypatch):
    monkeypatch.setenv("MEETSTREAM_API_KEY", "from-env")
    monkeypatch.setenv("MEETSTREAM_BASE_URL", API_URL)
    client = MeetStreamClient()
    assert client._http.headers["Authorization"] == "Token from-env"
    assert str(client._http.base_url).rstrip("/") == API_URL


def test_defaults_to_meetstream_api(monkeypatch):
    monkeypatch.delenv("MEETSTREAM_BASE_URL", raising=False)
    assert str(MeetStreamClient(api_key="k")._http.base_url).rstrip("/") == "https://api.meetstream.ai/api/v1"


def test_dispatch_bot_uses_meetstream_transcription_by_default(api_mock, client):
    api_mock.post("/bots/create_bot").mock(
        return_value=httpx.Response(201, json={"bot_id": BOT, "transcript_id": "tr_1", "meeting_url": "https://meet.google.com/abc-defg-hij", "status": "Active"})
    )
    result = client.dispatch_bot("https://meet.google.com/abc-defg-hij", bot_name="Bot")

    assert (result.bot_id, result.meeting_id, result.status, result.transcript_id) == (BOT, BOT, "Active", "tr_1")
    body = _sent(api_mock)
    assert body["meeting_link"] == "https://meet.google.com/abc-defg-hij"
    assert body["recording_config"]["transcript"] == {"provider": {"meetstream": {"language": "auto"}}}
    assert api_mock.calls.last.request.headers["Authorization"] == "Token ms_test_key"


def test_dispatch_bot_with_a_chosen_provider(api_mock, client):
    api_mock.post("/bots/create_bot").mock(return_value=httpx.Response(201, json={"bot_id": BOT, "status": "Active"}))
    client.dispatch_bot("https://zoom.us/j/1", transcription_provider="deepgram", transcription_options={"model": "nova-3"})
    assert _sent(api_mock)["recording_config"]["transcript"] == {"provider": {"deepgram": {"model": "nova-3"}}}


def test_client_level_provider_and_no_transcription(api_mock):
    api_mock.post("/bots/create_bot").mock(return_value=httpx.Response(201, json={"bot_id": BOT, "status": "Active"}))
    MeetStreamClient(api_key="k", base_url=API_URL, transcription_provider="assemblyai").dispatch_bot("https://zoom.us/j/1")
    assert _sent(api_mock)["recording_config"]["transcript"] == {"provider": {"assemblyai": {}}}

    MeetStreamClient(api_key="k", base_url=API_URL, transcription_provider=None).dispatch_bot("https://zoom.us/j/1")
    assert "recording_config" not in _sent(api_mock)


def test_dispatch_bot_is_not_retried_and_maps_invalid_link(api_mock, client):
    route = api_mock.post("/bots/create_bot").mock(return_value=httpx.Response(400, json={"message": "Invalid meeting link"}))
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.dispatch_bot("not-a-link")
    assert excinfo.value.code == "invalid_meeting_url" and str(excinfo.value) == "Invalid meeting link"

    route.mock(return_value=httpx.Response(503))
    with pytest.raises(MeetStreamAPIError):
        client.dispatch_bot("https://zoom.us/j/1")
    assert route.call_count == 2  # one per call: create is never repeated


def test_get_meeting_reads_bot_details(api_mock, client):
    _detail(
        api_mock,
        Status="InMeeting",
        Platform="Zoom",
        MeetingLink="https://us02web.zoom.us/j/1",
        StartTime="2026-03-19T13:57:52.810Z",
        EndTime="2026-03-19T14:08:08Z",
        custom_attributes={"team": "x"},
    )
    meeting = client.get_meeting(BOT)
    assert (meeting.status, meeting.platform, meeting.meeting_url, meeting.custom_attributes) == (
        "InMeeting", "zoom", "https://us02web.zoom.us/j/1", {"team": "x"}
    )
    assert (meeting.started_at, meeting.ended_at, meeting.transcript_id, meeting.title) == (
        "2026-03-19T13:57:52.810Z", "2026-03-19T14:08:08Z", "tr_1", None
    )


def test_get_meeting_maps_detail_platform_without_a_link(api_mock, client):
    _detail(api_mock, Platform="GMeet")
    assert client.get_meeting(BOT).platform == "google_meet"


def test_get_transcript_segment_list_format(api_mock, client):
    _detail(api_mock)
    api_mock.get("/transcript/tr_1/get_transcript").mock(
        return_value=httpx.Response(200, json=[
            {"speaker": "Alex", "transcript": "Hello there.", "start_time": 1.0, "end_time": 2.0, "words": []},
            {"speaker": 1, "transcript": "Hi.", "start_time": 2.5, "end_time": 3.0},
        ])
    )
    transcript = client.get_transcript(BOT)
    assert transcript.meeting_id == BOT
    assert [(s.speaker, s.text, s.start_time) for s in transcript.segments] == [("Alex", "Hello there.", 1.0), ("Speaker 2", "Hi.", 2.5)]


def test_get_transcript_uses_the_transcript_id_from_create_bot(api_mock, client):
    api_mock.post("/bots/create_bot").mock(return_value=httpx.Response(201, json={"bot_id": BOT, "transcript_id": "tr_9"}))
    detail = _detail(api_mock)
    api_mock.get("/transcript/tr_9/get_transcript").mock(return_value=httpx.Response(200, json=[{"transcript": "Hi"}]))
    client.dispatch_bot("https://meet.google.com/abc-defg-hij")
    assert client.get_transcript(BOT).segments[0].text == "Hi"
    assert not detail.called


def test_get_transcript_without_a_transcript_id(api_mock, client):
    _detail(api_mock, transcript_id=None)
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.get_transcript(BOT)
    assert excinfo.value.code == "transcript_unavailable"


def test_get_transcript_for_an_unknown_bot(api_mock, client):
    api_mock.get(f"/bots/{BOT}/detail").mock(return_value=httpx.Response(404, json={"message": "Bot not found"}))
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.get_transcript(BOT)
    assert excinfo.value.code == "meeting_not_found"


def test_get_transcript_participant_keyed_format():
    segments = parse_transcript([
        {"participant": {"id": 100, "name": "Ana"}, "language_code": "en",
         "words": [{"text": "Hello", "start_timestamp": {"relative": 1.0}, "end_timestamp": {"relative": 1.4}},
                   {"text": "all.", "start_timestamp": {"relative": 1.5}, "end_timestamp": {"relative": 1.9}}]},
        {"participant": {"name": "Ben"}, "words": []},
    ])
    assert [(s.speaker, s.text, s.start_time, s.end_time) for s in segments] == [("Ana", "Hello all.", 1.0, 1.9)]


def test_unrecognised_transcript_body():
    with pytest.raises(MeetStreamAPIError) as excinfo:
        parse_transcript({"id": "x", "status": "done"})
    assert excinfo.value.code == "transcript_format_error"


@pytest.mark.parametrize(
    ("status", "code"),
    [(202, "transcript_not_ready"), (409, "transcript_unavailable"), (422, "transcript_unavailable"),
     (404, "meeting_not_found"), (403, "forbidden"), (401, "authentication_failed")],
)
def test_transcript_status_codes(api_mock, client, status, code):
    _detail(api_mock)
    api_mock.get("/transcript/tr_1/get_transcript").mock(return_value=httpx.Response(status, json={"message": "m"}))
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.get_transcript(BOT)
    assert (excinfo.value.code, excinfo.value.status_code) == (code, status)


def test_transient_5xx_is_retried_for_reads(api_mock, client):
    route = api_mock.get(f"/bots/{BOT}/detail").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"bot_details": {"BotID": BOT, "Status": "Stopped"}})]
    )
    assert client.get_meeting(BOT).status == "Stopped"
    assert route.call_count == 2


def test_persistent_5xx_raises_upstream_error(api_mock, client):
    api_mock.get(f"/bots/{BOT}/detail").mock(return_value=httpx.Response(500))
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.get_meeting(BOT)
    assert excinfo.value.code == "upstream_error"


def test_chat_message_is_not_repeated_after_a_500(api_mock, client):
    route = api_mock.post(f"/bots/{BOT}/send_message").mock(return_value=httpx.Response(500))
    with pytest.raises(MeetStreamAPIError):
        client.send_chat_message(BOT, "hello")
    assert route.call_count == 1


def test_chat_message_is_retried_while_the_command_channel_connects(api_mock, client):
    route = api_mock.post(f"/bots/{BOT}/send_message").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"status": "accepted"})]
    )
    assert client.send_chat_message(BOT, "hello").detail == "accepted"
    assert route.call_count == 2


def test_create_bot_without_bot_id_is_an_error(api_mock, client):
    api_mock.post("/bots/create_bot").mock(return_value=httpx.Response(201, json={"status": "Active"}))
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.dispatch_bot("https://zoom.us/j/1")
    assert excinfo.value.code == "upstream_error"


def test_bot_ids_are_url_encoded(api_mock, client):
    route = api_mock.get("/bots/abc%2Fremove_bot%23/detail").mock(
        return_value=httpx.Response(200, json={"bot_details": {"BotID": "x", "Status": "Stopped"}})
    )
    removed = api_mock.get("/bots/abc/remove_bot").mock(return_value=httpx.Response(200, json={}))
    client.get_meeting("abc/remove_bot#")
    assert route.called and not removed.called


def test_non_json_transcript_body_is_a_format_error(api_mock, client):
    _detail(api_mock)
    api_mock.get("/transcript/tr_1/get_transcript").mock(return_value=httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.get_transcript(BOT)
    assert excinfo.value.code == "transcript_format_error"


def test_network_failure_raises_connection_error():
    client = MeetStreamClient(api_key="k", base_url="http://this-host-does-not-exist.invalid", timeout=1.0)
    with pytest.raises(MeetStreamConnectionError):
        client.get_meeting(BOT)


def test_in_meeting_actions(api_mock, client):
    api_mock.post(f"/bots/{BOT}/send_message").mock(
        return_value=httpx.Response(200, json={"status": "accepted", "bot_id": BOT, "command": "sendmsg"})
    )
    assert client.send_chat_message(BOT, "hello").detail == "accepted"
    assert _sent(api_mock) == {"message": "hello"}

    api_mock.post(f"/bots/{BOT}/send_image").mock(return_value=httpx.Response(200, json={"status": "accepted"}))
    client.send_image(BOT, "https://example.com/a.png", 10)
    assert _sent(api_mock) == {"img_url": "https://example.com/a.png", "display_duration": 10}

    api_mock.get(f"/bots/{BOT}/remove_bot").mock(return_value=httpx.Response(202, json={"bot_id": BOT, "status": "stop_requested"}))
    assert client.leave_meeting(BOT).detail == "stop_requested"


def test_action_when_bot_is_not_in_a_meeting(api_mock, client):
    api_mock.post(f"/bots/{BOT}/send_message").mock(return_value=httpx.Response(409, json={"message": "The bot is not currently in a meeting."}))
    with pytest.raises(MeetStreamAPIError) as excinfo:
        client.send_chat_message(BOT, "hello")
    assert excinfo.value.code == "meeting_not_active"
