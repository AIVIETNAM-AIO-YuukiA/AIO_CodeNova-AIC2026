from modules.asr.gipformer import _MAX_SEGMENT_SECONDS, _group_by_words, _speech_windows


class _FakeVad:
    """Stand-in for SileroVad — returns fixed segments without loading torch.hub."""

    def __init__(self, segments):
        self._segments = segments

    def speech_segments(self, wav_path):
        return self._segments


def test_speech_windows_passes_through_short_segments():
    vad = _FakeVad([(0.0, 5.0), (10.0, 12.0)])
    windows = _speech_windows(vad, "unused.wav")
    assert windows == [(0.0, 5.0), (10.0, 12.0)]


def test_speech_windows_caps_long_segment():
    """A VAD segment longer than _MAX_SEGMENT_SECONDS is force-split into
    multiple windows so gipformer (non-causal) never decodes an unbounded
    block of continuous speech."""
    long_end = _MAX_SEGMENT_SECONDS * 2.5
    vad = _FakeVad([(0.0, long_end)])
    windows = _speech_windows(vad, "unused.wav")

    assert len(windows) == 3
    assert windows[0] == (0.0, _MAX_SEGMENT_SECONDS)
    assert windows[1] == (_MAX_SEGMENT_SECONDS, _MAX_SEGMENT_SECONDS * 2)
    assert windows[2] == (_MAX_SEGMENT_SECONDS * 2, long_end)


def test_group_by_words_splits_at_target():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "one two three"},
        {"start": 2.0, "end": 4.0, "text": "four five six"},
        {"start": 4.0, "end": 6.0, "text": "seven eight nine"},
    ]
    chunks = _group_by_words(segments, target_words=6)

    # segments 1+2 (3+3=6 words) fit within target=6; adding segment 3 would
    # push the running count to 9 > 6, so the chunk closes before segment 3.
    assert len(chunks) == 2
    assert chunks[0]["text"] == "one two three four five six"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 4.0
    assert chunks[1]["text"] == "seven eight nine"
    assert chunks[1]["start"] == 4.0
    assert chunks[1]["end"] == 6.0


def test_group_by_words_single_segment_over_target_stays_whole():
    """A single segment longer than target_words is never split mid-segment —
    grouping only ever combines whole VAD segments, never cuts inside one."""
    segments = [{"start": 0.0, "end": 10.0, "text": " ".join(["word"] * 200)}]
    chunks = _group_by_words(segments, target_words=150)

    assert len(chunks) == 1
    assert chunks[0]["text"].split() == ["word"] * 200


def test_group_by_words_empty_input():
    assert _group_by_words([], target_words=150) == []


def test_group_by_words_skips_never_over_before_first():
    """The first segment always starts a chunk, even if it alone exceeds
    target_words — the over-target check only fires once a chunk is non-empty."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": " ".join(["word"] * 300)},
        {"start": 1.0, "end": 2.0, "text": "short tail"},
    ]
    chunks = _group_by_words(segments, target_words=150)

    assert len(chunks) == 2
    assert len(chunks[0]["text"].split()) == 300
    assert chunks[1]["text"] == "short tail"
