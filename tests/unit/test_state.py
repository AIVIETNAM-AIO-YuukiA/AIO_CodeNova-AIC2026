from indexing.state import JobState


def test_job_state_marks_and_skips_completed_items(tmp_path) -> None:
    state = JobState(tmp_path / "jobs.sqlite")

    assert not state.should_skip("video-1", "DISCOVER")

    state.mark("video-1", "DISCOVER", "COMPLETED")

    assert state.get_status("video-1", "DISCOVER") == "COMPLETED"
    assert state.should_skip("video-1", "DISCOVER")
    assert not state.should_skip("video-1", "DISCOVER", force=True)
