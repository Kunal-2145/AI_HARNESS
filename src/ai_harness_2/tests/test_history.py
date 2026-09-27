import tempfile
import unittest
from pathlib import Path

from ai_harness_2.state.history import ConversationStore, MAX_CONTEXT_CHARS


class ConversationStoreTests(unittest.TestCase):
    def test_session_history_persists_and_resumes_per_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            database_path = root / "state" / "history.sqlite3"

            store = ConversationStore(database_path)
            session_id, resumed = store.resume_or_create_session(workspace)
            self.assertFalse(resumed)
            store.save_exchange(session_id, "Remember this", "I will remember it.")

            reopened_store = ConversationStore(database_path)
            resumed_id, resumed = reopened_store.resume_or_create_session(workspace)
            self.assertTrue(resumed)
            self.assertEqual(resumed_id, session_id)
            self.assertEqual(
                reopened_store.recent_messages(resumed_id),
                [
                    {"role": "user", "content": "Remember this"},
                    {"role": "assistant", "content": "I will remember it."},
                ],
            )

    def test_new_session_has_no_previous_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = ConversationStore(root / "history.sqlite3")
            session_id = store.create_session(root)
            store.save_exchange(session_id, "Old request", "Old answer")
            fresh_session_id = store.create_session(root)
            self.assertEqual(store.recent_messages(fresh_session_id), [])

    def test_context_is_limited_by_total_character_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = ConversationStore(root / "history.sqlite3")
            session_id = store.create_session(root)
            store.save_exchange(session_id, "Question", "A" * (MAX_CONTEXT_CHARS * 2))
            messages = store.recent_messages(session_id)
            self.assertLessEqual(
                sum(len(message["content"]) for message in messages),
                MAX_CONTEXT_CHARS,
            )
            self.assertIn("earlier content truncated", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()