import os
import time
import uuid
from threading import Lock
from src.utils.logger import setup_logger

class SessionManager:
    def __init__(self, temp_dir, timeout_seconds):
        self.temp_dir = temp_dir
        self.timeout_seconds = timeout_seconds
        self.sessions = {}  # {session_id: {"files": [], "last_active": timestamp, "current_query": str, "current_results": list}}
        self.lock = Lock()
        self.logger = setup_logger()
        os.makedirs(temp_dir, exist_ok=True)

    def create_session(self):
        """Create a new session with a unique ID."""
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                "files": [],
                "last_active": time.time(),
                "current_query": None,
                "current_results": []
            }
        self.logger.info(f"Created session: {session_id}")
        return session_id

    def add_temp_file(self, session_id, file_path):
        """Add a temporary file to a session."""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["files"].append(file_path)
                self.sessions[session_id]["last_active"] = time.time()
                self.logger.info(f"Added temp file {file_path} to session {session_id}")

    def update_session_data(self, session_id, current_query=None, current_results=None):
        """Update query and results for a session."""
        with self.lock:
            if session_id in self.sessions:
                if current_query is not None:
                    self.sessions[session_id]["current_query"] = current_query
                if current_results is not None:
                    self.sessions[session_id]["current_results"] = current_results
                self.sessions[session_id]["last_active"] = time.time()
                self.logger.info(f"Updated session {session_id}: query={current_query}, results_len={len(current_results or [])}")

    def get_session_data(self, session_id):
        """Get query and results for a session."""
        with self.lock:
            if session_id in self.sessions:
                return (
                    self.sessions[session_id]["current_query"],
                    self.sessions[session_id]["current_results"]
                )
            return None, []

    def cleanup_session(self, session_id):
        """Delete all temporary files and session data."""
        with self.lock:
            if session_id in self.sessions:
                for file_path in self.sessions[session_id]["files"]:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            self.logger.info(f"Deleted temp file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete {file_path}: {e}")
                del self.sessions[session_id]
                self.logger.info(f"Cleaned up session: {session_id}")

    def cleanup_expired_sessions(self):
        """Clean up sessions that have timed out."""
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                sid for sid, data in self.sessions.items()
                if current_time - data["last_active"] > self.timeout_seconds
            ]
            for session_id in expired_sessions:
                self.cleanup_session(session_id)
        self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return f"Cleaned up {len(expired_sessions)} expired sessions"

    def update_session_activity(self, session_id):
        """Update the last active time for a session."""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["last_active"] = time.time()
                self.logger.debug(f"Updated activity for session: {session_id}")