import json
from typing import Dict, List
from datetime import datetime
import os

class LearningAnalyticsService:
    def __init__(self):
        self.progress_file = "user_progress.json"
        self._initialize_progress()

    def _initialize_progress(self):
        """Initialize or load user progress data"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress_data = json.load(f)
        else:
            self.progress_data = {
                "topics": {},
                "quiz_scores": {},
                "feedback_history": [],
                "last_updated": datetime.now().isoformat()
            }

    def save_progress(self):
        """Save progress data to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_data, f, indent=4)

    def add_feedback(self, topic: str, feedback_type: str):
        """Record user feedback"""
        self.progress_data["feedback_history"].append({
            "topic": topic,
            "type": feedback_type,
            "timestamp": datetime.now().isoformat()
        })
        self.save_progress()

    def record_quiz_score(self, topic: str, score: float):
        """Record quiz score for a topic"""
        self.progress_data["quiz_scores"][topic] = score
        self.save_progress()

    def get_weak_topics(self) -> List[str]:
        """Identify weak topics based on quiz scores and feedback"""
        weak_topics = []
        
        # Check quiz scores
        for topic, score in self.progress_data["quiz_scores"].items():
            if score < 0.7:  # Threshold for weak performance
                weak_topics.append(topic)

        # Check feedback history
        for feedback in self.progress_data["feedback_history"]:
            if feedback["type"] == "too_slow":
                weak_topics.append(feedback["topic"])

        return list(set(weak_topics))

    def get_progress_stats(self) -> Dict:
        """Get overall progress statistics"""
        total_topics = len(self.progress_data["topics"])
        completed_topics = len([t for t in self.progress_data["topics"] 
                               if self.progress_data["topics"][t].get("completed", False)])
        
        return {
            "total_topics": total_topics,
            "completed_topics": completed_topics,
            "progress_percentage": (completed_topics / total_topics * 100) if total_topics > 0 else 0,
            "weak_topics": self.get_weak_topics()
        }
