from sqlalchemy import Column, String, Text, TIMESTAMP
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from db.session import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String(100))
    user_query = Column(Text)
    user_id = Column(Text)
    bot_response = Column(Text)
    brain = Column(Text)
    # The retrieval provenance for this turn, as JSON. Without it the evidence panel would
    # survive only until the page reloads: bot_response is prose, and the chunk ids behind
    # it are not recoverable from the text. NULL for every turn recorded before this column
    # existed, and for turns with no retrieval (small talk, refusals).
    citations_json = Column(Text, nullable=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)