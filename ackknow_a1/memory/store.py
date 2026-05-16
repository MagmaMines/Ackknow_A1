from sqlmodel import SQLModel, Field, Session, create_engine, select
from ackknow_a1.types import ChatMessage


class MemoryRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    role: str
    content: str


class MemoryStore:
    def __init__(self, db_url: str) -> None:
        self.engine = create_engine(db_url)
        SQLModel.metadata.create_all(self.engine)

    def append(self, user_id: str, msg: ChatMessage) -> None:
        with Session(self.engine) as s:
            s.add(MemoryRow(user_id=user_id, role=msg.role, content=msg.content))
            s.commit()

    def history(self, user_id: str, limit: int = 10) -> list[ChatMessage]:
        with Session(self.engine) as s:
            rows = s.exec(select(MemoryRow).where(MemoryRow.user_id == user_id).order_by(MemoryRow.id.desc()).limit(limit)).all()
        return [ChatMessage(role=r.role, content=r.content) for r in reversed(rows)]
