class KnowledgeBaseError(Exception):
    """Knowledge base problem. user_message is safe to show to users."""

    def __init__(self, kind: str, user_message: str):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message