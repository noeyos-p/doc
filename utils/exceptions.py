class DSChatbotError(Exception):
    """베이스 예외"""


class ParserException(DSChatbotError):
    """PDF 파싱 오류"""


class PreprocessException(DSChatbotError):
    """전처리 오류"""


class VectorStoreException(DSChatbotError):
    """벡터 DB 오류"""


class ChatException(DSChatbotError):
    """챗봇 응답 오류"""
