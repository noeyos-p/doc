from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class TableContent:
    page_number: int
    markdown: str
    dataframe: Optional[pd.DataFrame] = None
    accuracy: Optional[float] = None
    caption: str = ""


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: list[TableContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    filename: str
    page_count: int
    file_size: int
    report_date: str = ""
    author: str = ""


@dataclass
class ParsedDocument:
    metadata: DocumentMetadata
    pages: list[PageContent]

    def get_all_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def get_all_tables(self) -> list[TableContent]:
        tables = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    chunk_type: str          # "text" | "table"
    section_title: str
    page_number: int
    source_file: str
    metadata: dict = field(default_factory=dict)
