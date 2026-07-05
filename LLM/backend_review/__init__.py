"""Backend-ready review helpers.

The primary review engine is qa_agent. Few-shot review can be attached as an
optional diagnostic signal, but it is not used as the blocking decision maker.
"""

from .service import (
    review_parsed_document,
    review_files,
    summarize_for_backend,
    iter_issue_messages,
)