import re
from typing import List

from . import CodeBlock


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    # Updated pattern to capture code blocks with and without language identifiers
    pattern = re.compile(r"```(\w+)?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    
    code_blocks: List[CodeBlock] = []
    
    for match in matches:
        language = match[0].strip() if match[0] else "markdown"  # Default to 'markdown' if no language is specified
        code_content = match[1].strip()  # Strip leading/trailing whitespace from code content

        # Only append non-empty code content to the list
        if code_content:
            if "markdown" not in language: # ignore executing markdown classified code
                code_blocks.append(CodeBlock(code=code_content, language=language))
    
    return code_blocks