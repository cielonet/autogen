# File based from: https://github.com/microsoft/autogen/blob/main/autogen/coding/base.py
# Credit to original authors

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

from autogen_core.base import CancellationToken



import subprocess
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


from autogen_ext.models.trained.language_inference import language_model, language_vectorizer

model: MultinomialNB = language_model
vectorizer: TfidfVectorizer = language_vectorizer


@dataclass
class CodeBlock:
    """A code block extracted from an agent message."""
    
    code: str
    language: str
    
    def __post_init__(self):
        # Automatically strip leading/trailing spaces based on language
        code_lines = self.code.splitlines()

        if self.language == "sh":
            # For shell commands (sh), remove leading spaces from every line
            self.code = "\n".join(line.lstrip() for line in code_lines)
        elif self.language == "python":
            # If the language is Python, format the code using a Python linter
            self.code = self.format_python_code(self.code)
        else:
            # For other languages, keep original formatting but clean up trailing spaces
            if len(code_lines) == 1:
                # For single-line code blocks, strip leading spaces (not trailing)
                self.code = self.code.lstrip()
            else:
                # For multi-line code blocks, maintain indentation but clean up unnecessary trailing spaces
                self.code = "\n".join(line.rstrip() for line in code_lines)
        
        # Update the language field using the lang_to_cmd function
        self.language = self.lang_to_cmd(self.code)

    def format_python_code(self, code: str) -> str:
        """Format the Python code using a linter (e.g., black or autopep8)."""
        try:
            # Install black (ensure it's installed)
            subprocess.run(["pip", "install", "black"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Use black to format the code
            result = subprocess.run(
                ["black", "-", "--quiet", "--fast"],  # Pass the code to black via stdin
                input=code.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.stdout.decode().strip()  # Return formatted code
        except subprocess.CalledProcessError as e:
            print(f"Error formatting Python code: {e.stderr.decode()}")
            return code  # Return the original code if formatting fails
    
    def lang_to_cmd(self, lang: str) -> str:
        # Preprocess the input_text using the loaded vectorizer
        input_vector = vectorizer.transform([lang])

        # Predict the label using the loaded model
        prediction = model.predict(input_vector)

        # Return the predicted label
        return "python" if prediction[0] == 0 else "sh"

@dataclass
class CodeResult:
    """Result of a code execution."""

    exit_code: int
    output: str


@runtime_checkable
class CodeExecutor(Protocol):
    """Executes code blocks and returns the result."""

    async def execute_code_blocks(
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> CodeResult:
        """Execute code blocks and return the result.

        This method should be implemented by the code executor.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CodeResult: The result of the code execution.

        Raises:
            ValueError: Errors in user inputs
            asyncio.TimeoutError: Code execution timeouts
            asyncio.CancelledError: CancellationToken evoked during execution
        """
        ...

    async def restart(self) -> None:
        """Restart the code executor.

        This method should be implemented by the code executor.

        This method is called when the agent is reset.
        """
        ...
