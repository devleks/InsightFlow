# InsightFlow - Testing Documentation

This document outlines the testing strategy for the InsightFlow application. As the project evolves, this documentation should be updated to reflect the current testing practices and coverage.

## 1. Testing Philosophy

Our goal is to ensure InsightFlow is reliable, robust, and provides accurate information to users. Testing will focus on:

*   **Core Functionality**: Document loading, parsing, embedding, retrieval, and question-answering.
*   **Data Integrity**: Correctness of metadata, especially source attribution and page numbers.
*   **User Interface**: Responsiveness and usability of the Streamlit application.
*   **Configuration**: Ensuring the application behaves correctly with different settings in `config.yaml`.

## 2. Testing Frameworks and Tools

*   **Python `unittest` / `pytest`**: For writing and running unit and integration tests.
    *   `pytest` is generally preferred for its rich feature set, fixtures, and easier test discovery.
*   **`streamlit` testing capabilities** (if available/mature): Streamlit may offer utilities or patterns for testing Streamlit apps specifically.
*   **`mock` library (`unittest.mock`)**: For mocking external dependencies like API calls (OpenAI, Groq), file system operations, or complex objects during unit tests.
*   **Coverage.py**: To measure test coverage.

## 3. Types of Tests and Coverage

### a. Unit Tests

*   **Purpose**: Test individual functions, methods, or classes in isolation.
*   **What's Covered (Examples)**:
    *   Configuration loading logic (`load_config`, `get_embedding_model`, `get_llm`).
    *   Metadata processing functions (e.g., cleaning metadata, ensuring correct page number extraction logic).
    *   Helper utilities (e.g., text formatting).
    *   Individual components of the RAG chain if broken down into smaller, testable units.
*   **Location**: Typically in a `tests/unit` directory.

### b. Integration Tests

*   **Purpose**: Test the interaction between different components of the application.
*   **What's Covered (Examples)**:
    *   **Document Processing Pipeline**: Test the flow from loading a sample document (PDF, TXT, DOCX) -> splitting -> embedding -> storing in a test ChromaDB instance.
        *   Verify that metadata (source, page number) is correctly preserved and stored.
        *   Verify that `filter_complex_metadata` correctly handles problematic metadata.
    *   **RAG Chain**: Test the full RAG chain with a sample query, a small set of test documents in a test vector store, and a mocked LLM.
        *   Verify that relevant documents are retrieved.
        *   Verify that the LLM prompt is constructed correctly.
        *   Verify that sources are correctly attributed in the (mocked) response.
    *   Interaction between `app.py` logic and configuration settings from `config.yaml`.
*   **Location**: Typically in a `tests/integration` directory.

### c. End-to-End (E2E) Tests (Future Aspiration)

*   **Purpose**: Test the entire application flow from the user's perspective, including the UI.
*   **What's Covered (Examples)**:
    *   Simulating a user uploading a document, asking a question via the Streamlit UI, and verifying the response and sources displayed.
*   **Tools**: May involve UI automation frameworks like Selenium, Playwright, or Streamlit-specific testing tools if they evolve.
*   **Status**: Currently, E2E testing for InsightFlow is primarily manual due to the interactive nature of Streamlit apps and the complexity of UI automation. This is an area for future enhancement.

## 4. How to Run Tests

*(This section assumes `pytest` is adopted)*

1.  Ensure all development dependencies, including `pytest` and any other testing tools, are installed:
    ```bash
    pip install pytest pytest-mock coverage
    # (or from a requirements-dev.txt)
    ```
2.  Navigate to the project root directory.
3.  Run tests using the `pytest` command:
    ```bash
    pytest
    ```
4.  To run tests with coverage reporting:
    ```bash
    coverage run -m pytest
    coverage report -m
    # For HTML report: coverage html
    ```

## 5. Current Testing Status & Test Coverage Reports

*   **Current Status**: The InsightFlow application has been developed with a focus on core functionality through iterative development and manual testing during pairing sessions with Cascade.
*   **Formal automated testing (unit/integration tests) is an area for significant future development.** While individual functions have been tested during development, a comprehensive automated test suite is not yet in place.
*   **Manual Testing Performed**: During development, extensive manual testing was performed for:
    *   Document loading of PDF, TXT, DOCX.
    *   Page number extraction from PDFs under various conditions.
    *   Resolution of `ValueError` with ChromaDB metadata.
    *   UI display of sources and chat messages.
    *   Basic Q&A functionality.
*   **Test Coverage**: No formal test coverage reports are available yet due to the absence of a comprehensive automated test suite.

## 6. Future Testing Goals

*   Develop a suite of unit tests for critical utility functions and configuration handlers.
*   Create integration tests for the document processing pipeline (loading to vector store) using sample files.
*   Implement integration tests for the RAG chain with mocked LLM responses to verify context retrieval and source attribution logic.
*   Explore options for lightweight E2E or UI testing for Streamlit if feasible tools emerge.
*   Set up CI/CD (e.g., GitHub Actions) to automatically run tests on pushes and pull requests.

Contributions to improving test coverage are highly welcome!
