# InsightFlow - User Guide

Welcome to InsightFlow! This guide will help you get started with using the application to chat with your documents.

## 1. What is InsightFlow?

InsightFlow is an application that allows you to upload your documents (PDFs, text files, Word documents) and ask questions about their content. It uses artificial intelligence to understand your questions and find relevant information within your documents, providing you with answers and direct references to the source material.

## 2. Getting Started

### Prerequisites

*   The InsightFlow application should be running. If you are running it yourself, follow the installation and usage instructions in the `README.md` file.
*   You need documents (PDF, TXT, or DOCX format) that you want to query.

### Preparing Your Documents

1.  Locate the document folder that InsightFlow is configured to use. (If you set up the app, this is the `path` specified under `data_source` in your `config.yaml` file, e.g., a folder named `documents` in the project directory).
2.  Place all the documents you want to work with into this folder.

### Accessing the Application

*   If run locally, the application usually opens automatically in your web browser at an address like `http://localhost:8501`.
*   If accessing a deployed version, use the URL provided by your administrator.

## 3. Using the Application Interface

The InsightFlow interface is designed to be simple and intuitive.

```
+--------------------------------------------------------------------+
| InsightFlow - Chat with Your Documents                             |
+--------------------------------------------------------------------+
| Sidebar (Left)                     | Main Chat Area (Right)          |
|                                    |                                 |
| [Available Source Files:]          | [Chat history appears here]     |
| - document1.pdf (Loaded)           |   User: Your question...        |
| - notes.txt (Loaded)               |   InsightFlow: My answer...     |
|                                    |     Sources:                    |
| [Settings/Info - if any]           |       - document1.pdf (Page: 5) |
|                                    |                                 |
|                                    | [_____________________________] |
|                                    | [Type your question here...   ] |
|                                    | [Send Button                  ] |
+--------------------------------------------------------------------+
```

### a. Sidebar

*   **Available Source Files**: On the left sidebar, you'll see a list of documents found in the configured source folder. It will indicate if they have been successfully loaded and processed by the application.
    *   This section updates automatically when the application starts or if new documents are detected (behavior may vary based on configuration).

### b. Main Chat Area

*   **Chat History**: This is where your conversation with InsightFlow will appear. Your questions and the application's answers (along with sources) will be displayed chronologically.
*   **Input Box**: At the bottom of the chat area, there's a text box labeled "Type your question here..." or similar.
*   **Send Button**: Next to the input box, there's a button to send your question.

## 4. Feature Walkthroughs

### a. Asking a Question

1.  Type your question about the content of your uploaded documents into the input box.
    *   _Example_: "What were the main conclusions of the Q3 report?"
    *   _Example_: "Summarize the section on market analysis in 'business_plan.docx'."
2.  Click the "Send" button or press Enter.
3.  InsightFlow will process your question, search through the documents, and generate an answer.
4.  The answer will appear in the chat history.

### b. Understanding Responses and Sources

*   **Answer**: InsightFlow will provide a textual answer to your question.
*   **Sources**: Below the answer, you will see a "Sources" section. This lists the specific document(s) and page number(s) from which the information was derived.
    *   This allows you to refer back to the original documents to verify the information or get more context.
    *   _Example Source_: `- my_research_paper.pdf (Page: 12)`

### c. Chat History

*   Your entire conversation (questions, answers, sources) is maintained during your current session.
*   You can scroll up to review previous parts of the conversation.
*   The context from previous turns in the conversation might be used by the LLM to understand follow-up questions better.

### d. Managing Documents

*   To add new documents, place them in the configured source folder.
*   To remove documents, delete them from the source folder.
*   **Note**: Depending on the application's configuration, you might need to restart the application or wait for it to automatically re-scan the document folder for changes to take effect in the vector store.

## 5. Troubleshooting FAQs

*   **Q: My documents are not appearing in the sidebar.**
    *   **A**: Ensure your documents are in the correct folder specified in the application's configuration (`config.yaml`, `data_source.path`).
    *   **A**: Check if the file formats are supported (PDF, TXT, DOCX).
    *   **A**: Look for any error messages in the sidebar or console (if you are running the app locally).

*   **Q: InsightFlow is slow to respond.**
    *   **A**: Processing large documents or complex queries can take time. If it's the first query after starting, document loading and embedding might still be in progress.
    *   **A**: Check your internet connection if using cloud-based LLM services.

*   **Q: The answers seem incorrect or irrelevant.**
    *   **A**: Try rephrasing your question to be more specific.
    *   **A**: Ensure the relevant information is present in the uploaded documents.
    *   **A**: The quality of answers depends on the LLM used and the clarity of the document content.

*   **Q: I'm getting an error message.**
    *   **A**: Note down the error message. If you are a technical user, check the application logs or console. Otherwise, report it to the application administrator or support contact.

*   **Q: How are page numbers determined for sources?**
    *   **A**: For PDF documents, InsightFlow attempts to extract precise page numbers. For other document types, page numbers might be less precise or not available. The quality of page number extraction depends on the structure and format of the PDF.

## 6. Tips for Effective Use

*   **Be Specific**: The more specific your question, the better InsightFlow can target the relevant information.
*   **Use Keywords**: Include keywords that are likely to appear in the relevant sections of your documents.
*   **Ask Follow-up Questions**: You can ask follow-up questions to delve deeper into a topic based on previous answers.
*   **Check Sources**: Always refer to the provided sources to verify critical information and gain a fuller understanding.

We hope this guide helps you make the most of InsightFlow!
