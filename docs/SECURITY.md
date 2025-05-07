# InsightFlow - Security Notes

This document outlines key security considerations, practices, and potential vulnerabilities for the InsightFlow application. It is intended to guide developers and administrators in maintaining the security of the application.

## 1. Authentication and Authorization

*   **Current Status**: InsightFlow, as a Streamlit application primarily run locally or in a trusted environment, does not currently implement its own user authentication or authorization layer for accessing the application itself.
    *   Access to the machine running the Streamlit app implies access to the app.
*   **Considerations for Deployment**: If deployed to a multi-user environment or the public internet, robust authentication (e.g., OAuth 2.0, SAML, or a managed identity proxy) **must** be implemented in front of the Streamlit application. Streamlit itself has limited built-in multi-user security features for public deployments.

## 2. API Key Management (Secrets)

*   **Method**: API keys for external services (OpenAI, Groq, Tavily, etc.) are managed via a `.env` file, loaded by `python-dotenv`.
*   **Best Practices**:
    *   The `.env` file **must not** be committed to version control (e.g., Git). It should be listed in `.gitignore`.
    *   A `.env.example` file should be provided in the repository, listing the required environment variables without their values.
    *   For production deployments, use secure secret management solutions provided by the hosting platform or dedicated secret managers (e.g., HashiCorp Vault, AWS Secrets Manager, Azure Key Vault).
    *   Restrict API key permissions to the minimum required for the application's functionality.
    *   Regularly rotate API keys where possible.

## 3. Data Protection

### a. Data in Transit

*   **External API Calls**: Interactions with external APIs (OpenAI, Groq) are performed over HTTPS by the respective client libraries, ensuring encryption in transit.
*   **Streamlit Application**: When running Streamlit locally, traffic is typically HTTP. If deployed, ensure it is served over HTTPS, usually by placing it behind a reverse proxy (e.g., Nginx, Caddy) that handles SSL/TLS termination.

### b. Data at Rest

*   **Uploaded Documents**: Original documents are stored on the local file system in the path specified by `data_source.path` in `config.yaml`.
    *   File system permissions should be appropriately restricted.
    *   Consider encryption at rest for the disk/volume where documents are stored, especially if they contain sensitive information.
*   **Vector Store (ChromaDB)**: ChromaDB persists data to the local file system in the directory specified by `vector_store.persist_directory`.
    *   The same considerations for file system permissions and disk-level encryption apply.
    *   ChromaDB itself may not offer fine-grained access control or encryption at the database level beyond what the file system provides for its persisted files.
*   **Configuration Files**: `config.yaml` and `.env` files contain sensitive settings and API keys. Restrict access to these files using file system permissions.

## 4. Input Sanitization and Output Encoding

*   **User Queries**: User input (queries) is passed to LLMs. While LLMs are generally robust, there's a theoretical risk of prompt injection if user input is crafted maliciously. Langchain and the LLM providers may have some built-in protections.
    *   Consider input length limitations to prevent overly long queries that might abuse resources.
*   **Document Content**: The application processes user-uploaded documents. Maliciously crafted documents (e.g., "zip bombs" if archive support were added, or documents exploiting parser vulnerabilities) could pose a risk.
    *   `UnstructuredFileLoader` is a generally robust library, but keeping it and its dependencies (like `pypdf`, `python-docx`) updated is important to mitigate known parser vulnerabilities.
*   **Rendered HTML**: Streamlit handles rendering. When displaying LLM-generated content or sources (especially if using `unsafe_allow_html=True` as we did for immediate source display), ensure that the content being rendered doesn't inadvertently contain malicious HTML/JavaScript if it's derived from sources that could be manipulated. In our current implementation, the HTML for sources is constructed safely in the backend.

## 5. Dependency Management

*   **Vulnerabilities in Dependencies**: Keep all project dependencies (listed in `requirements.txt`) up to date to patch known vulnerabilities.
*   Use tools like `pip-audit` or GitHub's Dependabot to scan for known vulnerabilities in dependencies.
*   Regularly review and update dependencies.

## 6. LLM-Specific Security

*   **Prompt Injection**: As mentioned, users might try to craft prompts to make the LLM ignore previous instructions or reveal sensitive system information. This is an ongoing area of research. Techniques like instruction Evasion, prompt filtering, or using models specifically fine-tuned for instruction following can help.
*   **Data Poisoning (for RAG)**: If the documents used for the RAG knowledge base can be tampered with, the LLM's responses could be biased or incorrect. Ensure the integrity of your source documents.
*   **Denial of Service**: Maliciously crafted queries or excessive API calls could lead to high costs or denial of service. Implement rate limiting or usage quotas if deploying publicly.

## 7. Logging and Monitoring

*   Implement logging for application events, errors, and potentially for security-relevant actions (e.g., failed access attempts if authentication is added).
*   Monitor logs for suspicious activity.
*   Streamlit's console output and logs from `app.py` provide basic logging.

## 8. Known Vulnerabilities or Mitigations

*   **No Built-in Authentication**: The primary known vulnerability for a wider deployment is the lack of built-in user authentication in the base Streamlit application. Mitigation: Deploy behind an authenticating reverse proxy or use an identity-aware proxy.
*   **Parser Vulnerabilities**: Reliance on document parsing libraries means the application is susceptible to vulnerabilities in those libraries. Mitigation: Keep dependencies updated.
*   **Exposure of File System Paths**: Source metadata includes file paths. In a local context, this is for user reference. If deployed where the file system structure could be sensitive, consider transforming or anonymizing these paths before display, though this might reduce user clarity.

This document provides a starting point. Security is an ongoing process, and regular reviews and updates are necessary as the application evolves and new threats emerge.
