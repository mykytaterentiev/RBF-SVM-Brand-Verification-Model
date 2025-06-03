# Security Policy

## Data Privacy and Security

This project handles brand verification data which may contain sensitive business information. Please follow these security guidelines:

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

### Data Handling

- **Never commit data files** to the repository
- **Use .env files** for configuration (never commit these)
- **Sanitize logs** to avoid exposing sensitive information
- **Use relative paths** for data files when possible

### Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Send details to: [your-security-email@example.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **24 hours**: Initial response acknowledging receipt
- **7 days**: Assessment and initial response with plan
- **30 days**: Resolution or detailed status update

### Security Measures

#### Code Security
- All dependencies are pinned to specific versions
- Pre-commit hooks prevent common security issues
- No hardcoded secrets or credentials

#### Data Security
- Data files are excluded from version control
- Sample weights protect against data leakage
- Model artifacts are treated as sensitive

#### Runtime Security
- Input validation for all user-provided data
- Safe file handling practices
- Secure model loading/saving

### Best Practices for Contributors

1. **Never commit**:
   - Data files (CSV, JSON, etc.)
   - Model files (.joblib, .pkl, etc.)
   - Configuration files with secrets
   - Log files with sensitive information

2. **Always**:
   - Use environment variables for sensitive config
   - Validate inputs
   - Follow principle of least privilege
   - Keep dependencies updated

3. **Code Review Requirements**:
   - All PRs require review
   - Security-sensitive changes require additional review
   - Automated security scanning via pre-commit hooks

### Contact

For security concerns: [mykytaterentievua@gmail.com]
For general questions: [mykytaterentievua@gmail.com]
