# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please do NOT create public GitHub issues for security vulnerabilities.**

Instead, report security issues via email:
- **Email:** security@kuzushilabs.com
- **Subject:** [SECURITY] Knomly - Brief description

### What to Include

Please provide as much information as possible:

1. **Description** - Clear description of the vulnerability
2. **Impact** - What could an attacker do with this vulnerability?
3. **Reproduction Steps** - How can we reproduce the issue?
4. **Affected Versions** - Which versions are affected?
5. **Suggested Fix** - If you have ideas for remediation

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Initial acknowledgment | Within 48 hours |
| Status update | Within 7 days |
| Resolution target | Within 30 days |
| Public disclosure | After fix released |

### What to Expect

1. We will acknowledge receipt of your report
2. We will investigate and validate the issue
3. We will work on a fix and coordinate disclosure
4. We will credit you (unless you prefer anonymity)

## Security Best Practices

When using Knomly in production, follow these security practices:

### Environment Variables

- **Never commit `.env` files** to version control
- Use `.env.example` as a template only
- Store production secrets in a secrets manager (HashiCorp Vault, AWS Secrets Manager, etc.)

### API Keys

- **Rotate keys regularly** - Especially after team member changes
- **Use least privilege** - Only grant necessary permissions
- **Monitor usage** - Watch for unusual API call patterns

### Multi-Tenancy

Knomly supports multi-tenant deployments with per-user credential isolation:

```python
# Credentials are scoped to ToolContext, never stored in frames
context = ToolContext(
    user_id="user-123",
    secrets={"api_key": "..."},  # Per-user, not shared
)
```

- **Never store secrets in Frame metadata** - ADR-004 compliance
- **Use secret callbacks** for production credential retrieval
- **Audit tool invocations** for unauthorized access

### Webhook Security

When exposing webhooks (Twilio, etc.):

- **Validate request signatures** - Verify requests come from trusted sources
- **Use HTTPS only** - Never expose webhooks over HTTP
- **Rate limit** - Protect against abuse

```python
# Example: Twilio signature validation
from twilio.request_validator import RequestValidator

validator = RequestValidator(auth_token)
is_valid = validator.validate(url, params, signature)
```

### Database Security

- **Enable authentication** - Don't run MongoDB without auth
- **Use TLS** - Encrypt connections in transit
- **Backup regularly** - Protect against data loss

### Logging

- **Never log secrets** - API keys, tokens, passwords
- **Sanitize PII** - Personal data in logs
- **Retain appropriately** - Follow data retention policies

## Known Security Considerations

### Frame Serialization

Frames are serializable for debugging and audit trails. Be aware:

- Frames may contain user-provided data
- Sanitize before logging to external systems
- Consider encryption for sensitive frame content

### LLM Prompt Injection

When using LLM providers with user input:

- Validate and sanitize user input
- Use system prompts to constrain behavior
- Monitor for unusual outputs

### Tool Execution

Agent tools can perform real actions. Implement safeguards:

- Use `ToolAnnotations` to mark destructive operations
- Implement approval workflows for sensitive actions
- Log all tool invocations for audit

## Security Updates

Security updates will be announced via:

- GitHub Security Advisories
- Release notes in CHANGELOG.md
- Email to registered maintainers

## Acknowledgments

We appreciate security researchers who help keep Knomly secure. Contributors will be acknowledged in release notes (unless they prefer anonymity).
