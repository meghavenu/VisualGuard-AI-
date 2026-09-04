# Security

## Secrets

VisualGuard AI uses `GEMINI_API_KEY` from the environment.

- Keep `.env` local.
- Use `.env.example` as the public template.
- Never commit API keys or passwords.
- If a key is exposed, revoke/rotate it immediately.

## Login credentials

Level 3 can accept login credentials through the UI. Prefer test accounts and never store production credentials in source control.

## Captured pages

Screenshots may contain sensitive or proprietary information. Review generated artifacts before sharing them publicly.
