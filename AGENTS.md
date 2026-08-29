# Repository Conventions

These conventions apply to all future work in this repository.

## Commit Style

- Write commit subjects in the imperative mood, capitalized, and without a trailing period (for example, `Add CLIP feature extractor`).
- Keep the subject line under 72 characters.
- When more detail is needed, add a blank line after the subject and use the body to explain what changed and why.
- Keep each commit to one logical change.

## Code Style

- Use descriptive names and avoid abbreviations (for example, `image` rather than `img`).
- Keep functions small and focused on a single responsibility.
- Do not use magic numbers; define named constants instead.
- Use docstrings and comments to explain why a decision exists, rather than restating what the code does.

## Branching and Pull Requests

- Never commit directly to `main`.
- Create feature branches per component using kebab-case (for example, `add-clip-features` or `add-eval-harness`).
- Keep pull requests small and focused: one feature or fix per pull request.
