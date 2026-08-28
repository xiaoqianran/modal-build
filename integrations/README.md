# Integrations

Each directory owns one model/runtime integration end to end: build tooling, pinned environment
metadata, runtime code, patches, scripts, tests, and integration-specific documentation as needed.

Do not extract code into `shared/` merely to reduce duplication; move it only after at least two
integrations have a stable, genuinely common contract.
