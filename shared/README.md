# shared

Only code that is genuinely reused by multiple integrations belongs here.
Integration-specific build, runtime, patch, environment, and test code stays under
`integrations/<name>/` even when moving it here would make the top-level tree look more uniform.
