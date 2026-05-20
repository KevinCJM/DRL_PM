# Purpose
Machine-first routing protocol for this subtree only.

# Scope Boundary
- Scope root: current working directory `.` only.
- Parent inspection/routing is forbidden.
- Cross-boundary edges must be marked `out_of_scope`.

# Required Read Order
1. `docs/repo_map.json`
2. `docs/task_routes.json`
3. `docs/pitfalls.json`
4. Target code/artifact files

# Hard Rules
- Treat routing JSON as navigation metadata, not implementation truth.
- Re-open code before edit; do not rely on memory.
- Do not infer modules/files outside visible subtree.
- Keep outputs machine-usable and concise.
- Use typed references `{ "kind", "ref" }` for routed paths and dependency edges.
- Keyword matching is owned only by `docs/task_routes.json`.

# Default Operating Sequence
1. Match task via `docs/task_routes.json`.
2. Resolve modules by `first_read_modules`.
3. Resolve module fields listed in `docs/repo_map.json#/routing_merge_contract/inherited_module_fields`.
4. Apply only present `*_override` fields using `docs/repo_map.json#/routing_merge_contract/route_override_fields`.
5. Resolve pitfall checks from `docs/pitfalls.json`.
6. Execute minimal-scope change/verification.

# Edit Safety Rules
- Verify entrypoint and path constants before edits.
- Check destructive operations before directory delete/overwrite.
- Preserve config schemas, test fixtures, and generated data schemas unless task requires change.
- If dependency/runtime uncertainty appears, mark `needs_code_confirmation`.

# Verification Rules
- Run resolved structured `minimum_regression` for touched modules/routes.
- Validate output schemas/counters when data, metrics, or report files change.
- Confirm generated files remain under current subtree paths only.

# Output Discipline
- No architecture prose.
- No duplicated module/task/pitfall catalogs outside JSON files.
- Use `unknown`, `needs_code_confirmation`, or `out_of_scope` when evidence is incomplete.
