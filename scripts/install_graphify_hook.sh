#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "${repo_root}" ]]; then
  echo "install_graphify_hook: run this inside a git repository" >&2
  exit 1
fi

hook_dir="${repo_root}/.git/hooks"
hook_path="${hook_dir}/post-commit"
mkdir -p "${hook_dir}"

if [[ -f "${hook_path}" ]] && ! grep -q "SSATK Graphify post-commit hook" "${hook_path}"; then
  backup="${hook_path}.backup.$(date +%Y%m%d%H%M%S)"
  cp "${hook_path}" "${backup}"
  echo "install_graphify_hook: existing post-commit hook backed up to ${backup}" >&2
fi

cat > "${hook_path}" <<'HOOK'
#!/usr/bin/env bash
# SSATK Graphify post-commit hook.
# Rebuilds a local repo knowledge graph after commits when graphify is installed.
# Generated output is ignored by git in graphify-out/.

set -u

case "${SSATK_GRAPHIFY_HOOK:-1}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    exit 0
    ;;
esac

if ! command -v graphify >/dev/null 2>&1; then
  echo "ssatk graphify: graphify CLI not found; install with: pipx install graphifyy" >&2
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${repo_root}" || exit 0
mkdir -p graphify-out

if [[ -f graphify-out/graph.json ]]; then
  graphify_args=(. --update --wiki)
else
  graphify_args=(. --wiki)
fi

run_graphify() {
  graphify "${graphify_args[@]}" > graphify-out/hook.log 2>&1 || {
    status=$?
    echo "ssatk graphify: update failed with status ${status}; see graphify-out/hook.log" >&2
    return 0
  }
}

case "${SSATK_GRAPHIFY_FOREGROUND:-0}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    run_graphify
    ;;
  *)
    ( run_graphify ) >/dev/null 2>&1 &
    echo "ssatk graphify: updating graph in background; see graphify-out/hook.log" >&2
    ;;
esac
HOOK

chmod +x "${hook_path}"
echo "install_graphify_hook: installed ${hook_path}"
