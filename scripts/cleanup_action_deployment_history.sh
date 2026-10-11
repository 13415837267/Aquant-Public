#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${REPOSITORY:?REPOSITORY is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"

deleted_actions='[]'
preserved_active='[]'
preserved_successful='[]'
preserved_other_completed='[]'
cleanup_errors='[]'

if ! runs_tsv="$(gh api --paginate "repos/$REPOSITORY/actions/runs?per_page=100" --jq '.workflow_runs[] | [.id, .created_at, .status, (.conclusion // ""), (.name // "")] | @tsv')"; then
  echo "读取历史工作流运行失败。"
  exit 1
fi

while IFS=$'\t' read -r run_id created_at status conclusion run_name; do
  [[ -n "${run_id:-}" ]] || continue

  # 只清理已完成且结论明确为失败或跳过的 Action 运行。
  # 运行中、排队中、成功以及取消等其他结论都必须保留。
  if [[ "$status" != "completed" ]]; then
    preserved_active="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg status "$status" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, status:$status, name:$name}]' <<< "$preserved_active")"
    echo "保留尚未完成的运行：$run_id（$status）"
    continue
  fi

  if [[ "$conclusion" != "failure" && "$conclusion" != "skipped" ]]; then
    if [[ "$conclusion" == "success" ]]; then
      preserved_successful="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, name:$name}]' <<< "$preserved_successful")"
    else
      preserved_other_completed="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg conclusion "$conclusion" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, conclusion:$conclusion, name:$name}]' <<< "$preserved_other_completed")"
    fi
    echo "保留非清理目标运行：$run_id（${conclusion:-无结论}）"
    continue
  fi

  if gh api --method DELETE "repos/$REPOSITORY/actions/runs/$run_id" >/dev/null; then
    deleted_actions="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg conclusion "$conclusion" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, conclusion:$conclusion, name:$name}]' <<< "$deleted_actions")"
    echo "已清理历史运行：$run_id $run_name（$conclusion）"
  else
    cleanup_errors="$(jq -c --arg kind "action_run" --arg id "$run_id" --arg conclusion "$conclusion" '. + [{kind:$kind, id:$id, conclusion:$conclusion}]' <<< "$cleanup_errors")"
    echo "清理历史运行失败：$run_id"
  fi
done <<< "$runs_tsv"

manifest_path="data/maintenance/action_deployment_cleanup/run-$GITHUB_RUN_ID.json"
mkdir -p "$(dirname "$manifest_path")"
jq -n \
  --arg repository "$REPOSITORY" \
  --arg cleanup_run_id "$GITHUB_RUN_ID" \
  --arg completed_at "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --argjson deleted_actions "$deleted_actions" \
  --argjson preserved_active_runs "$preserved_active" \
  --argjson preserved_successful_runs "$preserved_successful" \
  --argjson preserved_other_completed_runs "$preserved_other_completed" \
  --argjson errors "$cleanup_errors" \
  '{
    repository:$repository,
    cleanup_run_id:($cleanup_run_id|tonumber),
    completed_at:$completed_at,
    deleted_actions:$deleted_actions,
    preserved_active_runs:$preserved_active_runs,
    preserved_successful_runs:$preserved_successful_runs,
    preserved_other_completed_runs:$preserved_other_completed_runs,
    errors:$errors
  }' > "$manifest_path"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add "$manifest_path"
if ! git diff --cached --quiet; then
  git commit -m "维护：记录失败与跳过 Action 清理结果"
  pushed=false
  for attempt in 1 2 3; do
    if git push origin HEAD:main; then
      pushed=true
      break
    fi
    echo "审计清单推送遇到并发更新，第${attempt}次回基重试。"
    git pull --rebase origin main
  done
  if [[ "$pushed" != "true" ]]; then
    echo "审计清单连续三次推送失败。"
    exit 1
  fi
fi

echo "已清理失败或跳过的 Action 记录数：$(jq 'length' <<< "$deleted_actions")"
echo "保留的运行中或排队中记录数：$(jq 'length' <<< "$preserved_active")"
echo "保留的成功记录数：$(jq 'length' <<< "$preserved_successful")"
echo "保留的其他已完成记录数：$(jq 'length' <<< "$preserved_other_completed")"
echo "清理错误数：$(jq 'length' <<< "$cleanup_errors")"
if [[ "$(jq 'length' <<< "$cleanup_errors")" -gt 0 ]]; then
  exit 1
fi
