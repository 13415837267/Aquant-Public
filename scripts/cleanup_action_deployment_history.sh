#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${REPOSITORY:?REPOSITORY is required}"
: "${BASELINE_RUN_ID:?BASELINE_RUN_ID is required}"

if [[ ! "$BASELINE_RUN_ID" =~ ^[0-9]+$ ]]; then
  echo "基准运行编号必须是数字。"
  exit 1
fi

baseline_json="$(gh api "repos/$REPOSITORY/actions/runs/$BASELINE_RUN_ID")"
baseline_repo="$(jq -r '.repository.full_name' <<< "$baseline_json")"
cutoff="$(jq -r '.created_at' <<< "$baseline_json")"
if [[ "$baseline_repo" != "$REPOSITORY" || -z "$cutoff" || "$cutoff" == "null" ]]; then
  echo "无法核实基准运行所属仓库或创建时间。"
  exit 1
fi

echo "清理基准：$BASELINE_RUN_ID"
echo "基准创建时间：$cutoff"

deleted_actions='[]'
preserved_active='[]'
deleted_deployments='[]'
preserved_deployments='[]'
cleanup_errors='[]'
prior_partial_actions='[]'
prior_partial_attempts='[]'

# 上一次清理在处理部署列表前中断；把日志已确认删除的运行纳入本次审计清单。
if [[ "$BASELINE_RUN_ID" == "37879972344" ]]; then
  prior_partial_attempts='[37880672485]'
  prior_partial_actions='[
    {"run_id":37816631042,"name":"01 Python质量检查","removed_in_attempt":37880672485},
    {"run_id":37816228215,"name":"05 网页自动部署","removed_in_attempt":37880672485},
    {"run_id":37816202504,"name":"05 网页自动部署","removed_in_attempt":37880672485},
    {"run_id":37815930439,"name":"03 候选池生产","removed_in_attempt":37880672485},
    {"run_id":37815729871,"name":"03 候选池生产","removed_in_attempt":37880672485},
    {"run_id":37815696216,"name":"01 Python质量检查","removed_in_attempt":37880672485},
    {"run_id":37815597288,"name":"03 候选池生产","removed_in_attempt":37880672485},
    {"run_id":37815311741,"name":"01 Python质量检查","removed_in_attempt":37880672485},
    {"run_id":37815256679,"name":"03 候选池生产","removed_in_attempt":37880672485},
    {"run_id":37815106177,"name":"03 候选池生产","removed_in_attempt":37880672485},
    {"run_id":37814752037,"name":"02 数据维护","removed_in_attempt":37880672485},
    {"run_id":37814407952,"name":"01 Python质量检查","removed_in_attempt":37880672485},
    {"run_id":37814059641,"name":"05 网页自动部署","removed_in_attempt":37880672485},
    {"run_id":37813417530,"name":"01 Python质量检查","removed_in_attempt":37880672485},
    {"run_id":37812010156,"name":"01 Python质量检查","removed_in_attempt":37880672485},
    {"run_id":37811783939,"name":"01 Python质量检查","removed_in_attempt":37880672485},
    {"run_id":37810900460,"name":"01 Python质量检查","removed_in_attempt":37880672485}
  ]'
fi

if ! runs_tsv="$(gh api --paginate "repos/$REPOSITORY/actions/runs?per_page=100" --jq '.workflow_runs[] | [.id, .created_at, .status, (.conclusion // ""), (.name // "")] | @tsv')"; then
  echo "读取历史工作流运行失败。"
  exit 1
fi

while IFS= read -r run_row; do
  [[ -n "$run_row" ]] || continue
  run_id="$(cut -f1 <<< "$run_row")"
  created_at="$(cut -f2 <<< "$run_row")"
  status="$(cut -f3 <<< "$run_row")"
  conclusion="$(cut -f4 <<< "$run_row")"
  run_name="$(cut -f5 <<< "$run_row")"
  [[ -n "$run_id" ]] || continue
  [[ "$run_id" != "$BASELINE_RUN_ID" ]] || continue
  [[ "$created_at" < "$cutoff" ]] || continue

  if [[ "$status" != "completed" ]]; then
    preserved_active="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg status "$status" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, status:$status, name:$name}]' <<< "$preserved_active")"
    echo "保留尚未完成的旧运行：$run_id（$status）"
    continue
  fi

  if gh api --method DELETE "repos/$REPOSITORY/actions/runs/$run_id" >/dev/null; then
    deleted_actions="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg conclusion "$conclusion" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, conclusion:$conclusion, name:$name}]' <<< "$deleted_actions")"
    echo "已清理历史运行：$run_id $run_name"
  else
    cleanup_errors="$(jq -c --arg kind "action_run" --arg id "$run_id" '. + [{kind:$kind, id:$id}]' <<< "$cleanup_errors")"
    echo "清理历史运行失败：$run_id"
  fi
done <<< "$runs_tsv"

if ! deployments_tsv="$(gh api --paginate "repos/$REPOSITORY/deployments?per_page=100" --jq '.[] | [.id, .created_at, (.environment // ""), (.task // ""), (.sha // "")] | @tsv')"; then
  echo "读取历史部署失败。"
  exit 1
fi

seen_env_file="$RUNNER_TEMP/seen-environments.txt"
: > "$seen_env_file"
while IFS= read -r deployment_row; do
  [[ -n "$deployment_row" ]] || continue
  deployment_id="$(cut -f1 <<< "$deployment_row")"
  created_at="$(cut -f2 <<< "$deployment_row")"
  environment="$(cut -f3 <<< "$deployment_row")"
  task_name="$(cut -f4 <<< "$deployment_row")"
  commit_sha="$(cut -f5 <<< "$deployment_row")"
  [[ -n "$deployment_id" ]] || continue
  environment_key="$environment"
  if [[ -z "$environment_key" ]]; then
    environment_key="未指定环境"
  fi

  if [[ "$created_at" > "$cutoff" || "$created_at" == "$cutoff" ]]; then
    printf '%s\n' "$environment_key" >> "$seen_env_file"
    preserved_deployments="$(jq -c --arg id "$deployment_id" --arg created "$created_at" --arg environment "$environment_key" --arg task "$task_name" --arg sha "$commit_sha" --arg reason "基准时间之后的部署" '. + [{deployment_id:($id|tonumber), created_at:$created, environment:$environment, task:$task, commit_sha:$sha, reason:$reason}]' <<< "$preserved_deployments")"
    continue
  fi

  if ! grep -Fqx -- "$environment_key" "$seen_env_file"; then
    printf '%s\n' "$environment_key" >> "$seen_env_file"
    preserved_deployments="$(jq -c --arg id "$deployment_id" --arg created "$created_at" --arg environment "$environment_key" --arg task "$task_name" --arg sha "$commit_sha" --arg reason "该环境最新现行基线" '. + [{deployment_id:($id|tonumber), created_at:$created, environment:$environment, task:$task, commit_sha:$sha, reason:$reason}]' <<< "$preserved_deployments")"
    echo "保留环境最新部署基线：$deployment_id（$environment_key）"
    continue
  fi

  latest_state="$(gh api "repos/$REPOSITORY/deployments/$deployment_id/statuses?per_page=1" --jq '.[0].state // ""' 2>/dev/null || true)"
  if [[ "$latest_state" != "inactive" ]]; then
    if ! gh api --method POST "repos/$REPOSITORY/deployments/$deployment_id/statuses" -f state=inactive -f description="清理现行基线之前的历史部署记录" >/dev/null; then
      cleanup_errors="$(jq -c --arg kind "deployment_deactivate" --arg id "$deployment_id" '. + [{kind:$kind, id:$id}]' <<< "$cleanup_errors")"
      echo "无法将历史部署标记为停用，保留记录：$deployment_id"
      continue
    fi
  fi

  if gh api --method DELETE "repos/$REPOSITORY/deployments/$deployment_id" >/dev/null; then
    deleted_deployments="$(jq -c --arg id "$deployment_id" --arg created "$created_at" --arg environment "$environment_key" --arg task "$task_name" --arg sha "$commit_sha" '. + [{deployment_id:($id|tonumber), created_at:$created, environment:$environment, task:$task, commit_sha:$sha}]' <<< "$deleted_deployments")"
    echo "已清理旧部署：$deployment_id（$environment_key）"
  else
    cleanup_errors="$(jq -c --arg kind "deployment_delete" --arg id "$deployment_id" '. + [{kind:$kind, id:$id}]' <<< "$cleanup_errors")"
    echo "清理旧部署失败：$deployment_id"
  fi
done <<< "$deployments_tsv"

all_deleted_actions="$(jq -n --argjson prior "$prior_partial_actions" --argjson current "$deleted_actions" '$prior + $current')"
manifest_path="data/maintenance/action_deployment_cleanup/$BASELINE_RUN_ID.json"
mkdir -p "$(dirname "$manifest_path")"
jq -n \
  --arg repository "$REPOSITORY" \
  --arg baseline_run_id "$BASELINE_RUN_ID" \
  --arg cutoff_created_at "$cutoff" \
  --arg completed_at "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --argjson deleted_actions "$all_deleted_actions" \
  --argjson preserved_active_runs "$preserved_active" \
  --argjson deleted_deployments "$deleted_deployments" \
  --argjson preserved_deployments "$preserved_deployments" \
  --argjson prior_partial_attempts "$prior_partial_attempts" \
  --argjson errors "$cleanup_errors" \
  '{repository:$repository, baseline_run_id:($baseline_run_id|tonumber), cutoff_created_at:$cutoff_created_at, completed_at:$completed_at, deleted_actions:$deleted_actions, preserved_active_runs:$preserved_active_runs, deleted_deployments:$deleted_deployments, preserved_deployments:$preserved_deployments, prior_partial_cleanup_attempt_run_ids:$prior_partial_attempts, errors:$errors}' \
  > "$manifest_path"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add "$manifest_path"
if ! git diff --cached --quiet; then
  git commit -m "维护：记录历史 Action 与部署清理结果"
  pushed=false
  for attempt in 1 2 3; do
    if git push origin HEAD:main; then
      pushed=true
      break
    fi
    echo "审计清单推送遇到并发更新，第$attempt次回基重试。"
    git pull --rebase origin main
  done
  if [[ "$pushed" != "true" ]]; then
    echo "审计清单连续三次推送失败。"
    exit 1
  fi
fi

echo "历史运行清理数量（包含上次部分成功记录）：$(jq 'length' <<< "$all_deleted_actions")"
echo "保留的活动运行数量：$(jq 'length' <<< "$preserved_active")"
echo "已清理部署数量：$(jq 'length' <<< "$deleted_deployments")"
echo "保留部署基线数量：$(jq 'length' <<< "$preserved_deployments")"
echo "清理错误数量：$(jq 'length' <<< "$cleanup_errors")"
if [[ "$(jq 'length' <<< "$cleanup_errors")" -gt 0 ]]; then
  exit 1
fi
