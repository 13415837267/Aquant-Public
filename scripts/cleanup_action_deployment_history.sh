#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${REPOSITORY:?REPOSITORY is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"

deleted_actions='[]'
preserved_active='[]'
preserved_successful='[]'
preserved_other_completed='[]'
cancelled_obsolete_runs='[]'
cleanup_errors='[]'

# 仅处理工作流中显式配置的旧运行编号；校验名称以防止误取消其他任务。
cancel_poll_interval_seconds="${CANCEL_POLL_INTERVAL_SECONDS:-5}"
cancel_poll_max_attempts="${CANCEL_POLL_MAX_ATTEMPTS:-60}"
if [[ -n "${CANCEL_OBSOLETE_RUN_IDS:-}" ]]; then
  for run_id in $CANCEL_OBSOLETE_RUN_IDS; do
    if [[ ! "$run_id" =~ ^[0-9]+$ ]]; then
      cleanup_errors="$(jq -c --arg id "$run_id" '. + [{kind:"cancel_run", id:$id, reason:"运行编号格式无效"}]' <<< "$cleanup_errors")"
      echo "取消旧运行失败：运行编号格式无效。"
      continue
    fi

    if ! run_info="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '[.id, .name, .status, (.conclusion // "-")] | @tsv' 2>/dev/null)"; then
      echo "指定旧运行已不存在，无需再次取消：$run_id"
      continue
    fi
    IFS=$'\t' read -r actual_id actual_name actual_status actual_conclusion <<< "$run_info"

    if [[ "$actual_name" != "06 模型研究" ]]; then
      cleanup_errors="$(jq -c --arg id "$run_id" --arg name "$actual_name" '. + [{kind:"cancel_run", id:$id, reason:("目标名称不匹配：" + $name)}]' <<< "$cleanup_errors")"
      echo "拒绝取消运行 $run_id：工作流名称不符合预期。"
      continue
    fi
    if [[ "$actual_status" == "completed" ]]; then
      echo "指定旧运行已结束：$run_id（\${actual_conclusion}）"
      continue
    fi

    if ! gh api --method POST "repos/$REPOSITORY/actions/runs/$run_id/cancel" >/dev/null 2>&1; then
      current_status="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '.status' 2>/dev/null || printf 'unknown')"
      if [[ "$current_status" != "completed" ]]; then
        cleanup_errors="$(jq -c --arg id "$run_id" --arg status "$current_status" '. + [{kind:"cancel_run", id:$id, reason:("取消请求失败，当前状态：" + $status)}]' <<< "$cleanup_errors")"
        echo "取消旧运行失败：$run_id（当前状态：$current_status）"
        continue
      fi
    fi

    attempt=0
    current_status="$actual_status"
    while [[ "$attempt" -lt "$cancel_poll_max_attempts" ]]; do
      attempt=$((attempt + 1))
      current_status="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '.status' 2>/dev/null || printf 'unknown')"
      if [[ "$current_status" == "completed" ]]; then
        final_conclusion="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '.conclusion // "-"' 2>/dev/null || printf 'unknown')"
        cancelled_obsolete_runs="$(jq -c --arg id "$run_id" --arg name "$actual_name" --arg prior_status "$actual_status" --arg conclusion "$final_conclusion" '. + [{run_id:($id|tonumber), name:$name, prior_status:$prior_status, conclusion:$conclusion}]' <<< "$cancelled_obsolete_runs")"
        echo "旧研究运行已结束：$run_id（结论：$final_conclusion）"
        break
      fi
      sleep "$cancel_poll_interval_seconds"
    done

    if [[ "$current_status" != "completed" ]]; then
      cleanup_errors="$(jq -c --arg id "$run_id" --arg status "$current_status" '. + [{kind:"cancel_run", id:$id, reason:("等待取消完成超时，当前状态：" + $status)}]' <<< "$cleanup_errors")"
      echo "等待旧运行取消完成超时：$run_id"
    fi
  done
fi

# 取消请求完成后重新读取运行列表，再依据状态逐条决定是否删除记录。
if ! runs_tsv="$(gh api --paginate "repos/$REPOSITORY/actions/runs?per_page=100" --jq '.workflow_runs[] | [.id, .created_at, .status, (.conclusion // "-"), (.name // "")] | @tsv')"; then
  echo "读取历史工作流运行失败。"
  exit 1
fi

while IFS=$'\t' read -r run_id created_at status conclusion run_name; do
  [[ -n "${run_id:-}" ]] || continue

  # 只清理已完成且结论为失败、跳过或取消的历史 Action 运行。
  # 运行中、排队中和成功记录必须保留；其他结论也不在本轮清理范围。
  if [[ "$status" != "completed" ]]; then
    preserved_active="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg status "$status" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, status:$status, name:$name}]' <<< "$preserved_active")"
    echo "保留尚未完成的运行：$run_id（$status）"
    continue
  fi

  if [[ "$conclusion" != "failure" && "$conclusion" != "skipped" && "$conclusion" != "cancelled" ]]; then
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
  --argjson cancelled_obsolete_runs "$cancelled_obsolete_runs" \
  --argjson preserved_active_runs "$preserved_active" \
  --argjson preserved_successful_runs "$preserved_successful" \
  --argjson preserved_other_completed_runs "$preserved_other_completed" \
  --argjson errors "$cleanup_errors" \
  '{
    repository:$repository,
    cleanup_run_id:($cleanup_run_id|tonumber),
    completed_at:$completed_at,
    deleted_actions:$deleted_actions,
    cancelled_obsolete_runs:$cancelled_obsolete_runs,
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

echo "已取消的旧研究运行数：$(jq 'length' <<< "$cancelled_obsolete_runs")"
echo "已清理失败、跳过或取消的 Action 记录数：$(jq 'length' <<< "$deleted_actions")"
echo "其中失败：$(jq '[.[] | select(.conclusion == "failure")] | length' <<< "$deleted_actions")"
echo "其中跳过：$(jq '[.[] | select(.conclusion == "skipped")] | length' <<< "$deleted_actions")"
echo "其中取消：$(jq '[.[] | select(.conclusion == "cancelled")] | length' <<< "$deleted_actions")"
echo "保留的运行中或排队中记录数：$(jq 'length' <<< "$preserved_active")"
echo "保留的成功记录数：$(jq 'length' <<< "$preserved_successful")"
echo "保留的其他已完成记录数：$(jq 'length' <<< "$preserved_other_completed")"
echo "清理错误数：$(jq 'length' <<< "$cleanup_errors")"
if [[ "$(jq 'length' <<< "$cleanup_errors")" -gt 0 ]]; then
  exit 1
fi
\t' read -r actual_id actual_name actual_status actual_conclusion <<< "$run_info"

    if [[ "$actual_name" != "06 模型研究" ]]; then
      cleanup_errors="$(jq -c --arg id "$run_id" --arg name "$actual_name" '. + [{kind:"cancel_run", id:$id, reason:("目标名称不匹配：" + $name)}]' <<< "$cleanup_errors")"
      echo "拒绝取消运行 $run_id：工作流名称不符合预期。"
      continue
    fi
    if [[ "$actual_status" == "completed" ]]; then
      echo "指定旧运行已结束：$run_id（${actual_conclusion}）"
      continue
    fi

    if ! gh api --method POST "repos/$REPOSITORY/actions/runs/$run_id/cancel" >/dev/null 2>&1; then
      current_status="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '.status' 2>/dev/null || printf 'unknown')"
      if [[ "$current_status" != "completed" ]]; then
        cleanup_errors="$(jq -c --arg id "$run_id" --arg status "$current_status" '. + [{kind:"cancel_run", id:$id, reason:("取消请求失败，当前状态：" + $status)}]' <<< "$cleanup_errors")"
        echo "取消旧运行失败：$run_id（当前状态：$current_status）"
        continue
      fi
    fi

    attempt=0
    current_status="$actual_status"
    while [[ "$attempt" -lt "$cancel_poll_max_attempts" ]]; do
      attempt=$((attempt + 1))
      current_status="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '.status' 2>/dev/null || printf 'unknown')"
      if [[ "$current_status" == "completed" ]]; then
        final_conclusion="$(gh api "repos/$REPOSITORY/actions/runs/$run_id" --jq '.conclusion // "-"' 2>/dev/null || printf 'unknown')"
        cancelled_obsolete_runs="$(jq -c --arg id "$run_id" --arg name "$actual_name" --arg prior_status "$actual_status" --arg conclusion "$final_conclusion" '. + [{run_id:($id|tonumber), name:$name, prior_status:$prior_status, conclusion:$conclusion}]' <<< "$cancelled_obsolete_runs")"
        echo "旧研究运行已结束：$run_id（结论：$final_conclusion）"
        break
      fi
      sleep "$cancel_poll_interval_seconds"
    done

    if [[ "$current_status" != "completed" ]]; then
      cleanup_errors="$(jq -c --arg id "$run_id" --arg status "$current_status" '. + [{kind:"cancel_run", id:$id, reason:("等待取消完成超时，当前状态：" + $status)}]' <<< "$cleanup_errors")"
      echo "等待旧运行取消完成超时：$run_id"
    fi
  done
fi

# 取消请求完成后重新读取运行列表，再依据状态逐条决定是否删除记录。
if ! runs_tsv="$(gh api --paginate "repos/$REPOSITORY/actions/runs?per_page=100" --jq '.workflow_runs[] | [.id, .created_at, .status, (.conclusion // "-"), (.name // "")] | @tsv')"; then
  echo "读取历史工作流运行失败。"
  exit 1
fi

while IFS=$'\t' read -r run_id created_at status conclusion run_name; do
  [[ -n "${run_id:-}" ]] || continue

  # 只清理已完成且结论为失败、跳过或取消的历史 Action 运行。
  # 运行中、排队中和成功记录必须保留；其他结论也不在本轮清理范围。
  if [[ "$status" != "completed" ]]; then
    preserved_active="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg status "$status" --arg name "$run_name" '. + [{run_id:($id|tonumber), created_at:$created, status:$status, name:$name}]' <<< "$preserved_active")"
    echo "保留尚未完成的运行：$run_id（$status）"
    continue
  fi

  if [[ "$conclusion" != "failure" && "$conclusion" != "skipped" && "$conclusion" != "cancelled" ]]; then
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

echo "已清理失败、跳过或取消的 Action 记录数：$(jq 'length' <<< "$deleted_actions")"
echo "其中失败：$(jq '[.[] | select(.conclusion == "failure")] | length' <<< "$deleted_actions")"
echo "其中跳过：$(jq '[.[] | select(.conclusion == "skipped")] | length' <<< "$deleted_actions")"
echo "其中取消：$(jq '[.[] | select(.conclusion == "cancelled")] | length' <<< "$deleted_actions")"
echo "保留的运行中或排队中记录数：$(jq 'length' <<< "$preserved_active")"
echo "保留的成功记录数：$(jq 'length' <<< "$preserved_successful")"
echo "保留的其他已完成记录数：$(jq 'length' <<< "$preserved_other_completed")"
echo "清理错误数：$(jq 'length' <<< "$cleanup_errors")"
if [[ "$(jq 'length' <<< "$cleanup_errors")" -gt 0 ]]; then
  exit 1
fi
