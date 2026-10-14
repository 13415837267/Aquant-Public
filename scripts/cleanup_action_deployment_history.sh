#!/usr/bin/env bash
set -euo pipefail

: "\${GH_TOKEN:?GH_TOKEN is required}"
: "\${REPOSITORY:?REPOSITORY is required}"
: "\${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"

deleted_actions='[]'
deleted_deployments='[]'
preserved_active_deployments='[]'
preserved_other_deployments='[]'
preserved_current_deployment='null'
preserved_active='[]'
preserved_successful='[]'
preserved_other_completed='[]'
cancelled_obsolete_runs='[]'
cleanup_errors='[]'

# GitHub API 只读请求增加有限重试，降低短暂 5xx/网络波动导致整次清理中断的概率。
gh_api_read_with_retry() {
  local attempt=1
  local max_attempts="${GH_API_READ_MAX_ATTEMPTS:-4}"
  local delay_seconds="${GH_API_READ_RETRY_DELAY_SECONDS:-3}"
  local last_status=1

  while (( attempt <= max_attempts )); do
    if gh api "$@"; then
      return 0
    else
      last_status=$?
    fi

    if (( attempt >= max_attempts )); then
      echo "GitHub API 只读请求连续失败：第 ${attempt}/${max_attempts} 次。" >&2
      return "$last_status"
    fi

    echo "GitHub API 只读请求失败；等待 ${delay_seconds} 秒后重试第 $((attempt + 1))/${max_attempts} 次。" >&2
    sleep "$delay_seconds"
    attempt=$((attempt + 1))
  done

  return "$last_status"
}


# 删除失败运行前先保存诊断摘要；完整运行记录仍按项目规则清理，关键根因线索进入审计清单。
capture_run_diagnostics() {
  local target_run_id="$1"
  local log_file jobs_json capture_status excerpt
  log_file="$(mktemp)"
  jobs_json='[]'
  capture_status="partial"

  if jobs_json="$(gh_api_read_with_retry "repos/$REPOSITORY/actions/runs/$target_run_id/jobs?per_page=100" --jq '[.jobs[] | {id, name, status, conclusion, html_url}]' 2>/dev/null)"; then
    :
  else
    jobs_json='[]'
  fi

  if gh run view "$target_run_id" --repo "$REPOSITORY" --log >"$log_file" 2>&1; then
    capture_status="complete"
  fi

  excerpt="$(grep -iE 'error|failed|failure|traceback|exception|assertion|fatal|denied|not found|timed out|timeout|exit code|no such file|unexpected' "$log_file" | tail -n 30 | tail -c 7000 || true)"
  if [[ -z "$excerpt" ]]; then
    excerpt="$(tail -n 20 "$log_file" | tail -c 4000 || true)"
  fi

  jq -cn \
    --arg run_id "$target_run_id" \
    --arg capture_status "$capture_status" \
    --arg excerpt "$excerpt" \
    --argjson jobs "$jobs_json" \
    '{run_id:($run_id|tonumber), captured_at:(now|todateiso8601), log_capture_status:$capture_status, jobs:$jobs, failure_excerpt:$excerpt}'
  rm -f "$log_file"
}

# 可选的定向取消列表只允许来自维护工作流的显式配置；目标名称必须匹配。
cancel_poll_interval_seconds="\${CANCEL_POLL_INTERVAL_SECONDS:-5}"
cancel_poll_max_attempts="\${CANCEL_POLL_MAX_ATTEMPTS:-60}"
if [[ -n "\${CANCEL_OBSOLETE_RUN_IDS:-}" ]]; then
  for run_id in ${CANCEL_OBSOLETE_RUN_IDS:-}; do
    if [[ ! "$run_id" =~ ^[0-9]+$ ]]; then
      cleanup_errors="$(jq -c --arg id "$run_id" '. + [{kind:"cancel_run", id:$id, reason:"运行编号格式无效"}]' <<< "$cleanup_errors")"
      echo "取消旧运行失败：运行编号格式无效。"
      continue
    fi

    if ! run_info="$(gh_api_read_with_retry "repos/$REPOSITORY/actions/runs/$run_id" --jq '[.id, .name, .status, (.conclusion // "-")] | @tsv' 2>/dev/null)"; then
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
      echo "指定旧运行已结束：$run_id（$actual_conclusion）"
      continue
    fi

    if ! gh api --method POST "repos/$REPOSITORY/actions/runs/$run_id/force-cancel" >/dev/null 2>&1; then
      current_status="$(gh_api_read_with_retry "repos/$REPOSITORY/actions/runs/$run_id" --jq '.status' 2>/dev/null || printf 'unknown')"
      if [[ "$current_status" != "completed" ]]; then
        cleanup_errors="$(jq -c --arg id "$run_id" --arg status "$current_status" '. + [{kind:"cancel_run", id:$id, reason:("强制取消请求失败，当前状态：" + $status)}]' <<< "$cleanup_errors")"
        echo "强制取消旧运行失败：$run_id（当前状态：$current_status）"
        continue
      fi
    fi

    attempt=0
    current_status="$actual_status"
    while [[ "$attempt" -lt "$cancel_poll_max_attempts" ]]; do
      attempt=$((attempt + 1))
      current_status="$(gh_api_read_with_retry "repos/$REPOSITORY/actions/runs/$run_id" --jq '.status' 2>/dev/null || printf 'unknown')"
      if [[ "$current_status" == "completed" ]]; then
        final_conclusion="$(gh_api_read_with_retry "repos/$REPOSITORY/actions/runs/$run_id" --jq '.conclusion // "-"' 2>/dev/null || printf 'unknown')"
        cancelled_obsolete_runs="$(jq -c --arg id "$run_id" --arg name "$actual_name" --arg prior_status "$actual_status" --arg conclusion "$final_conclusion" '. + [{run_id:($id|tonumber), name:$name, prior_status:$prior_status, conclusion:$conclusion}]' <<< "$cancelled_obsolete_runs")"
        echo "旧研究运行已结束：$run_id（结论：$final_conclusion）"
        break
      fi
      sleep "$cancel_poll_interval_seconds"
    done

    if [[ "$current_status" != "completed" ]]; then
      cleanup_errors="$(jq -c --arg id "$run_id" --arg status "$current_status" '. + [{kind:"cancel_run", id:$id, reason:("等待强制取消完成超时，当前状态：" + $status)}]' <<< "$cleanup_errors")"
      echo "等待旧运行取消完成超时：$run_id"
    fi
  done
fi

# 取消完成后重新读取状态；仅删除已结束的失败、跳过、取消记录。
if ! runs_tsv="$(gh_api_read_with_retry --paginate "repos/$REPOSITORY/actions/runs?per_page=100" --jq '.workflow_runs[] | [.id, .created_at, .status, (.conclusion // "-"), (.name // "")] | @tsv')"; then
  echo "读取历史工作流运行失败。"
  exit 1
fi

while IFS=$'\t' read -r run_id created_at status conclusion run_name; do
  [[ -n "\${run_id:-}" ]] || continue

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
    echo "保留非清理目标运行：$run_id（\${conclusion:-无结论}）"
    continue
  fi

  run_diagnostic='null'
  if [[ "$conclusion" == "failure" ]]; then
    echo "删除失败运行前保存诊断摘要：$run_id $run_name"
    run_diagnostic="$(capture_run_diagnostics "$run_id")"
  fi

  if gh api --method DELETE "repos/$REPOSITORY/actions/runs/$run_id" >/dev/null; then
    deleted_actions="$(jq -c --arg id "$run_id" --arg created "$created_at" --arg conclusion "$conclusion" --arg name "$run_name" --argjson diagnostic "$run_diagnostic" '. + [{run_id:($id|tonumber), created_at:$created, conclusion:$conclusion, name:$name, diagnostic:$diagnostic}]' <<< "$deleted_actions")"
    echo "已清理历史运行：$run_id $run_name（$conclusion）"
  else
    cleanup_errors="$(jq -c --arg kind "action_run" --arg id "$run_id" --arg conclusion "$conclusion" '. + [{kind:$kind, id:$id, conclusion:$conclusion}]' <<< "$cleanup_errors")"
    echo "清理历史运行失败：$run_id"
  fi
done <<< "$runs_tsv"

# 旧部署清理：只处理 GitHub Pages 历史部署，保留 main 上最新的成功部署及所有未结束的部署。
if ! deployment_rows="$(gh_api_read_with_retry --paginate "repos/$REPOSITORY/deployments?per_page=100" --jq '.[] | [.id, .ref, .sha, .environment, (.task // ""), .created_at] | @tsv')"; then
  cleanup_errors="$(jq -c '. + [{kind:"deployment_list", reason:"读取部署记录失败，未执行部署删除"}]' <<< "$cleanup_errors")"
  deployment_rows=""
fi

deployment_inventory=""
current_success_deployment_id=""
while IFS=$'\t' read -r deployment_id deployment_ref deployment_sha deployment_environment deployment_task deployment_created; do
  [[ -n "${deployment_id:-}" ]] || continue
  if ! deployment_state="$(gh_api_read_with_retry "repos/$REPOSITORY/deployments/$deployment_id/statuses?per_page=1" --jq '.[0].state // "none"' 2>/dev/null)"; then
    deployment_state="unknown"
    cleanup_errors="$(jq -c --arg id "$deployment_id" '. + [{kind:"deployment_status", deployment_id:($id|tonumber), reason:"读取部署状态失败，保留该部署"}]' <<< "$cleanup_errors")"
  fi
  deployment_inventory+="$deployment_id"$'\t'"$deployment_ref"$'\t'"$deployment_sha"$'\t'"$deployment_environment"$'\t'"$deployment_task"$'\t'"$deployment_created"$'\t'"$deployment_state"$'\n'

  # deployments 接口默认按创建时间由新到旧返回；只认 main + github-pages + 最新状态 success。
  if [[ -z "$current_success_deployment_id" && "$deployment_environment" == "github-pages" && "$deployment_ref" == "main" && "$deployment_state" == "success" ]]; then
    current_success_deployment_id="$deployment_id"
  fi
done <<< "$deployment_rows"

if [[ -z "$current_success_deployment_id" ]]; then
  echo "未找到 main 分支上状态成功的 GitHub Pages 部署；为防止误删可用部署，将保留所有成功状态的部署。"
else
  echo "保留当前成功的 GitHub Pages 部署：$current_success_deployment_id"
fi

while IFS=$'\t' read -r deployment_id deployment_ref deployment_sha deployment_environment deployment_task deployment_created deployment_state; do
  [[ -n "${deployment_id:-}" ]] || continue

  if [[ "$deployment_environment" != "github-pages" ]]; then
    preserved_other_deployments="$(jq -c --arg id "$deployment_id" --arg environment "$deployment_environment" --arg ref "$deployment_ref" --arg state "$deployment_state" '. + [{deployment_id:($id|tonumber), environment:$environment, ref:$ref, state:$state}]' <<< "$preserved_other_deployments")"
    continue
  fi

  case "$deployment_state" in
    queued|pending|in_progress|unknown)
      preserved_active_deployments="$(jq -c --arg id "$deployment_id" --arg environment "$deployment_environment" --arg ref "$deployment_ref" --arg state "$deployment_state" '. + [{deployment_id:($id|tonumber), environment:$environment, ref:$ref, state:$state}]' <<< "$preserved_active_deployments")"
      echo "保留未结束或状态未知的部署：$deployment_id（$deployment_state）"
      continue
      ;;
  esac

  if [[ "$deployment_id" == "$current_success_deployment_id" ]]; then
    preserved_current_deployment="$(jq -cn --arg id "$deployment_id" --arg sha "$deployment_sha" --arg created "$deployment_created" --arg ref "$deployment_ref" '{deployment_id:($id|tonumber), sha:$sha, ref:$ref, created_at:$created, state:"success"}')"
    continue
  fi

  if [[ "$deployment_state" == "success" && -z "$current_success_deployment_id" ]]; then
    preserved_other_deployments="$(jq -c --arg id "$deployment_id" --arg environment "$deployment_environment" --arg ref "$deployment_ref" --arg state "$deployment_state" '. + [{deployment_id:($id|tonumber), environment:$environment, ref:$ref, state:$state, reason:"未找到可确认保留的 main 成功部署"}]' <<< "$preserved_other_deployments")"
    continue
  fi

  # 先将历史成功部署标为 inactive；失败、错误、已失效的部署直接删除。
  if [[ "$deployment_state" != "inactive" && "$deployment_state" != "failure" && "$deployment_state" != "error" ]]; then
    if ! gh api --method POST "repos/$REPOSITORY/deployments/$deployment_id/statuses" -f state=inactive -f description="清理旧部署记录；当前成功部署已另行保留" >/dev/null; then
      cleanup_errors="$(jq -c --arg id "$deployment_id" '. + [{kind:"deployment_inactive", deployment_id:($id|tonumber), reason:"无法将历史部署标记为 inactive，未删除"}]' <<< "$cleanup_errors")"
      echo "标记旧部署为 inactive 失败，保留记录：$deployment_id"
      continue
    fi
  fi

  if gh api --method DELETE "repos/$REPOSITORY/deployments/$deployment_id" >/dev/null; then
    deleted_deployments="$(jq -c --arg id "$deployment_id" --arg created "$deployment_created" --arg sha "$deployment_sha" --arg ref "$deployment_ref" --arg state "$deployment_state" '. + [{deployment_id:($id|tonumber), created_at:$created, sha:$sha, ref:$ref, prior_state:$state}]' <<< "$deleted_deployments")"
    echo "已清理旧 GitHub Pages 部署记录：$deployment_id（原状态：$deployment_state）"
  else
    cleanup_errors="$(jq -c --arg id "$deployment_id" --arg state "$deployment_state" '. + [{kind:"deployment_delete", deployment_id:($id|tonumber), prior_state:$state, reason:"删除历史部署记录失败"}]' <<< "$cleanup_errors")"
    echo "清理旧部署记录失败：$deployment_id"
  fi
done <<< "$deployment_inventory"

manifest_path="data/maintenance/action_deployment_cleanup/run-$GITHUB_RUN_ID.json"
mkdir -p "$(dirname "$manifest_path")"
jq -n \
  --arg repository "$REPOSITORY" \
  --arg cleanup_run_id "$GITHUB_RUN_ID" \
  --arg completed_at "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --argjson deleted_actions "$deleted_actions" \
  --argjson deleted_deployments "$deleted_deployments" \
  --argjson preserved_active_deployments "$preserved_active_deployments" \
  --argjson preserved_other_deployments "$preserved_other_deployments" \
  --argjson preserved_current_deployment "$preserved_current_deployment" \
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
    deleted_deployments:$deleted_deployments,
    preserved_active_deployments:$preserved_active_deployments,
    preserved_other_deployments:$preserved_other_deployments,
    preserved_current_deployment:$preserved_current_deployment,
    cancelled_obsolete_runs:$cancelled_obsolete_runs,
    preserved_active_runs:$preserved_active_runs,
    preserved_successful_runs:$preserved_successful_runs,
    preserved_other_completed_runs:$preserved_other_completed_runs,
    errors:$errors
  }' > "$manifest_path"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
# 审计归档：较早的重复全量状态快照压缩为 archive.json，保留逐条删项/错误和最近 10 份完整清单。
audit_dir="$(dirname "$manifest_path")"
archive_path="$audit_dir/archive.json"
archive_updated_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
if [[ -f "$archive_path" ]]; then
  archive_json="$(jq -c --arg updated_at "$archive_updated_at" '.updated_at=$updated_at' "$archive_path")"
else
  archive_json="$(jq -cn --arg updated_at "$archive_updated_at" '{schema_version:1, updated_at:$updated_at, runs:[]}')"
fi
archive_changed=false
archive_count=0

mapfile -t audit_manifests < <(find "$audit_dir" -maxdepth 1 -type f -name 'run-*.json' -printf '%f\n' | sort -V)
if (( ${#audit_manifests[@]} > 10 )); then
  archive_count=$((${#audit_manifests[@]} - 10))
  for ((index=0; index<archive_count; index++)); do
    archived_path="$audit_dir/${audit_manifests[$index]}"
    archive_entry="$(jq -c '{
      cleanup_run_id,
      completed_at,
      deleted_actions:(.deleted_actions // []),
      deleted_deployments:(.deleted_deployments // []),
      cancelled_obsolete_runs:(.cancelled_obsolete_runs // []),
      errors:(.errors // []),
      preserved_active_run_count:((.preserved_active_runs // [])|length),
      preserved_successful_run_count:((.preserved_successful_runs // [])|length),
      preserved_other_completed_run_count:((.preserved_other_completed_runs // [])|length)
    }' "$archived_path")"
    archive_json="$(jq -c --argjson entry "$archive_entry" '.runs = ((.runs + [$entry]) | unique_by(.cleanup_run_id) | sort_by(.cleanup_run_id))' <<< "$archive_json")"
    rm "$archived_path"
    archive_changed=true
  done
fi

# 迁移旧命名格式的维护记录，保留旧清理批次中被删除的运行和部署明细。
while IFS= read -r legacy_path; do
  [[ -n "$legacy_path" ]] || continue
  legacy_entry="$(jq -c '{
    cleanup_run_id:(.cleanup_run_id // .baseline_run_id // .run_id),
    completed_at,
    legacy_record:true,
    cutoff_created_at:(.cutoff_created_at // null),
    deleted_actions:(.deleted_actions // []),
    deleted_deployments:(.deleted_deployments // []),
    preserved_deployments:(.preserved_deployments // []),
    cancelled_obsolete_runs:(.cancelled_obsolete_runs // []),
    errors:(.errors // [])
  }' "$legacy_path")"
  archive_json="$(jq -c --argjson entry "$legacy_entry" '.runs = ((.runs + [$entry]) | unique_by(.cleanup_run_id) | sort_by(.cleanup_run_id))' <<< "$archive_json")"
  rm "$legacy_path"
  archive_changed=true
done < <(find "$audit_dir" -maxdepth 1 -type f -name '*.json' ! -name 'archive.json' ! -name 'run-*.json' -print | sort)

if [[ "$archive_changed" == "true" ]]; then
  archive_json="$(jq -c --arg updated_at "$archive_updated_at" '.updated_at=$updated_at' <<< "$archive_json")"
  printf '%s\n' "$archive_json" > "$archive_path"
  echo "旧维护审计已压缩归档：压缩完整快照 $archive_count 份，并迁移旧格式记录；逐条操作编号和错误详情均保留。"
fi

git add -A "$audit_dir"
if ! git diff --cached --quiet; then
  git commit -m "维护：记录失败与跳过 Action 清理结果"
  pushed=false
  for attempt in 1 2 3; do
    if git push origin HEAD:main; then
      pushed=true
      break
    fi
    echo "审计清单推送遇到并发更新，第\${attempt}次回基重试。"
    git pull --rebase origin main
  done
  if [[ "$pushed" != "true" ]]; then
    echo "审计清单连续三次推送失败。"
    exit 1
  fi
fi

cancelled_count="$(jq 'length' <<< "$cancelled_obsolete_runs")"
deleted_count="$(jq 'length' <<< "$deleted_actions")"
failed_count="$(jq '[.[] | select(.conclusion == "failure")] | length' <<< "$deleted_actions")"
skipped_count="$(jq '[.[] | select(.conclusion == "skipped")] | length' <<< "$deleted_actions")"
deleted_cancelled_count="$(jq '[.[] | select(.conclusion == "cancelled")] | length' <<< "$deleted_actions")"
active_count="$(jq 'length' <<< "$preserved_active")"
active_deployment_count="$(jq 'length' <<< "$preserved_active_deployments")"
deployment_deleted_count="$(jq 'length' <<< "$deleted_deployments")"
successful_count="$(jq 'length' <<< "$preserved_successful")"
other_completed_count="$(jq 'length' <<< "$preserved_other_completed")"
error_count="$(jq 'length' <<< "$cleanup_errors")"

echo "已取消的旧研究运行数：$cancelled_count"
echo "已清理失败、跳过或取消的 Action 记录数：$deleted_count"
echo "已清理的旧部署记录数：$deployment_deleted_count"
echo "保留的进行中或状态未知部署数：$active_deployment_count"
echo "其中失败：$failed_count"
echo "其中跳过：$skipped_count"
echo "其中取消：$deleted_cancelled_count"
echo "保留的运行中或排队中记录数：$active_count"
echo "保留的成功记录数：$successful_count"
echo "保留的其他已完成记录数：$other_completed_count"
echo "清理错误数：$error_count"
if [[ "$error_count" -gt 0 ]]; then
  exit 1
fi
