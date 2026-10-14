import candidatesData from "@/data/candidates.json";
import auditData from "@/data/system_audit.json";
import productionStatusData from "@/data/production_status.json";
import researchStatusData from "@/data/research_status.json";
import nextTradingPlanData from "@/data/next_trading_day_plan.json";
import researchTrialData from "@/data/research_trial_status.json";
import systemVersionData from "@/config/system_version.json";

type Candidate = {
  rank:number; symbol:string; name:string; price:number; change_pct:number; overnight_1d_pct?:number; intraday_return_pct?:number;
  return_3d_pct:number; return_5d_pct:number; return_10d_pct:number;
  volume_ratio_5d:number; turnover_pct:number; amount:number;
  volatility_10d_pct:number; close_strength:number; score:number; precision_probability:number; admission_tier:string; flags?:string[];
};
type Snapshot = {
  as_of:string; status:string; system_version?:string; lifecycle_status?:string; strategy_version:string; strategy_commit:string; signal_horizon:string;
  candidate_admission_policy:string; candidates:Candidate[];
  model_definition?:{type:string;features?:string[];primary_probability_threshold?:number;maximum_candidates?:number;release_status?:string};
  factor_weights?:Record<string,number> | null;
  market?:{breadth_pct:number;median_return_pct:number;regime:string};
  diagnostics?:{scorable_rows?:number;candidate_count?:number;risk_off_no_trade?:boolean};
  future_function?:boolean;
};


type NextTradingPlan = {
  next_trading_day:string; signal_date:string; data_cutoff:string; status:string; title:string; summary:string;
  strategy_version:string; strategy_commit:string; candidate_policy:string; candidate_count:number;
  market:{breadth_pct:number;median_return_pct:number;regime:string};
  candidates:Array<{rank:number;symbol:string;name:string;price:number;score:number;precision_probability:number;admission_tier:string;signal_date:string;execution_date:string;earliest_exit_date:string;net_win_threshold_pct:number;stop_loss_pct:number;flags:string[]}>;
  source_snapshot?:{path:string;as_of:string;strategy_version:string;strategy_commit:string};
  steps:Array<{time:string;action:string}>; hard_rules:string[]; production_release:boolean;
};

type ResearchTrial = { status:string; as_of:string; system_version:string; strategy_version:string; strategy_commit:string; model_code_commit?:string; data_cutoff?:string; training_window?:{start:string;end:string}; selected_operating_point?:{probability_threshold:number;final_samples:number;final_sample_share_pct:number;final_win_rate_pct:number;final_wilson_lower_pct:number;final_mean_best_return_pct:number;final_mean_5d_close_return_pct:number}; validation?:{samples:number;win_rate_pct:number;wilson_lower_pct:number}; audits?:{no_future_features:boolean;entry_is_T_plus_1_open:boolean;exit_starts_T_plus_2:boolean;strict_profit_label:boolean}; production_release:boolean; formal_gate_pct?:number; note:string; };

type ResearchStatus = {
  status:string;
  as_of:string;
  strategy_version:string;
  strategy_commit:string;
  preferred_operating_point?:{name:string;final:{samples:number;win_rate_pct:number;wilson_lower_pct:number}};
  highest_observed_operating_point?:{name:string;final:{samples:number;win_rate_pct:number;wilson_lower_pct:number}};
  daily_top1?:{final:{samples:number;win_rate_pct:number;wilson_lower_pct:number}};
  path_aware_caveat?:{final_win_rate_pct:number;note:string};
  production_release:boolean;
};

const snapshot=candidatesData as Snapshot;
const audit=auditData as {production_gate?:string; strategy_version?:string; strategy_commit?:string; audit?:{production_gate_passed?:boolean}};
const provenanceMatches=audit.strategy_version===snapshot.strategy_version && audit.strategy_commit===snapshot.strategy_commit && snapshot.strategy_version===systemVersion.strategy_version && snapshot.strategy_commit===systemVersion.strategy_commit;
const productionStatus=productionStatusData as {status:string;system_version:string;strategy_version:string;production_version:string|null;release_gate:boolean;system_audit:boolean;strategy_quality_gate_passed:boolean};
const researchStatus=researchStatusData as ResearchStatus;
const nextTradingPlan=nextTradingPlanData as NextTradingPlan;
const researchTrial=researchTrialData as ResearchTrial;
const systemVersion=systemVersionData as {system_version:string;strategy_version:string;lifecycle_status:string;release_status:string;strategy_commit:string;candidate_admission_policy:string;production_release:boolean;automatic_ordering:boolean};
const productionReady=productionStatus.status==="released" && productionStatus.release_gate===true && productionStatus.system_audit===true;
const liveOrderingEnabled=productionReady && productionStatus.strategy_quality_gate_passed===true;
const fmt=(n:number,d=2)=>n.toLocaleString("zh-CN",{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=(n:number)=>(n>=0?"+":"")+fmt(n)+"%";
const dirClass=(n:number)=>n>0?"rise":n<0?"fall":"flat";
const amount=(n:number)=>n>=1e8?fmt(n/1e8,1)+"亿":fmt(n/1e4,0)+"万";
const weight=(n:number)=>fmt(n*100,0)+"%";
const maxCandidates=Number((snapshot.candidate_admission_policy.match(/precision_top_(\d+)/)||snapshot.candidate_admission_policy.match(/dynamic_top_score_(\d+)/)||snapshot.candidate_admission_policy.match(/top_(\d+)/)||[])[1]||0);

export default function Home(){
  const strategyMetadataReady=!!snapshot.strategy_version && !!snapshot.strategy_commit && !!researchTrial.strategy_version && !!researchTrial.strategy_commit;
  const candidatesAreCurrent=snapshot.status==="ready" && strategyMetadataReady && provenanceMatches && snapshot.system_version===systemVersion.system_version && snapshot.strategy_version===researchTrial.strategy_version && snapshot.strategy_commit===researchTrial.strategy_commit && snapshot.future_function===false;
  const selectedPoint=researchTrial.selected_operating_point;
  const validationMetrics=researchTrial.validation;
  const currentFeatures=candidatesAreCurrent?(snapshot.model_definition?.features??[]):[];
  const featureLabels:Record<string,string>={return_1d_pct:"近1日收益",return_3d_pct:"近3日收益",return_5d_pct:"近5日收益",return_10d_pct:"近10日收益",return_20d_pct:"近20日收益",overnight_1d_pct:"近1日隔夜收益",overnight_3d_pct:"近3日隔夜收益",overnight_5d_pct:"近5日隔夜收益",overnight_10d_pct:"近10日隔夜收益",volume_ratio_5d:"5日量比",amount_20d:"20日成交额",volatility_10d_pct:"10日波动率",close_strength:"收盘位置强度",intraday_return_pct:"日内涨跌幅",limit_up_5d_count:"近5日涨停次数",turnover_pct:"换手率",change_pct:"当日涨跌幅",market_breadth_pct:"市场上涨广度",market_median_return_pct:"市场中位收益"};
  const rows=candidatesAreCurrent?(snapshot.candidates??[]):[];
  const planMatchesSnapshot=candidatesAreCurrent && nextTradingPlan.strategy_version===snapshot.strategy_version && nextTradingPlan.strategy_commit===snapshot.strategy_commit && nextTradingPlan.data_cutoff===snapshot.as_of.slice(0,10) && nextTradingPlan.candidate_count===rows.length && nextTradingPlan.candidates.length===rows.length && nextTradingPlan.candidates.every((candidate,index)=>{const row=rows[index]; return candidate.rank===row.rank && candidate.symbol===row.symbol && candidate.name===row.name && candidate.price===row.price && candidate.score===row.score && candidate.precision_probability===row.precision_probability && candidate.admission_tier===row.admission_tier;});
  const m=snapshot.market;
  const preferred=researchStatus.preferred_operating_point?.final;
  const highest=researchStatus.highest_observed_operating_point?.final;
  const dailyTop1=researchStatus.daily_top1?.final;
  return <main className="container">
    <header className="header">
      <div><div className="eyebrow">Aquant / 短线运行系统</div><h1>短线候选池</h1><div className="color-legend"><span className="legend-rise">红涨</span><span className="legend-fall">绿跌</span><span className="legend-flat">平盘中性</span></div>
      <p className="subtitle">T 日收盘生成信号，T+1 开盘执行，T+2 起最早退出；当前仅研究/模拟，不自动下单。</p></div>
      <div className="pill">{systemVersion.system_version===systemVersion.strategy_version?"研究阶段":"版本不一致"} · 系统/策略 {systemVersion.system_version}</div>
    </header>

    <section className="grid">
      <div className="card"><div className="metric-label">当前候选数量</div><div className="metric-value">{candidatesAreCurrent?rows.length:"—"}</div><div className="metric-note">{candidatesAreCurrent?("本轮候选上限 "+maxCandidates):"等待当前版本候选快照"}</div></div>
      <div className="card"><div className="metric-label">可评分股票</div><div className="metric-value">{snapshot.diagnostics?.scorable_rows??"—"}</div><div className="metric-note">20日短线特征</div></div>
      <div className="card"><div className="metric-label">市场状态</div><div className="metric-value">{m?.regime==="neutral"?"中性":m?.regime==="risk_on"?"风险偏好":m?.regime==="risk_off"?"风险规避":m?.regime??"—"}</div><div className="metric-note">广度 {m?fmt(m.breadth_pct,1)+"%":"—"}</div></div>
      <div className="card"><div className="metric-label">交易窗口</div><div className="metric-value">1–5日</div><div className="metric-note">T+1 开盘执行</div></div>
    </section>

    <section className="card next-plan-card">
      <div className="table-head">
        <div><div className="table-title">下一交易日计划</div><div className="table-subtitle">{nextTradingPlan.title}</div></div>
        <div className="badge">交易日 {nextTradingPlan.next_trading_day}</div>
      </div>
      <div className="next-plan-summary">{candidatesAreCurrent?nextTradingPlan.summary:"当前候选与计划仍来自旧策略版本，已暂停展示；待云端生成与2.6.0一致的研究快照后恢复。"}</div>
      <div className="plan-grid">
        {!candidatesAreCurrent?<div className="empty">候选快照尚未通过当前策略与系统审计一致性检查，计划中的候选暂不单独展示，避免出现两套信号。</div>:!planMatchesSnapshot?<div className="empty">下一交易日计划与候选快照未逐项一致，系统拒绝展示分叉信号。</div>:rows.map((candidate)=><div className="plan-item" key={candidate.symbol}><div className="plan-time">计划候选 #{candidate.rank}</div><div className="plan-action"><strong>{candidate.symbol} {candidate.name}</strong> · 收盘价 {fmt(candidate.price)} · 模型评分 {fmt(candidate.precision_probability*100,2)}分（模型概率 {fmt(candidate.precision_probability*100,2)}%）。准入层级：{candidate.admission_tier}。信号日 {nextTradingPlan.signal_date} → 执行日 {nextTradingPlan.next_trading_day} → 最早退出 {nextTradingPlan.candidates[candidate.rank-1].earliest_exit_date}。单笔净利润目标 +{fmt(nextTradingPlan.candidates[candidate.rank-1].net_win_threshold_pct,1)}%，止损 {fmt(nextTradingPlan.candidates[candidate.rank-1].stop_loss_pct,1)}%。</div></div>)}
      </div>
      {candidatesAreCurrent&&planMatchesSnapshot?<><div className="next-plan-summary">数据截止 {nextTradingPlan.data_cutoff} · 策略 {nextTradingPlan.strategy_version} · 市场 {nextTradingPlan.market.regime==="neutral"?"中性":nextTradingPlan.market.regime==="risk_on"?"风险偏好":"风险规避"} · 候选数 {nextTradingPlan.candidate_count} · 唯一候选来源 {nextTradingPlan.source_snapshot?.path??"data/candidates.json"}</div><div className="plan-grid">{nextTradingPlan.steps.map((step)=><div className="plan-item" key={step.time}><div className="plan-time">{step.time}</div><div className="plan-action">{step.action}</div></div>)}</div></>:null}
      <div className="next-plan-warning">当前生命周期：<strong>研究阶段，未发布正式版本</strong>。系统版本与策略版本必须均为 {systemVersion.system_version}；禁止实盘与自动下单。当前候选只供研究/模拟，且必须来自信号日收盘时已知的数据。</div>
    </section>

    <section className="card research-card">
      <div className="table-head"><div><div className="table-title">当前策略研究状态</div><div className="table-subtitle">系统版本 {systemVersion.system_version} · 策略版本 {systemVersion.strategy_version} · 策略提交 {systemVersion.strategy_commit.slice(0,10)}</div></div><div className="badge">仅研究</div></div>
      {selectedPoint&&validationMetrics&&researchTrial.status!=="research_pending"?<><div className="research-grid"><div><div className="metric-label">补充期净收益目标命中率</div><div className="metric-value">{fmt(selectedPoint.final_win_rate_pct,2)}%</div><div className="metric-note">{selectedPoint.final_samples.toLocaleString("zh-CN")} 个标签样本；仅作标签统计</div></div><div><div className="metric-label">验证期标签命中率</div><div className="metric-value">{fmt(validationMetrics.win_rate_pct,2)}%</div><div className="metric-note">{validationMetrics.samples.toLocaleString("zh-CN")} 个标签样本</div></div><div><div className="metric-label">当前参数阈值</div><div className="metric-value">{fmt(selectedPoint.probability_threshold,2)}</div><div className="metric-note">需结合可执行组合复利与回撤判断</div></div></div><div className="research-warning">这些数字只有在模型版本和策略提交均匹配时才用于当前研究判断；标签命中率不等于实际交易盈利率。</div></>:<div className="research-warning"><strong>2.6.0 尚无与当前策略提交完全匹配的完整验证结果。</strong> 旧近期训练窗口神经网络的历史指标已归档，不挪作当前策略成绩。当前版本须重新完成云端历史验证、执行约束组合回测、成本压力测试、参数稳健性检查与未来影子模拟。</div>}
      <div className="research-warning"><strong>当前状态：未发布。</strong> 版本对齐仅代表系统已绑定唯一策略源，不代表策略已经通过盈利门槛；不得自动晋级或实盘下单。</div>
    </section>
    <section className="card research-card">
      <div className="table-head">
        <div><div className="table-title">历史策略审计结果（归档对照）</div><div className="table-subtitle">此区域为历史审计数据，仅作对照，不代表当前2.6.0策略表现。</div></div>
        <div className="badge">研究时间 {researchStatus.as_of.replace("T"," ").replace("Z","")}</div>
      </div>
      <div className="research-grid">
        <div><div className="metric-label">历史策略稳健研究点</div><div className="metric-value">{preferred?fmt(preferred.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{preferred?preferred.samples.toLocaleString("zh-CN")+" 个样本 · 威尔逊下界 "+fmt(preferred.wilson_lower_pct,2)+"%":"—"}</div></div>
        <div><div className="metric-label">历史策略最高观察点</div><div className="metric-value">{highest?fmt(highest.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{highest?highest.samples.toLocaleString("zh-CN")+" 个样本 · 威尔逊下界 "+fmt(highest.wilson_lower_pct,2)+"%":"—"}</div></div>
        <div><div className="metric-label">历史策略每日首选</div><div className="metric-value">{dailyTop1?fmt(dailyTop1.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{dailyTop1?dailyTop1.samples+" 个交易日 · 威尔逊下界 "+fmt(dailyTop1.wilson_lower_pct,2)+"%":"—"}</div></div>
      </div>
      <div className="research-warning">历史路径感知模型记录约 {researchStatus.path_aware_caveat?.final_win_rate_pct??"—"}%：该数字属于旧研究版本，只作为审计对照，不能代表当前策略。</div>
    </section>

    <section className="card snapshot-warning">
      <strong>{candidatesAreCurrent?"当前研究候选通过版本一致性检查":"当前候选快照已过期或待重新审计"}</strong>
      <span>系统版本 {systemVersion.system_version} · 策略版本 {systemVersion.strategy_version} · 绑定提交 {systemVersion.strategy_commit.slice(0,10)}；当前快照版本 {snapshot.strategy_version} · 提交 {snapshot.strategy_commit.slice(0,10)}。{candidatesAreCurrent?"":"旧候选已隐藏，待云端按当前策略重新生成。"}</span>
    </section>

    <div className="main">
      <section className="card table-card">
        <div className="table-head"><div><div className="table-title">{liveOrderingEnabled?"生产候选":"研究候选快照"}</div><div className="table-subtitle">{liveOrderingEnabled?"模型评分（模型概率×100） + 市场门控":"候选快照；策略质量门槛未通过，不作为实盘信号"}</div></div><div className="badge">数据时点 {snapshot.as_of.replace("T"," ")}</div></div>
        {!candidatesAreCurrent?<div className="empty">当前网页不展示旧策略候选。训练基准 {researchTrial.strategy_version} 尚未生成与当前版本完全一致的候选快照；系统宁可暂不展示，也不混用旧信号。</div>:rows.length===0?<div className="empty">当前市场门控未产生候选。系统允许空仓，而不是为了凑够候选数量强行入选。</div>:
        <div className="table-wrap"><table><thead><tr><th>#</th><th>股票</th><th>价格</th><th>隔夜</th><th>今日</th><th>3日</th><th>5日</th><th>10日</th><th>量比</th><th>成交额</th><th>10日波动</th><th>收盘强度</th><th>模型评分</th></tr></thead>
        <tbody>{rows.map(r=><tr key={r.symbol}><td className="rank">{r.rank}</td><td><span className="symbol">{r.symbol}</span><span className="name">{r.name}</span></td><td>{fmt(r.price)}</td><td className={r.overnight_1d_pct==null?"":dirClass(r.overnight_1d_pct)}>{r.overnight_1d_pct==null?"—":pct(r.overnight_1d_pct)}</td><td className={dirClass(r.change_pct)}>{pct(r.change_pct)}</td><td className={dirClass(r.return_3d_pct)}>{pct(r.return_3d_pct)}</td><td className={dirClass(r.return_5d_pct)}>{pct(r.return_5d_pct)}</td><td className={dirClass(r.return_10d_pct)}>{pct(r.return_10d_pct)}</td><td>{fmt(r.volume_ratio_5d,2)}倍</td><td>{amount(r.amount)}</td><td>{fmt(r.volatility_10d_pct,2)}%</td><td>{fmt(r.close_strength*100,1)}%</td><td className="score">{fmt(r.score)}</td></tr>)}</tbody></table></div>}
      </section>

      <aside className="side">
        <div className="card"><h2>当前策略评分因子</h2><p>当前版本直接使用 Aquant-Private/main 的评分与准入实现。只有当候选快照与2.6.0版本和提交号一致时才展示当前因子列表。</p>{currentFeatures.length===0?<div className="empty">等待当前策略候选快照生成后展示因子清单。</div>:<div className="feature-list">{currentFeatures.map((feature,index)=><div className="factor" key={feature}><div className="factor-row"><span className="factor-name">{index+1}. {featureLabels[feature]??"其他策略因子"}</span><span className="factor-weight">评分输入</span></div></div>)}</div>}<div className="next-plan-summary">评分来源 Aquant-Private/strategy/model.py · 主准入阈值 {fmt(snapshot.model_definition?.primary_probability_threshold??0.89,2)} · 最多候选 {maxCandidates}</div></div>
        <div className="card source-box"><h2>运行信息</h2><dl className="kv"><dt>信号</dt><dd>{snapshot.signal_horizon.replace("T收盘信号","T日收盘信号").replace("T+1开盘进入","T+1开盘执行")}</dd><dt>系统版本</dt><dd>{systemVersion.system_version}</dd><dt>策略版本</dt><dd>{systemVersion.strategy_version}</dd><dt>策略提交</dt><dd>{systemVersion.strategy_commit.slice(0,10)}</dd><dt>候选快照</dt><dd>{snapshot.strategy_version}</dd><dt>发布状态</dt><dd>研究中，未发布</dd><dt>质量门禁</dt><dd>{selectedPoint?fmt(selectedPoint.final_win_rate_pct,2)+"%（历史标签）":"等待当前版本验证"}</dd></dl></div>
      </aside>
    </div>

    <footer className="footer"><span>研究 / 模拟交易系统 · 不连接券商执行 · 策略来源 {systemVersion.strategy_commit.slice(0,10)}</span><span>研究/模拟模式，禁止实盘</span></footer>
  </main>;
}
