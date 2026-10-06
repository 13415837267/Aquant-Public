import candidatesData from "@/data/candidates.json";
import auditData from "@/data/system_audit.json";
import productionStatusData from "@/data/production_status.json";
import researchStatusData from "@/data/research_status.json";
import nextTradingPlanData from "@/data/next_trading_day_plan.json";
import researchTrialData from "@/data/research_trial_status.json";

type Candidate = {
  rank:number; symbol:string; name:string; price:number; change_pct:number; overnight_1d_pct?:number; intraday_return_pct?:number;
  return_3d_pct:number; return_5d_pct:number; return_10d_pct:number;
  volume_ratio_5d:number; turnover_pct:number; amount:number;
  volatility_10d_pct:number; close_strength:number; score:number; precision_probability:number; admission_tier:string; flags?:string[];
};
type Snapshot = {
  as_of:string; status:string; strategy_version:string; strategy_commit:string; signal_horizon:string;
  candidate_admission_policy:string; candidates:Candidate[];
  factor_weights?:Record<string,number>;
  market?:{breadth_pct:number;median_return_pct:number;regime:string};
  diagnostics?:{scorable_rows?:number;candidate_count?:number;risk_off_no_trade?:boolean};
};
type NextTradingPlan = {
  next_trading_day:string; signal_date:string; data_cutoff:string; status:string; title:string; summary:string;
  strategy_version:string; strategy_commit:string; candidate_policy:string; candidate_count:number;
  market:{breadth_pct:number;median_return_pct:number;regime:string};
  candidates:Array<{rank:number;symbol:string;name:string;price:number;score:number;precision_probability:number;admission_tier:string;signal_date:string;execution_date:string;earliest_exit_date:string;net_win_threshold_pct:number;stop_loss_pct:number;flags:string[]}>;
  source_snapshot?:{path:string;as_of:string;strategy_version:string;strategy_commit:string};
  steps:Array<{time:string;action:string}>; hard_rules:string[]; production_release:boolean;
};

type ResearchTrial = { status:string; as_of:string; strategy_version:string; model_code_commit:string; data_cutoff:string; training_window:{start:string;end:string}; selected_operating_point:{probability_threshold:number;final_samples:number;final_sample_share_pct:number;final_win_rate_pct:number;final_wilson_lower_pct:number;final_mean_best_return_pct:number;final_mean_5d_close_return_pct:number}; validation:{samples:number;win_rate_pct:number;wilson_lower_pct:number}; audits:{no_future_features:boolean;entry_is_T_plus_1_open:boolean;exit_starts_T_plus_2:boolean;strict_profit_label:boolean}; production_release:boolean; formal_gate_pct:number; note:string; };

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
const provenanceMatches=audit.strategy_version===snapshot.strategy_version && audit.strategy_commit===snapshot.strategy_commit;
const productionStatus=productionStatusData as {status:string;production_version:string|null;release_gate:boolean;system_audit:boolean};
const researchStatus=researchStatusData as ResearchStatus;
const nextTradingPlan=nextTradingPlanData as NextTradingPlan;
const researchTrial=researchTrialData as ResearchTrial;
const productionReady=productionStatus.status==="released" && productionStatus.release_gate===true && productionStatus.system_audit===true;
const fmt=(n:number,d=2)=>n.toLocaleString("zh-CN",{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=(n:number)=>(n>=0?"+":"")+fmt(n)+"%";
const dirClass=(n:number)=>n>0?"rise":n<0?"fall":"flat";
const amount=(n:number)=>n>=1e8?fmt(n/1e8,1)+"亿":fmt(n/1e4,0)+"万";
const weight=(n:number)=>fmt(n*100,0)+"%";
const maxCandidates=Number((snapshot.candidate_admission_policy.match(/precision_top_(\\d+)/)||snapshot.candidate_admission_policy.match(/dynamic_top_score_(\\d+)/)||snapshot.candidate_admission_policy.match(/top_(\\d+)/)||[])[1]||0);

export default function Home(){
  const strategyMetadataReady=!!snapshot.strategy_version && !!snapshot.strategy_commit && !!researchTrial.strategy_version && !!researchTrial.model_code_commit;
  const candidatesAreCurrent=snapshot.status==="ready" && strategyMetadataReady && snapshot.strategy_version===researchTrial.strategy_version && snapshot.strategy_commit===researchTrial.model_code_commit && snapshot.future_function===false;
  const rows=candidatesAreCurrent?(snapshot.candidates??[]):[];
  const planMatchesSnapshot=candidatesAreCurrent && nextTradingPlan.strategy_version===snapshot.strategy_version && nextTradingPlan.strategy_commit===snapshot.strategy_commit && nextTradingPlan.data_cutoff===snapshot.as_of.slice(0,10) && nextTradingPlan.candidate_count===rows.length && nextTradingPlan.candidates.length===rows.length && nextTradingPlan.candidates.every((candidate,index)=>{const row=rows[index]; return candidate.rank===row.rank && candidate.symbol===row.symbol && candidate.name===row.name && candidate.price===row.price && candidate.score===row.score && candidate.precision_probability===row.precision_probability && candidate.admission_tier===row.admission_tier;});
  const w=snapshot.factor_weights??{};
  const m=snapshot.market;
  const preferred=researchStatus.preferred_operating_point?.final;
  const highest=researchStatus.highest_observed_operating_point?.final;
  const dailyTop1=researchStatus.daily_top1?.final;
  return <main className="container">
    <header className="header">
      <div><div className="eyebrow">Aquant / 短线运行系统</div><h1>短线候选池</h1><div className="color-legend"><span className="legend-rise">红涨</span><span className="legend-fall">绿跌</span><span className="legend-flat">平盘中性</span></div>
      <p className="subtitle">T 日收盘生成信号，T+1 开盘执行，最长持有 5 个交易日；弱市场允许空仓。</p></div>
      <div className="pill">第一版系统 {productionReady?"正式启用":"未启用"} · {researchTrial.strategy_version}</div>
    </header>

    <section className="grid">
      <div className="card"><div className="metric-label">当前候选数量</div><div className="metric-value">{candidatesAreCurrent?rows.length:"—"}</div><div className="metric-note">{candidatesAreCurrent?("模型候选上限 "+maxCandidates):"等待第一版候选快照"}</div></div>
      <div className="card"><div className="metric-label">可评分股票</div><div className="metric-value">{snapshot.diagnostics?.scorable_rows??"—"}</div><div className="metric-note">20日短线特征</div></div>
      <div className="card"><div className="metric-label">市场状态</div><div className="metric-value">{m?.regime==="neutral"?"中性":m?.regime==="risk_on"?"风险偏好":m?.regime==="risk_off"?"风险规避":m?.regime??"—"}</div><div className="metric-note">广度 {m?fmt(m.breadth_pct,1)+"%":"—"}</div></div>
      <div className="card"><div className="metric-label">交易窗口</div><div className="metric-value">1–5日</div><div className="metric-note">T+1 开盘执行</div></div>
    </section>

    <section className="card next-plan-card">
      <div className="table-head">
        <div><div className="table-title">下一交易日计划</div><div className="table-subtitle">{nextTradingPlan.title}</div></div>
        <div className="badge">交易日 {nextTradingPlan.next_trading_day}</div>
      </div>
      <div className="next-plan-summary">{nextTradingPlan.summary}</div>
      <div className="plan-grid">
        {!candidatesAreCurrent?<div className="empty">候选快照尚未通过当前策略与系统审计一致性检查，计划中的候选暂不单独展示，避免出现两套信号。</div>:!planMatchesSnapshot?<div className="empty">下一交易日计划与候选快照未逐项一致，系统拒绝展示分叉信号。</div>:rows.map((candidate)=><div className="plan-item" key={candidate.symbol}><div className="plan-time">计划候选 #{candidate.rank}</div><div className="plan-action"><strong>{candidate.symbol} {candidate.name}</strong> · 收盘价 {fmt(candidate.price)} · 综合分 {fmt(candidate.score)} · 精度概率 {fmt(candidate.precision_probability*100,2)}%。准入层级：{candidate.admission_tier}。信号日 {nextTradingPlan.signal_date} → 执行日 {nextTradingPlan.next_trading_day} → 最早退出 {nextTradingPlan.candidates[candidate.rank-1].earliest_exit_date}。单笔净利润目标 +{fmt(nextTradingPlan.candidates[candidate.rank-1].net_win_threshold_pct,1)}%，止损 {fmt(nextTradingPlan.candidates[candidate.rank-1].stop_loss_pct,1)}%。</div></div>)}
      </div>
      <div className="next-plan-summary">数据截止 {nextTradingPlan.data_cutoff} · 策略 {nextTradingPlan.strategy_version} · 市场 {nextTradingPlan.market.regime==="neutral"?"中性":nextTradingPlan.market.regime==="risk_on"?"风险偏好":"风险规避"} · 候选数 {nextTradingPlan.candidate_count} · 唯一候选来源 {nextTradingPlan.source_snapshot?.path??"data/candidates.json"}</div>
      <div className="plan-grid">
        {nextTradingPlan.steps.map((step)=><div className="plan-item" key={step.time}><div className="plan-time">{step.time}</div><div className="plan-action">{step.action}</div></div>)}
      </div>
      <div className="next-plan-warning">当前生产门禁：<strong>{nextTradingPlan.production_release?"允许生产":"禁止实盘"}</strong>。本计划候选必须与 data/candidates.json 候选快照逐项一致；只使用 {nextTradingPlan.signal_date} 收盘及此前已知数据，不使用执行日或之后的行情结果参与选股。</div>
    </section>

    <section className="card research-card"><div className="table-head"><div><div className="table-title">第一版训练基准表现</div><div className="table-subtitle">近期训练窗口：{researchTrial.training_window.start} 至 {researchTrial.training_window.end}；当前系统正式采用，策略质量门槛仍为80%。</div></div><div className="badge">正式基线</div></div><div className="research-grid"><div><div className="metric-label">最终留出胜率</div><div className="metric-value">{fmt(researchTrial.selected_operating_point.final_win_rate_pct,2)}%</div><div className="metric-note">{researchTrial.selected_operating_point.final_samples.toLocaleString("zh-CN")} 个样本 · 威尔逊下界 {fmt(researchTrial.selected_operating_point.final_wilson_lower_pct,2)}%</div></div><div><div className="metric-label">验证集胜率</div><div className="metric-value">{fmt(researchTrial.validation.win_rate_pct,2)}%</div><div className="metric-note">{researchTrial.validation.samples.toLocaleString("zh-CN")} 个样本 · 威尔逊下界 {fmt(researchTrial.validation.wilson_lower_pct,2)}%</div></div><div><div className="metric-label">样本覆盖率</div><div className="metric-value">{fmt(researchTrial.selected_operating_point.final_sample_share_pct,2)}%</div><div className="metric-note">阈值 {fmt(researchTrial.selected_operating_point.probability_threshold,2)}</div></div></div><div className="research-warning"><strong>系统已正式启用，但策略质量门槛尚未通过。</strong> 当前最终留出 {fmt(researchTrial.selected_operating_point.final_win_rate_pct,2)}%，仍低于正式 {fmt(researchTrial.formal_gate_pct,0)}% 门槛；网页不会因此解除生产门禁。T+1 开盘执行、T+2 起退出、严格标签和无未来函数审计均保持有效。</div></section>
    <section className="card research-card">
      <div className="table-head">
        <div><div className="table-title">正式策略历史审计结果</div><div className="table-subtitle">旧策略仅作为历史审计对照，不参与第一版系统候选生成。</div></div>
        <div className="badge">研究时间 {researchStatus.as_of.replace("T"," ").replace("Z","")}</div>
      </div>
      <div className="research-grid">
        <div><div className="metric-label">历史策略稳健研究点</div><div className="metric-value">{preferred?fmt(preferred.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{preferred?preferred.samples.toLocaleString("zh-CN")+" 个样本 · 威尔逊下界 "+fmt(preferred.wilson_lower_pct,2)+"%":"—"}</div></div>
        <div><div className="metric-label">历史策略最高观察点</div><div className="metric-value">{highest?fmt(highest.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{highest?highest.samples.toLocaleString("zh-CN")+" 个样本 · 威尔逊下界 "+fmt(highest.wilson_lower_pct,2)+"%":"—"}</div></div>
        <div><div className="metric-label">历史策略每日首选</div><div className="metric-value">{dailyTop1?fmt(dailyTop1.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{dailyTop1?dailyTop1.samples+" 个交易日 · 威尔逊下界 "+fmt(dailyTop1.wilson_lower_pct,2)+"%":"—"}</div></div>
      </div>
      <div className="research-warning">路径感知模型目前约 {researchStatus.path_aware_caveat?.final_win_rate_pct??"—"}%：该定义要求 +1% 目标先于 -3% 止损触发，因此 <strong>当前高精度结果仍属于研究候选，不得视为已通过正式生产门槛。</strong></div>
    </section>

    <section className="card snapshot-warning">
      <strong>{candidatesAreCurrent?"候选快照已通过策略与审计一致性检查":"候选快照尚未通过策略与审计一致性检查"}</strong>
      <span>当前训练基准 {researchTrial.strategy_version} · {researchTrial.model_code_commit.slice(0,10)}；网页候选快照版本 {snapshot.strategy_version} · {snapshot.strategy_commit.slice(0,10)}；正式策略审计仍独立保留。{candidatesAreCurrent?"":"待研究/审计流水线完成后自动替换。"}</span>
    </section>

    <div className="main">
      <section className="card table-card">
        <div className="table-head"><div><div className="table-title">{productionReady?"生产候选":"研究候选快照"}</div><div className="table-subtitle">{productionReady?"短线综合分 + 市场门控":"候选快照；生产门控未通过，不作为实盘信号"}</div></div><div className="badge">数据时点 {snapshot.as_of.replace("T"," ")}</div></div>
        {!candidatesAreCurrent?<div className="empty">当前网页不展示旧策略候选。训练基准 {researchTrial.strategy_version} 尚未生成与当前版本完全一致的候选快照；系统宁可暂不展示，也不混用旧信号。</div>:rows.length===0?<div className="empty">当前市场门控未产生候选。系统允许空仓，而不是为了凑够候选数量强行入选。</div>:
        <div className="table-wrap"><table><thead><tr><th>#</th><th>股票</th><th>价格</th><th>隔夜</th><th>今日</th><th>3日</th><th>5日</th><th>10日</th><th>量比</th><th>成交额</th><th>10日波动</th><th>收盘强度</th><th>综合分</th></tr></thead>
        <tbody>{rows.map(r=><tr key={r.symbol}><td className="rank">{r.rank}</td><td><span className="symbol">{r.symbol}</span><span className="name">{r.name}</span></td><td>{fmt(r.price)}</td><td className={r.overnight_1d_pct==null?"":dirClass(r.overnight_1d_pct)}>{r.overnight_1d_pct==null?"—":pct(r.overnight_1d_pct)}</td><td className={dirClass(r.change_pct)}>{pct(r.change_pct)}</td><td className={dirClass(r.return_3d_pct)}>{pct(r.return_3d_pct)}</td><td className={dirClass(r.return_5d_pct)}>{pct(r.return_5d_pct)}</td><td className={dirClass(r.return_10d_pct)}>{pct(r.return_10d_pct)}</td><td>{fmt(r.volume_ratio_5d,2)}倍</td><td>{amount(r.amount)}</td><td>{fmt(r.volatility_10d_pct,2)}%</td><td>{fmt(r.close_strength*100,1)}%</td><td className="score">{fmt(r.score)}</td></tr>)}</tbody></table></div>}
      </section>

      <aside className="side">
        <div className="card"><h2>训练模型输入</h2><p>当前候选池以近期训练窗口模型为基准，使用短周期价格、隔夜结构、量能、波动、收盘强度、换手与市场状态特征；旧 2.5.0 候选规则不再作为当前候选基准。</p>
        {[["短线动量","momentum_short"],["隔夜结构","overnight_structure"],["量能活跃","volume_activity"],["价格强度","price_strength"],["流动性","liquidity"],["安全","safety"]].map(([label,key])=><div className="factor" key={key}><div className="factor-row"><span className="factor-name">{label}</span><span className="factor-weight">{w[key]==null?"—":weight(w[key])}</span></div></div>)}</div>
        <div className="card source-box"><h2>运行信息</h2><dl className="kv"><dt>信号</dt><dd>{snapshot.signal_horizon.replace("T收盘信号","T日收盘信号").replace("T+1开盘进入","T+1开盘执行")}</dd><dt>训练基准</dt><dd>版本 {researchTrial.strategy_version}</dd><dt>模型代码</dt><dd>{researchTrial.model_code_commit.slice(0,10)}</dd><dt>候选快照</dt><dd>版本 {snapshot.strategy_version}</dd><dt>生产状态</dt><dd>{productionReady?"第一版系统已启用":"未启用"}</dd><dt>策略质量</dt><dd>{fmt(researchTrial.selected_operating_point.final_win_rate_pct,2)}% / {fmt(researchTrial.formal_gate_pct,0)}%</dd></dl></div>
      </aside>
    </div>

    <footer className="footer"><span>研究 / 模拟交易系统 · 不连接券商执行。</span><span>{productionReady?(snapshot.diagnostics?.risk_off_no_trade?"弱市：允许无候选":"第一版系统候选可用"):"系统未启用"}</span></footer>
  </main>;
}
