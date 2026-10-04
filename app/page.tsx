import candidatesData from "@/data/candidates.json";
import auditData from "@/data/system_audit.json";
import productionStatusData from "@/data/production_status.json";
import researchStatusData from "@/data/research_status.json";

type Candidate = {
  rank:number; symbol:string; name:string; price:number; change_pct:number; overnight_1d_pct?:number; intraday_return_pct?:number;
  return_3d_pct:number; return_5d_pct:number; return_10d_pct:number;
  volume_ratio_5d:number; turnover_pct:number; amount:number;
  volatility_10d_pct:number; close_strength:number; score:number;
};
type Snapshot = {
  as_of:string; status:string; strategy_version:string; strategy_commit:string; signal_horizon:string;
  candidate_admission_policy:string; candidates:Candidate[];
  factor_weights?:Record<string,number>;
  market?:{breadth_pct:number;median_return_pct:number;regime:string};
  diagnostics?:{scorable_rows?:number;candidate_count?:number;risk_off_no_trade?:boolean};
};
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
const productionReady=productionStatus.status==="released" && productionStatus.release_gate===true && productionStatus.system_audit===true;
const fmt=(n:number,d=2)=>n.toLocaleString("zh-CN",{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=(n:number)=>(n>=0?"+":"")+fmt(n)+"%";
const dirClass=(n:number)=>n>0?"rise":n<0?"fall":"flat";
const amount=(n:number)=>n>=1e8?fmt(n/1e8,1)+"亿":fmt(n/1e4,0)+"万";
const weight=(n:number)=>fmt(n*100,0)+"%";
const maxCandidates=Number((snapshot.candidate_admission_policy.match(/dynamic_top_score_(\\d+)_with_market_gate/)||[])[1]||0);

export default function Home(){
  const candidatesAreCurrent=snapshot.strategy_version===researchStatus.strategy_version && snapshot.strategy_commit===researchStatus.strategy_commit;
  const rows=candidatesAreCurrent?(snapshot.candidates??[]):[];
  const w=snapshot.factor_weights??{};
  const m=snapshot.market;
  const preferred=researchStatus.preferred_operating_point?.final;
  const highest=researchStatus.highest_observed_operating_point?.final;
  const dailyTop1=researchStatus.daily_top1?.final;
  return <main className="container">
    <header className="header">
      <div><div className="eyebrow">Aquant / 短线运行系统</div><h1>短线候选池</h1><div className="color-legend"><span className="legend-rise">红涨</span><span className="legend-fall">绿跌</span><span className="legend-flat">平盘中性</span></div>
      <p className="subtitle">T 日收盘生成信号，T+1 开盘执行，最长持有 5 个交易日；弱市场允许空仓。</p></div>
      <div className="pill">研究策略 {researchStatus.strategy_version} · {productionReady?"生产可用":"研究中 · 尚未发布"}</div>
    </header>

    <section className="grid">
      <div className="card"><div className="metric-label">当前候选数量</div><div className="metric-value">{candidatesAreCurrent?rows.length:"—"}</div><div className="metric-note">{candidatesAreCurrent?("当前快照上限 "+maxCandidates):"等待最新策略候选快照"}</div></div>
      <div className="card"><div className="metric-label">可评分股票</div><div className="metric-value">{snapshot.diagnostics?.scorable_rows??"—"}</div><div className="metric-note">20日短线特征</div></div>
      <div className="card"><div className="metric-label">市场状态</div><div className="metric-value">{m?.regime==="neutral"?"中性":m?.regime==="risk_on"?"风险偏好":m?.regime==="risk_off"?"风险规避":m?.regime??"—"}</div><div className="metric-note">广度 {m?fmt(m.breadth_pct,1)+"%":"—"}</div></div>
      <div className="card"><div className="metric-label">交易窗口</div><div className="metric-value">1–5日</div><div className="metric-note">T+1 开盘执行</div></div>
    </section>

    <section className="card research-card">
      <div className="table-head">
        <div><div className="table-title">高精度研究结果</div><div className="table-subtitle">定义：单笔净利润达到 +1% 才计为胜；以下均为留出集研究结果，不等同于正式生产胜率。</div></div>
        <div className="badge">研究时间 {researchStatus.as_of.replace("T"," ").replace("Z","")}</div>
      </div>
      <div className="research-grid">
        <div><div className="metric-label">稳健研究点 ≥0.89</div><div className="metric-value">{preferred?fmt(preferred.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{preferred?preferred.samples.toLocaleString("zh-CN")+" 个样本 · 威尔逊下界 "+fmt(preferred.wilson_lower_pct,2)+"%":"—"}</div></div>
        <div><div className="metric-label">最高观察点 ≥0.90</div><div className="metric-value">{highest?fmt(highest.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{highest?highest.samples.toLocaleString("zh-CN")+" 个样本 · 威尔逊下界 "+fmt(highest.wilson_lower_pct,2)+"%":"—"}</div></div>
        <div><div className="metric-label">每日首选</div><div className="metric-value">{dailyTop1?fmt(dailyTop1.win_rate_pct,2)+"%":"—"}</div><div className="metric-note">{dailyTop1?dailyTop1.samples+" 个交易日 · 威尔逊下界 "+fmt(dailyTop1.wilson_lower_pct,2)+"%":"—"}</div></div>
      </div>
      <div className="research-warning">路径感知模型目前约 {researchStatus.path_aware_caveat?.final_win_rate_pct??"—"}%：该定义要求 +1% 目标先于 -3% 止损触发，因此 <strong>当前高精度结果仍属于研究候选，不得视为已通过正式生产门槛。</strong></div>
    </section>

    <section className="card snapshot-warning">
      <strong>{candidatesAreCurrent?"候选快照已与当前研究策略一致":"候选快照尚未同步到当前研究策略"}</strong>
      <span>当前研究策略 {researchStatus.strategy_version} · {researchStatus.strategy_commit.slice(0,10)}；网页候选快照 版本 {snapshot.strategy_version} · {snapshot.strategy_commit.slice(0,10)}。{candidatesAreCurrent?"":"待研究流水线完成后自动替换。"}</span>
    </section>

    <div className="main">
      <section className="card table-card">
        <div className="table-head"><div><div className="table-title">{productionReady?"生产候选":"研究候选快照"}</div><div className="table-subtitle">{productionReady?"短线综合分 + 市场门控":"候选快照；生产门控未通过，不作为实盘信号"}</div></div><div className="badge">数据时点 {snapshot.as_of.replace("T"," ")}</div></div>
        {!candidatesAreCurrent?<div className="empty">当前网页不展示旧策略候选。最新研究策略 {researchStatus.strategy_version} 尚未生成与当前版本完全一致的候选快照；系统宁可暂不展示，也不混用旧信号。</div>:rows.length===0?<div className="empty">当前市场门控未产生候选。系统允许空仓，而不是为了凑够候选数量强行入选。</div>:
        <div className="table-wrap"><table><thead><tr><th>#</th><th>股票</th><th>价格</th><th>隔夜</th><th>今日</th><th>3日</th><th>5日</th><th>10日</th><th>量比</th><th>成交额</th><th>10日波动</th><th>收盘强度</th><th>综合分</th></tr></thead>
        <tbody>{rows.map(r=><tr key={r.symbol}><td className="rank">{r.rank}</td><td><span className="symbol">{r.symbol}</span><span className="name">{r.name}</span></td><td>{fmt(r.price)}</td><td className={r.overnight_1d_pct==null?"":dirClass(r.overnight_1d_pct)}>{r.overnight_1d_pct==null?"—":pct(r.overnight_1d_pct)}</td><td className={dirClass(r.change_pct)}>{pct(r.change_pct)}</td><td className={dirClass(r.return_3d_pct)}>{pct(r.return_3d_pct)}</td><td className={dirClass(r.return_5d_pct)}>{pct(r.return_5d_pct)}</td><td className={dirClass(r.return_10d_pct)}>{pct(r.return_10d_pct)}</td><td>{fmt(r.volume_ratio_5d,2)}倍</td><td>{amount(r.amount)}</td><td>{fmt(r.volatility_10d_pct,2)}%</td><td>{fmt(r.close_strength*100,1)}%</td><td className="score">{fmt(r.score)}</td></tr>)}</tbody></table></div>}
      </section>

      <aside className="side">
        <div className="card"><h2>短线因子</h2><p>核心驱动为短周期价格行为、量能、价格强度、流动性和安全，不以 PB/PE 作为生产信号核心。</p>
        {[["短线动量","momentum_short"],["隔夜结构","overnight_structure"],["量能活跃","volume_activity"],["价格强度","price_strength"],["流动性","liquidity"],["安全","safety"]].map(([label,key])=><div className="factor" key={key}><div className="factor-row"><span className="factor-name">{label}</span><span className="factor-weight">{w[key]==null?"—":weight(w[key])}</span></div></div>)}</div>
        <div className="card source-box"><h2>运行信息</h2><dl className="kv"><dt>信号</dt><dd>{snapshot.signal_horizon.replace("T收盘信号","T日收盘信号").replace("T+1开盘进入","T+1开盘执行")}</dd><dt>研究策略</dt><dd>版本 {researchStatus.strategy_version}</dd><dt>候选快照</dt><dd>版本 {snapshot.strategy_version}</dd><dt>生产状态</dt><dd>{productionReady?"正式生产":"尚未发布"}</dd><dt>研究状态</dt><dd>仅研究</dd></dl></div>
      </aside>
    </div>

    <footer className="footer"><span>研究 / 模拟交易系统 · 不连接券商执行。</span><span>{productionReady?(snapshot.diagnostics?.risk_off_no_trade?"弱市：允许无候选":"生产候选可用"):"当前仍处于研究阶段"}</span></footer>
  </main>;
}
