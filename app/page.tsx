import candidatesData from "@/data/candidates.json";
import auditData from "@/data/system_audit.json";

type Candidate = {
  rank:number; symbol:string; name:string; price:number; change_pct:number; overnight_1d_pct?:number; intraday_return_pct?:number;
  return_3d_pct:number; return_5d_pct:number; return_10d_pct:number;
  volume_ratio_5d:number; turnover_pct:number; amount:number;
  volatility_10d_pct:number; close_strength:number; score:number; flags:string[];
};
type Snapshot = {
  as_of:string; status:string; strategy_version:string; strategy_commit:string; signal_horizon:string;
  candidate_admission_policy:string; candidates:Candidate[];
  factor_weights?:Record<string,number>;
  market?:{breadth_pct:number;median_return_pct:number;regime:string};
  diagnostics?:{scorable_rows?:number;candidate_count?:number;risk_off_no_trade?:boolean};
};

const snapshot=candidatesData as Snapshot;
const audit=auditData as {production_gate?:string; strategy_version?:string; strategy_commit?:string; audit?:{production_gate_passed?:boolean}};
const provenanceMatches=audit.strategy_version===snapshot.strategy_version && audit.strategy_commit===snapshot.strategy_commit;
const productionReady=audit.production_gate==="passed" && audit.audit?.production_gate_passed===true && provenanceMatches;
const fmt=(n:number,d=2)=>n.toLocaleString("zh-CN",{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=(n:number)=>(n>=0?"+":"")+fmt(n)+"%";
const dirClass=(n:number)=>n>0?"pos":n<0?"neg":"";
const amount=(n:number)=>n>=1e8?fmt(n/1e8,1)+"亿":fmt(n/1e4,0)+"万";
const weight=(n:number)=>fmt(n*100,0)+"%";

export default function Home(){
  const rows=snapshot.candidates??[];
  const w=snapshot.factor_weights??{};
  const m=snapshot.market;
  return <main className="container">
    <header className="header">
      <div><div className="eyebrow">Aquant / Short-Term Runtime</div><h1>短线候选池</h1><div className="color-legend"><span className="legend-up">红涨</span><span className="legend-down">绿跌</span><span className="legend-flat">平盘中性</span></div>
      <p className="subtitle">T 日收盘生成信号，T+1 开盘进入，最长持有 5 个交易日。弱市场允许无交易信号。</p></div>
      <div className="pill">策略 v{snapshot.strategy_version} · {productionReady?"生产可用":(provenanceMatches?"研究中 · 未通过生产门槛":"研究中 · 审计版本不一致")}</div>
    </header>
    <section className="grid">
      <div className="card"><div className="metric-label">候选数量</div><div className="metric-value">{rows.length}</div><div className="metric-note">动态上限 3</div></div>
      <div className="card"><div className="metric-label">可评分股票</div><div className="metric-value">{snapshot.diagnostics?.scorable_rows??"—"}</div><div className="metric-note">20日短线特征</div></div>
      <div className="card"><div className="metric-label">市场状态</div><div className="metric-value">{m?.regime??"—"}</div><div className="metric-note">广度 {m?fmt(m.breadth_pct,1)+"%":"—"}</div></div>
      <div className="card"><div className="metric-label">交易窗口</div><div className="metric-value">1–5日</div><div className="metric-note">T+1 开盘执行</div></div>
    </section>
    <div className="main">
      <section className="card table-card">
        <div className="table-head"><div><div className="table-title">{productionReady?"生产候选":"研究候选"}</div><div className="table-subtitle">{productionReady?"短线综合分 + 市场门控":"研究信号快照；生产门控未通过，不作为实盘信号"}</div></div><div className="badge">数据时点 {snapshot.as_of.replace("T"," ")}</div></div>
        {rows.length===0?<div className="empty">当前市场门控未产生候选。系统允许空仓，而不是为了凑够 Top-3 强行入选。</div>:
        <div className="table-wrap"><table><thead><tr><th>#</th><th>股票</th><th>价格</th><th>隔夜</th><th>今日</th><th>3日</th><th>5日</th><th>10日</th><th>量比</th><th>成交额</th><th>10日波动</th><th>收盘强度</th><th>综合分</th></tr></thead>
        <tbody>{rows.map(r=><tr key={r.symbol}><td className="rank">{r.rank}</td><td><span className="symbol">{r.symbol}</span><span className="name">{r.name}</span></td><td>{fmt(r.price)}</td><td className={r.overnight_1d_pct==null?"":dirClass(r.overnight_1d_pct)}>{r.overnight_1d_pct==null?"—":pct(r.overnight_1d_pct)}</td><td className={dirClass(r.change_pct)}>{pct(r.change_pct)}</td><td className={dirClass(r.return_3d_pct)}>{pct(r.return_3d_pct)}</td><td className={dirClass(r.return_5d_pct)}>{pct(r.return_5d_pct)}</td><td className={dirClass(r.return_10d_pct)}>{pct(r.return_10d_pct)}</td><td>{fmt(r.volume_ratio_5d,2)}x</td><td>{amount(r.amount)}</td><td>{fmt(r.volatility_10d_pct,2)}%</td><td>{fmt(r.close_strength*100,1)}%</td><td className="score">{fmt(r.score)}</td></tr>)}</tbody></table></div>}
      </section>
      <aside className="side">
        <div className="card"><h2>短线因子</h2><p>核心驱动改为短周期价格行为、量能、价格强度、流动性和安全，不再用 PB/PE 作为生产信号核心。</p>
        {[["短线动量","momentum_short"],["量能活跃","volume_activity"],["价格强度","price_strength"],["流动性","liquidity"],["安全","safety"]].map(([label,key])=><div className="factor" key={key}><div className="factor-row"><span className="factor-name">{label}</span><span className="factor-weight">{w[key]==null?"—":weight(w[key])}</span></div></div>)}</div>
        <div className="card source-box"><h2>运行信息</h2><dl className="kv"><dt>信号</dt><dd>{snapshot.signal_horizon}</dd><dt>止盈参考</dt><dd>+6%</dd><dt>止损参考</dt><dd>-3%</dd><dt>候选规则</dt><dd>动态 Top-3 + 市场门控</dd><dt>生产门控</dt><dd>{productionReady?"已通过":(provenanceMatches?"研究阶段，未通过":"审计与候选版本不一致")}</dd><dt>策略版本</dt><dd>{snapshot.strategy_version}</dd></dl></div>
      </aside>
    </div>
    <footer className="footer"><span>Research / paper-trading system · not broker execution.</span><span>{productionReady?(snapshot.diagnostics?.risk_off_no_trade?"弱市：允许无候选":"生产候选可用"):"研究候选：未进入生产信号层"}</span></footer>
  </main>;
}
