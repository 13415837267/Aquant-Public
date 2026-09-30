import candidatesData from "@/data/candidates.json";

type Candidate = {
  rank: number; symbol: string; name: string; price: number; change_pct: number;
  momentum_60d: number; turnover_pct: number; pb: number | null; pe: number | null;
  amount: number; volatility_proxy: number; score: number; flags: string[];
};
type Snapshot = {
  as_of: string | null; timezone: string; source: string; status: string;
  strategy_version: string; universe: string; candidates: Candidate[];
};

const snapshot = candidatesData as Snapshot;
const fmt = (n:number,digits=2) => n.toLocaleString("zh-CN",{maximumFractionDigits:digits,minimumFractionDigits:digits});
const pct = (n:number) => (n >= 0 ? "+" : "") + fmt(n) + "%";
const amount = (n:number) => n >= 1e8 ? fmt(n/1e8,1) + "亿" : fmt(n/1e4,0) + "万";

export default function Home() {
  const rows = snapshot.candidates ?? [];
  const scores = rows.map(r=>r.score);
  const avgScore = scores.length ? scores.reduce((a,b)=>a+b,0)/scores.length : 0;
  const positive = rows.filter(r=>r.change_pct>0).length;
  const highMomentum = rows.filter(r=>r.momentum_60d>=10).length;

  return (
    <main className="container">
      <header className="header">
        <div>
          <div className="eyebrow">Aquant / Daily Research Runtime</div>
          <h1>候选池</h1>
          <p className="subtitle">基于公开市场日频数据的横截面多因子筛选。18:00（北京时间）更新当日收盘数据，并只写入新增/变化快照。</p>
        </div>
        <div className="pill">策略 v{snapshot.strategy_version} · {snapshot.status}</div>
      </header>

      <section className="grid">
        <div className="card"><div className="metric-label">候选数量</div><div className="metric-value">{rows.length || "—"}</div><div className="metric-note">目标池 30</div></div>
        <div className="card"><div className="metric-label">平均因子分</div><div className="metric-value">{rows.length ? fmt(avgScore) : "—"}</div><div className="metric-note">0–100 标准化</div></div>
        <div className="card"><div className="metric-label">当日上涨</div><div className="metric-value">{rows.length ? positive + "/" + rows.length : "—"}</div><div className="metric-note">候选池内部</div></div>
        <div className="card"><div className="metric-label">60日动量 ≥ 10%</div><div className="metric-value">{rows.length ? highMomentum + "/" + rows.length : "—"}</div><div className="metric-note">趋势子因子</div></div>
      </section>

      <div className="main">
        <section className="card table-card">
          <div className="table-head">
            <div><div className="table-title">今日候选池</div><div className="table-subtitle">按综合评分降序；数据更新后重新排名</div></div>
            {snapshot.as_of && <div className="badge">数据时点 {snapshot.as_of.replace("T"," ")}</div>}
          </div>
          {rows.length === 0 ? (
            <div className="empty">暂无快照。GitHub Actions 会在北京时间每日 18:00 运行数据增量任务；也可手动触发 workflow。</div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>#</th><th>股票</th><th>价格</th><th>今日</th><th>60日动量</th><th>换手</th><th>PB</th><th>成交额</th><th>波动代理</th><th>综合分</th><th>风险提示</th></tr></thead>
                <tbody>
                  {rows.map(r => (
                    <tr key={r.symbol}>
                      <td className="rank">{r.rank}</td>
                      <td><span className="symbol">{r.symbol}</span><span className="name">{r.name}</span></td>
                      <td>{fmt(r.price)}</td>
                      <td className={r.change_pct>0 ? "pos" : r.change_pct<0 ? "neg" : ""}>{pct(r.change_pct)}</td>
                      <td className={r.momentum_60d>0 ? "pos" : "neg"}>{pct(r.momentum_60d)}</td>
                      <td>{fmt(r.turnover_pct)}%</td>
                      <td>{r.pb == null ? "—" : fmt(r.pb)}</td>
                      <td>{amount(r.amount)}</td>
                      <td>{fmt(r.volatility_proxy)}%</td>
                      <td className="score">{fmt(r.score)}</td>
                      <td>{r.flags?.length ? r.flags.map(f => <span className="badge" key={f}>{f}</span>) : <span className="badge">正常</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <aside className="side">
          <div className="card">
            <h2>策略结构</h2>
            <p>第一版优先做可解释、可复现的横截面筛选；研究母版保存在私库，公库仅保留可发布运行版。</p>
            <div className="factor"><div className="factor-row"><span className="factor-name">动量 / Trend</span><span className="factor-weight">30%</span></div></div>
            <div className="factor"><div className="factor-row"><span className="factor-name">流动性 / Liquidity</span><span className="factor-weight">20%</span></div></div>
            <div className="factor"><div className="factor-row"><span className="factor-name">价值 / Value</span><span className="factor-weight">20%</span></div></div>
            <div className="factor"><div className="factor-row"><span className="factor-name">波动 / Risk</span><span className="factor-weight">15%</span></div></div>
            <div className="factor"><div className="factor-row"><span className="factor-name">交易强度 / Activity</span><span className="factor-weight">15%</span></div></div>
          </div>
          <div className="card source-box">
            <h2>运行信息</h2>
            <dl className="kv">
              <dt>数据源</dt><dd>{snapshot.source}</dd>
              <dt>时区</dt><dd>{snapshot.timezone}</dd>
              <dt>股票池</dt><dd>{snapshot.universe}</dd>
              <dt>策略版本</dt><dd>{snapshot.strategy_version}</dd>
              <dt>更新窗口</dt><dd>每日 18:00 BJT</dd>
            </dl>
          </div>
        </aside>
      </div>

      <footer className="footer">
        <span>Research system · not an execution or investment recommendation service.</span>
        <span>{snapshot.source === "pending" ? "等待首次收盘数据" : "数据源已同步"}</span>
      </footer>
    </main>
  );
}
