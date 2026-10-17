import report from "@/data/research/latest_report.json";

type FactorRow = {因子:string; "盈利减亏损标准化均值差":number|null; 盈利组:{均值:number|null;样本数:number}; 亏损组:{均值:number|null;样本数:number}; 平盘组:{均值:number|null;样本数:number}};
const r:any=report;
const data:any=r["数据范围"]||{};
const formula:any=r["首版公式候选"]||{};
const splits:any=r["时间分段"]||{};
function num(x:any){return typeof x==="number"?x.toLocaleString("zh-CN"):"—"}
function percent(x:any){return typeof x==="number"?x.toFixed(2)+"%":"—"}
function effect(x:any){return typeof x==="number"?x.toFixed(3):"—"}
export default function Home(){
 const comparison:any=(r["分组对比"]||{})["训练期"]||{};
 const factors:FactorRow[]=(comparison["全因子比较"]||[]);
 const rules=formula["selected_leaf_rules"]||[];
 return <main className="page">
  <header className="hero">
   <p className="eyebrow">个人量化交易系统 · 首版研究</p>
   <h1>全市场因子与两日开盘收益研究</h1>
   <p className="lead">逐股票、逐交易日比较 T 日已知因子与 T+1 开盘买入、T+2 开盘卖出的净收益结果。</p>
   <div className="pill">研究状态：{r["状态"]||"等待云端研究"}</div>
  </header>
  <section className="cards">
   <article><span>历史信号日</span><strong>{num(data["信号交易日数"])}</strong><small>{data["历史数据库开始日期"]||"—"} 至 {data["历史数据库最后信号日期"]||"—"}</small></article>
   <article><span>可用因子</span><strong>{num(data["可用因子总数"])}</strong><small>行情、衍生技术、市场、点时财务</small></article>
   <article><span>完整样本行数</span><strong>{num(data["完整样本行数"])}</strong><small>包括可执行与不可执行记录</small></article>
   <article><span>可执行三分类</span><strong>{num(data["可执行三分类样本数"])}</strong><small>盈利 / 亏损 / 平盘</small></article>
  </section>
  <section className="panel">
   <h2>固定标签口径</h2>
   <div className="formula">(T+2 开盘价 ÷ T+1 开盘价 − 1) × 100% − 往返交易成本</div>
   <p>信号只使用 T 日收盘时可见信息。平盘容忍区间为 ±{percent(r["研究目标"]?.["平盘容忍区间百分比"])}。停牌、开盘涨跌停和价格缺失单独标记，不伪装成正常可执行交易。</p>
  </section>
  <section className="panel">
   <h2>时间隔离</h2>
   <div className="splitGrid">{Object.entries(splits).map(([name,v]:any)=><article key={name}><strong>{name}</strong><p>{v["开始日期"]||"—"} 至 {v["分段信号结束日期"]||"—"}</p><small>有效标签截止：{v["参与差异分析最晚信号日期"]||"—"}</small></article>)}</div>
  </section>
  <section className="panel">
   <h2>训练期因子差异排名</h2>
   <p className="muted">按盈利组与亏损组标准化均值差的绝对值排序。差异表示历史关联，不代表因果；请同时查看样本量与缺失率。</p>
   {factors.length?<div className="tableWrap"><table><thead><tr><th>因子</th><th>标准化差</th><th>盈利组均值</th><th>亏损组均值</th><th>平盘组均值</th><th>盈利组样本</th><th>亏损组样本</th></tr></thead><tbody>{factors.slice(0,20).map(row=><tr key={row.因子}><td>{row.因子}</td><td>{effect(row["盈利减亏损标准化均值差"])}</td><td>{num(row.盈利组?.均值)}</td><td>{num(row.亏损组?.均值)}</td><td>{num(row.平盘组?.均值)}</td><td>{num(row.盈利组?.样本数)}</td><td>{num(row.亏损组?.样本数)}</td></tr>)}</tbody></table></div>:<p className="empty">云端研究完成后，此处显示全部因子的分组比较结果。</p>}
   <details><summary>查看完整因子目录（{num(data["可用因子总数"])} 项）</summary><div className="factorList">{(data["因子清单"]||[]).map((name:string)=><span key={name}>{name}</span>)}</div></details>
  </section>
  <section className="panel">
   <h2>首版公式候选</h2>
   <div className="statusLine">公式状态：{formula.status||"unavailable"} · 验证通过：{formula.validation_passed?"是":"否"} · 最终留出通过：{formula.final_holdout_passed?"是":"否"}</div>
   {rules.length?<div className="rules">{rules.map((rule:any)=><article key={rule.leaf_id}><strong>规则 {rule.leaf_id}</strong><p>{(rule.conditions||[]).map((c:any)=>c.factor+" "+c.operator+" "+c.threshold).join(" 且 ")}</p></article>)}</div>:<p className="empty">尚无通过验证的可解释公式；不会凭空生成选股条件。</p>}
   <p className="warning">正式发布：关闭。自动下单：关闭。即使历史验证通过，也不代表已获准实盘。</p>
  </section>
  <footer>所有运行、测试与研究仅在 GitHub Actions 云端执行。研究状态以最近一次成功产物为准。</footer>
 </main>
}