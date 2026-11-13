/* EM Dispatch Lab: page rendering and interaction. Language follows <html lang>. */
(() => {
const LANG = (document.documentElement.lang || 'zh').toLowerCase().startsWith('zh') ? 'zh' : 'en';
const STR = {
  zh: {
    ACT: ['成本最低', '技能優先'], ACTS: ['成', '技'],
    SIT_NOTE: ['成本最低人選技能差很多', '成本最低人選技能差一點', '成本最低人選技能足夠'],
    play: '自動播放', pause: '暫停', reveal: '揭曉答案', hide: '隱藏答案',
    nodeOf: n => '（第 ' + n + ' 格）',
    envH: ['情境', '說明', '出現機率', '成：成本', '成：修好率', '技：成本', '技：修好率'],
    cellTitle: (t, s, a) => '第 ' + t + ' 步：情境 ' + s + '，' + a, skillCount: '技 ',
    qH: ['情境', '動作', 'φ(s,a)', 'E[R] = θᵀφ', 'Var[R]', '−½(β−1)Var', 'γE[V(s′)]', 'Q', 'ΔQ（技 − 成）'],
    sit: '情境 ', piX: '候選 β（對數刻度）；縱軸為 π(技 | s)',
    llH: ['步', '情境', '選擇', 'π(選的)', 'log π', '累計'], llX: i => 'τ' + i + ' 的 log P(τ | β)，對每個候選 β',
    elboX: 'EM 輪數', elboEmpty: '按「下一輪」後，這裡會畫出 ELBO 隨輪數的變化。',
    prior: '先驗 pₙ', priorTitle: (n, b, v) => '第 ' + n + ' 格 β = ' + b + '，pₙ = ' + v,
    traj: '軌跡', trueBeta: '真實 β', wTitle: (i, n, b, w, p) => 'τ' + i + '，第 ' + n + ' 格 β = ' + b + '，w = ' + w + '（先驗 ' + p + '）',
    bhatX: 'β（對數刻度；1 為中性）', bhatTrue: '<span class="reveal">×</span> 為真實 β。',
    gdH: ['候選 β', 'w', 'g₁（假如是這種個性）', 'g₂', 'w·g₁', 'w·g₂'], gSum: '合計 gᵢ（整張表）', gOnly: '只列把握度 &gt; 0.002 的格子',
    gtH: ['軌跡', 'g₁', 'g₂', '只用 β̂ 時 g₁', '只用 β̂ 時 g₂'],
    ops: { mean: '平均', pcgrad: 'PCGrad', cagrad: 'CAGrad' }, ax1: 'θ₁ 方向', ax2: 'θ₂ 方向',
    kConf: '衝突配對', kMean: '平均 d', kUsed: '本輪使用', kBhat: '（只用 β̂）', kUpd: '更新', kNext: '下一輪 θ',
    start: '起點', pathX: 'θ₁（派工成本權重）；縱軸 θ₂（重工權重）', histH: ['輪', 'θ₁', 'θ₂', 'ELBO'],
    pbLearned: b => '用學到的 θ̂，β = ' + b, pbCheap: '永遠成本最低', pbSafe: '永遠技能優先', pbTruth: '用真實 θ⋆，β = 1（參考）',
    pbH: ['規則', '情境 A', '情境 B', '情境 C', '平均派工成本', '重工率', 'J（用真實 θ⋆ 計分）'],
    pbNote: '步驟：以 θ̂ 當獎勵 r̂ = θ̂ᵀφ，代入均值-變異數 Q 算出指定 β 的 ΔQ，每個情境選較高者，得到一張規則表；再用真實獎勵 θ⋆ 為這張規則表計分（只有模擬中知道 θ⋆）。把 β 拉到 0.5 以下，看看規則在哪個情境改變、J 怎麼變。'
  },
  en: {
    ACT: ['Lowest cost', 'Skill first'], ACTS: ['C', 'S'],
    SIT_NOTE: ['Cheapest engineer far under-skilled', 'Cheapest engineer slightly under-skilled', 'Cheapest engineer skilled enough'],
    play: 'Auto play', pause: 'Pause', reveal: 'Reveal answers', hide: 'Hide answers',
    nodeOf: n => ' (node ' + n + ')',
    envH: ['Situation', 'Description', 'Probability', 'C: cost', 'C: fix rate', 'S: cost', 'S: fix rate'],
    cellTitle: (t, s, a) => 'Step ' + t + ': situation ' + s + ', ' + a, skillCount: 'S ',
    qH: ['Situation', 'Action', 'φ(s,a)', 'E[R] = θᵀφ', 'Var[R]', '−½(β−1)Var', 'γE[V(s′)]', 'Q', 'ΔQ (S − C)'],
    sit: 'Situation ', piX: 'Candidate β (log scale); vertical axis π(S | s)',
    llH: ['Step', 'Situation', 'Choice', 'π(chosen)', 'log π', 'Cumulative'], llX: i => 'log P(τ' + i + ' | β) for each candidate β',
    elboX: 'EM iteration', elboEmpty: 'Press “Next iteration” to plot the ELBO over iterations.',
    prior: 'Prior pₙ', priorTitle: (n, b, v) => 'node ' + n + ', β = ' + b + ', pₙ = ' + v,
    traj: 'Trajectory', trueBeta: 'True β', wTitle: (i, n, b, w, p) => 'τ' + i + ', node ' + n + ', β = ' + b + ', w = ' + w + ' (prior ' + p + ')',
    bhatX: 'β (log scale; 1 = neutral)', bhatTrue: '<span class="reveal">×</span> marks the true β.',
    gdH: ['Candidate β', 'w', 'g₁ (if this β)', 'g₂', 'w·g₁', 'w·g₂'], gSum: 'Total gᵢ (whole table)', gOnly: 'Only nodes with w &gt; 0.002 are listed',
    gtH: ['Trajectory', 'g₁', 'g₂', 'g₁ with β̂ only', 'g₂ with β̂ only'],
    ops: { mean: 'Mean', pcgrad: 'PCGrad', cagrad: 'CAGrad' }, ax1: 'θ₁ direction', ax2: 'θ₂ direction',
    kConf: 'Conflicting pairs', kMean: 'Mean d', kUsed: 'Used this iteration', kBhat: ' (β̂ only)', kUpd: 'Update', kNext: 'Next θ',
    start: 'Start', pathX: 'θ₁ (cost weight); vertical axis θ₂ (rework weight)', histH: ['Iter.', 'θ₁', 'θ₂', 'ELBO'],
    pbLearned: b => 'Learned θ̂, β = ' + b, pbCheap: 'Always lowest cost', pbSafe: 'Always skill first', pbTruth: 'True θ⋆, β = 1 (reference)',
    pbH: ['Rule', 'Situation A', 'Situation B', 'Situation C', 'Mean dispatch cost', 'Rework rate', 'J (scored with true θ⋆)'],
    pbNote: 'Steps: take r̂ = θ̂ᵀφ as the reward, compute ΔQ of the mean-variance Q for the chosen β, and pick the better action in each situation to obtain a rule table; then score that table with the true reward θ⋆ (known only in simulation). Drag β below 0.5 to see in which situation the rule changes and how J moves.'
  }
};
const T = STR[LANG];

const $ = id => document.getElementById(id);
const fx = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
let data = EM.DATA.map(tr => tr.map(c => c.slice()));
let S = { sel: 0, reveal: false, hist: [], playing: null };
const P = () => ({ lambda: +$('lam').value, gamma: +$('gam').value, R0: 1, alpha: +$('alpha').value, op: $('opSel').value, bhatOnly: $('bhatOnly').value === '1' });
let G = EM.grid(64);
function th0() { return [+$('th1').value, +$('th2').value]; }
function reset() { G = EM.grid(+$('kSel').value); $('bSel').max = G.nodes.length - 1; if (+$('bSel').value > G.nodes.length - 1) $('bSel').value = Math.floor(G.nodes.length * 0.6);
  S.hist = [{ th: th0() }]; stop(); render(); }
function cur() { return S.hist[S.hist.length - 1]; }
function step() { const r = EM.iterate(cur().th, data, P(), G); cur().elbo = r.elbo; S.hist.push({ th: r.next }); render(); return r; }
function stop() { if (S.playing) { clearInterval(S.playing); S.playing = null; } $('btnPlay').setAttribute('aria-pressed', 'false'); $('btnPlay').textContent = T.play; }
function converged() { const h = S.hist; if (h.length < 3) return false; const a = h[h.length - 1].th, b = h[h.length - 2].th; return Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-4; }

// ---------- small SVG helpers ----------
function svgEl(w, h) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px" role="img">`; }
function lineChart(el, series, o) {
  const W = 520, H = 230, L = 48, R = 12, T = 12, B = 32; const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y).filter(Number.isFinite);
  const x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs); let y0 = o.ymin ?? Math.min(...ys), y1 = o.ymax ?? Math.max(...ys); if (y1 - y0 < 1e-9) { y0 -= 1; y1 += 1; }
  const X = v => L + (o.logx ? (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : (v - x0) / (x1 - x0)) * (W - L - R); const Y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  let s = svgEl(W, H) + `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="none" stroke="${css('--line')}"/>`;
  (o.yticks || []).forEach(v => { s += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css('--line')}" stroke-dasharray="2 3"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${o.yfmt ? o.yfmt(v) : v}</text>`; });
  (o.xticks || []).forEach(v => { s += `<text x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`; });
  if (o.xlabel) s += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">${o.xlabel}</text>`;
  if (o.vline != null) s += `<line x1="${X(o.vline)}" x2="${X(o.vline)}" y1="${T}" y2="${H - B}" stroke="${css('--accent')}" stroke-width="1.5"/>`;
  series.forEach(sr => { const pts = sr.x.map((x, i) => [X(x), Y(sr.y[i])]).filter(p => Number.isFinite(p[1]));
    s += `<polyline fill="none" stroke="${sr.color}" stroke-width="2" points="${pts.map(p => p.join(',')).join(' ')}"/>`;
    if (sr.dots) pts.forEach(p => { s += `<circle cx="${p[0]}" cy="${p[1]}" r="2.6" fill="${sr.color}"/>`; });
    if (sr.label) { const p = pts[pts.length - 1]; s += `<text x="${p[0] - 4}" y="${p[1] + (sr.ldy ?? -6)}" text-anchor="end" style="fill:${sr.color};font-weight:700">${sr.label}</text>`; } });
  el.innerHTML = s + '</svg>';
}

// ---------- render ----------
function render() {
  const p = P(), h = cur(), r = EM.iterate(h.th, data, p, G), K = G.nodes.length, N = data.length, sel = S.sel, bi = +$('bSel').value, bn = G.nodes[bi];
  h.elbo = r.elbo;
  ['th1', 'th2', 'alpha', 'lam', 'gam'].forEach(k => $(k + 'v').textContent = ' ' + (+$(k).value).toFixed(2));
  $('bSelv').textContent = ' ' + fx(bn, 3) + T.nodeOf(bi + 1); $('bTargetv').textContent = ' ' + (+$('bTarget').value).toFixed(1);
  $('iterNo').textContent = S.hist.length - 1; $('thNow').textContent = '(' + fx(h.th[0], 3) + ', ' + fx(h.th[1], 3) + ')'; $('elboNow').textContent = fx(r.elbo, 3);
  $('revealNote').hidden = !S.reveal; $('btnReveal').setAttribute('aria-pressed', String(S.reveal)); $('btnReveal').textContent = S.reveal ? T.hide : T.reveal;
  // environment table
  $('envTable').innerHTML = '<tr>' + T.envH.map((x, k) => `<th${k < 2 ? ' class="l"' : ''}>${x}</th>`).join('') + '</tr>' +
    EM.SIT.map((n, s) => `<tr><td class="l"><b>${n}</b></td><td class="l">${T.SIT_NOTE[s]}</td><td>${EM.PS[s]}</td><td>${EM.COST[s][0].toFixed(2)}</td><td>${EM.PFIX[s][0].toFixed(2)}</td><td>${EM.COST[s][1].toFixed(2)}</td><td>${EM.PFIX[s][1].toFixed(2)}</td></tr>`).join('');
  // data grid
  $('trajGrid').innerHTML = data.map((tr, i) => `<span class="rowlab ${i === sel ? 'sel' : ''}" data-row="${i}" role="button" tabindex="0">τ${i + 1}</span><div class="cells">` +
    tr.map(([s, a], t) => `<span class="cell a${a}" data-i="${i}" data-t="${t}" title="${T.cellTitle(t + 1, EM.SIT[s], T.ACT[a])}">${EM.SIT[s]}<small>${T.ACTS[a]}</small></span>`).join('') +
    `</div><span class="mono" style="font-size:13px">${S.reveal ? '<span class="reveal">β = ' + EM.BETA_TRUE[i] + '</span>' : T.skillCount + tr.filter(c => c[1]).length + '/' + tr.length}</span>`).join('');
  // mean-variance Q table
  const q = EM.qTable(h.th, bn, p); const dq = EM.dQ(h.th, bn, p);
  $('qTable').innerHTML = '<tr>' + T.qH.map((x, k) => `<th${k < 2 ? ' class="l"' : ''}>${x}</th>`).join('') + '</tr>' +
    [0, 1, 2].map(s => [0, 1].map(a => `<tr><td class="l">${a === 0 ? '<b>' + EM.SIT[s] + '</b>' : ''}</td><td class="l" style="color:var(${a ? '--safe' : '--risky'})">${T.ACT[a]}</td><td class="mono">(${EM.COST[s][a].toFixed(2)}, ${(1 - EM.PFIX[s][a]).toFixed(2)})</td><td>${fx(q[s][a].ER)}</td><td>${fx(q[s][a].VR)}</td><td>${fx(-q[s][a].pen)}</td><td>${fx(q[s][a].cont)}</td><td><b>${fx(q[s][a].Q)}</b></td><td>${a === 1 ? '<b>' + fx(dq[s].d) + '</b>' : ''}</td></tr>`).join('')).join('');
  // softmax choice chart
  const cols = [css('--risky'), css('--accent'), css('--muted')];
  lineChart($('piChart'), [0, 1, 2].map(s => ({ x: G.nodes, y: r.pi.map(row => row[s]), color: cols[s], dots: K <= 7, label: T.sit + EM.SIT[s], ldy: [-6, 15, -6][s] })),
    { logx: true, ymin: 0, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], xticks: [0.02, 0.1, 0.3, 1, 3, 10].filter(v => v >= G.nodes[0] && v <= G.nodes[K - 1]), xlabel: T.piX, vline: bn });
  // trajectory likelihood detail
  let acc = 0;
  $('llDetail').innerHTML = '<tr>' + T.llH.map((x, k) => `<th${k === 1 || k === 2 ? ' class="l"' : ''}>${x}</th>`).join('') + '</tr>' + data[sel].map(([s, a], t) => {
    const pt = r.pi[bi][s]; const pr = a ? pt : 1 - pt; acc += Math.log(Math.max(pr, 1e-300));
    return `<tr><td>${t + 1}</td><td class="l">${EM.SIT[s]}</td><td class="l" style="color:var(${a ? '--safe' : '--risky'})">${T.ACT[a]}</td><td>${fx(pr)}</td><td>${fx(Math.log(Math.max(pr, 1e-300)))}</td><td>${fx(acc)}</td></tr>`; }).join('');
  lineChart($('llSum'), [{ x: G.nodes, y: r.LL[sel], color: css('--accent'), dots: K <= 7, label: 'τ' + (sel + 1) }],
    { logx: true, yticks: niceTicks(Math.min(...r.LL[sel].filter(Number.isFinite)), Math.max(...r.LL[sel])), xticks: [0.02, 0.1, 0.3, 1, 3, 10].filter(v => v >= G.nodes[0] && v <= G.nodes[K - 1]), xlabel: T.llX(sel + 1), vline: bn });
  // ELBO
  $('elboBig').textContent = fx(r.elbo, 3);
  const eh = S.hist.map((x, i) => [i, x.elbo]).filter(x => Number.isFinite(x[1]));
  if (eh.length > 1) lineChart($('elboChart'), [{ x: eh.map(v => v[0]), y: eh.map(v => v[1]), color: css('--accent'), dots: eh.length < 40 }], { yticks: niceTicks(Math.min(...eh.map(v => v[1])), Math.max(...eh.map(v => v[1]))), xticks: niceTicks(0, eh.length - 1, 5).map(Math.round), xlabel: T.elboX });
  else $('elboChart').innerHTML = '<p class="note">' + T.elboEmpty + '</p>';
  // posterior heat map
  const showNum = K <= 7; const heat = css('--heat');
  const hdr = G.nodes.map((b, n) => `<th style="text-align:center">${K <= 7 ? b : (n % 8 === 0 ? fx(b, 2) : '')}</th>`).join('');
  const pm = G.nodes.reduce((a, b, n) => a + G.prior[n] * b, 0), psd = Math.sqrt(G.nodes.reduce((a, b, n) => a + G.prior[n] * (b - pm) ** 2, 0));
  const hc = v => Math.min(1, v * (K <= 7 ? 1.4 : 8)), cellSt = v => `background:rgba(${heat},${hc(v).toFixed(3)});color:${hc(v) > 0.55 ? css('--surface') : css('--ink')};padding:${K <= 7 ? '5px 6px' : '5px 0'};min-width:${K <= 7 ? 46 : 9}px`;
  const priorRow = `<tr class="prior"><td class="l"><b>${T.prior}</b></td>` + G.prior.map((v, n) => `<td class="h" title="${T.priorTitle(n + 1, fx(G.nodes[n], 3), fx(v, 4))}" style="${cellSt(v)}">${showNum ? fx(v, 3) : ''}</td>`).join('') + `<td class="mono">${fx(pm, 2)}</td><td class="mono">${fx(psd, 2)}</td>${S.reveal ? '<td></td>' : ''}</tr>`;
  $('wTable').innerHTML = `<tr><th class="l">${T.traj}</th>${hdr}<th>β̂</th><th>σ</th>${S.reveal ? '<th class="reveal">' + T.trueBeta + '</th>' : ''}</tr>` + priorRow + r.W.map((w, i) => `<tr class="click ${i === sel ? 'sel' : ''}" data-row="${i}"><td class="l mono">τ${i + 1}</td>` +
    w.map((v, n) => `<td class="h" title="${T.wTitle(i + 1, n + 1, fx(G.nodes[n], 3), fx(v, 4), fx(G.prior[n], 4))}" style="${cellSt(v)}">${showNum ? fx(v, 3) : ''}</td>`).join('') +
    `<td class="mono">${fx(r.bhat[i], 2)}</td><td class="mono">${fx(r.sd[i], 2)}</td>${S.reveal ? '<td class="mono reveal">' + EM.BETA_TRUE[i] + '</td>' : ''}</tr>`).join('');
  // posterior summary chart
  bhatChart(r);
  // per-trajectory gradient detail
  const Gsel = r.Gn[sel];
  $('gDetail').innerHTML = '<tr>' + T.gdH.map(x => `<th>${x}</th>`).join('') + '</tr>' +
    G.nodes.map((b, n) => ({ b, n, w: r.W[sel][n] })).filter(o => K <= 7 || o.w > 0.002).map(o => `<tr><td>${fx(o.b, 3)}</td><td>${fx(o.w, 3)}</td><td>${fx(Gsel[o.n][0], 2)}</td><td>${fx(Gsel[o.n][1], 2)}</td><td>${fx(o.w * Gsel[o.n][0], 3)}</td><td>${fx(o.w * Gsel[o.n][1], 3)}</td></tr>`).join('') +
    `<tr><td class="l" colspan="4"><b>${T.gSum}</b>${K > 7 ? '<br><small style="color:var(--muted)">' + T.gOnly + '</small>' : ''}</td><td><b>${fx(Gsel.reduce((a, v, n) => a + r.W[sel][n] * v[0], 0), 3)}</b></td><td><b>${fx(Gsel.reduce((a, v, n) => a + r.W[sel][n] * v[1], 0), 3)}</b></td></tr>`;
  const gHat = EM.iterate(h.th, data, { ...p, bhatOnly: true }, G).g;
  $('gTable').innerHTML = '<tr>' + T.gtH.map((x, k) => `<th${k === 0 ? ' class="l"' : ''}>${x}</th>`).join('') + '</tr>' + r.g.map((g, i) => `<tr class="click ${i === sel ? 'sel' : ''}" data-row="${i}"><td class="l mono">τ${i + 1}</td><td>${fx(g[0], 3)}</td><td>${fx(g[1], 3)}</td><td style="color:var(--muted)">${fx(gHat[i][0], 3)}</td><td style="color:var(--muted)">${fx(gHat[i][1], 3)}</td></tr>`).join('');
  // aggregation and update
  const dMean = EM.aggregate(r.g, 'mean'), dPC = EM.aggregate(r.g, 'pcgrad'), dCA = EM.aggregate(r.g, 'cagrad');
  vecChart(r.g, [[dMean, css('--muted'), T.ops.mean], [dPC, css('--risky'), 'PCGrad'], [dCA, css('--accent'), 'CAGrad']]);
  $('updateKV').innerHTML = `<dt>${T.kConf}</dt><dd>${r.conflicts} / ${N * (N - 1) / 2}</dd><dt>${T.kMean}</dt><dd>(${fx(dMean[0])}, ${fx(dMean[1])})</dd><dt>PCGrad d</dt><dd>(${fx(dPC[0])}, ${fx(dPC[1])})</dd><dt>CAGrad d</dt><dd>(${fx(dCA[0])}, ${fx(dCA[1])})</dd>` +
    `<dt>${T.kUsed}</dt><dd>${T.ops[p.op]}${p.bhatOnly ? T.kBhat : ''}</dd><dt>${T.kUpd}</dt><dd>(${fx(h.th[0])}, ${fx(h.th[1])}) + ${p.alpha.toFixed(2)} × (${fx(r.d[0])}, ${fx(r.d[1])})</dd><dt>${T.kNext}</dt><dd><b>(${fx(r.next[0])}, ${fx(r.next[1])})</b></dd>`;
  // path and history
  pathChart();
  const hs = S.hist.map((x, i) => ({ i, ...x })); const show = hs.length <= 12 ? hs : [...hs.slice(0, 6), null, ...hs.slice(-5)];
  $('histTable').innerHTML = '<tr>' + T.histH.map(x => `<th>${x}</th>`).join('') + '</tr>' + show.map(x => x ? `<tr><td>${x.i}</td><td>${fx(x.th[0])}</td><td>${fx(x.th[1])}</td><td>${fx(x.elbo)}</td></tr>` : '<tr><td colspan="4" style="text-align:center">⋮</td></tr>').join('');
  // Phase B
  const bt = +$('bTarget').value; const mine = EM.phaseB(h.th, bt, p), truth = EM.phaseB(EM.THETA_TRUE, 1, p), cheap = { rule: [0, 0, 0] }, safe = { rule: [1, 1, 1] };
  const evalRule = rule => [0, 1, 2].reduce((a, s) => a + EM.PS[s] * (EM.THETA_TRUE[0] * EM.COST[s][rule[s]] + EM.THETA_TRUE[1] * (1 - EM.PFIX[s][rule[s]])), 0);
  const row = (name, rule, bold) => { const per = evalRule(rule), rw = [0, 1, 2].reduce((a, s) => a + EM.PS[s] * (1 - EM.PFIX[s][rule[s]]), 0), c = [0, 1, 2].reduce((a, s) => a + EM.PS[s] * EM.COST[s][rule[s]], 0);
    return `<tr${bold ? ' class="sel"' : ''}><td class="l">${name}</td>${rule.map(a => `<td class="l" style="color:var(${a ? '--safe' : '--risky'})">${T.ACT[a]}</td>`).join('')}<td>${fx(c * 100, 1)}</td><td>${fx(rw * 100, 1)}%</td><td><b>${fx(per / (1 - p.gamma), 2)}</b></td></tr>`; };
  $('pbTable').innerHTML = '<tr>' + T.pbH.map((x, k) => `<th${k < 4 ? ' class="l"' : ''}>${x}</th>`).join('') + '</tr>' +
    row(T.pbLearned(bt.toFixed(1)), mine.rule, true) + row(T.pbCheap, cheap.rule) + row(T.pbSafe, safe.rule) + (S.reveal ? row('<span class="reveal">' + T.pbTruth + '</span>', truth.rule) : '');
  $('pbNote').textContent = T.pbNote;
  // keep bindings
  document.querySelectorAll('[data-row]').forEach(el => { el.onclick = () => { S.sel = +el.dataset.row; render(); }; el.onkeydown = e => { if (e.key === 'Enter') { S.sel = +el.dataset.row; render(); } }; });
  document.querySelectorAll('.cell').forEach(el => el.onclick = () => { const i = +el.dataset.i, t = +el.dataset.t; data[i][t][1] = 1 - data[i][t][1]; S.sel = i; reset(); });
}
function niceTicks(a, b, n = 4) { if (!Number.isFinite(a) || !Number.isFinite(b)) return []; if (b - a < 1e-9) return [a]; const raw = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))); const st = [1, 2, 2.5, 5, 10].map(k => k * mag).find(k => k >= raw); const out = []; for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(+v.toFixed(6)); return out; }
function bhatChart(r) {
  const W = 520, H = 290, L = 46, R = 16, T0 = 10, B = 34, N = r.bhat.length, lo = 0.1, hi = 10;
  const X = v => L + (Math.log(Math.min(Math.max(v, lo), hi)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (W - L - R), Y = i => T0 + (i + 0.5) * (H - T0 - B) / N;
  let s = svgEl(W, H); [0.1, 0.3, 1, 3, 10].forEach(v => { s += `<line x1="${X(v)}" x2="${X(v)}" y1="${T0}" y2="${H - B}" stroke="${css('--line')}" stroke-dasharray="${v === 1 ? '' : '2 3'}"/><text x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`; });
  s += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">${T.bhatX}</text>`;
  r.bhat.forEach((b, i) => { const c = i === S.sel ? css('--accent') : css('--ink');
    s += `<text x="${L - 8}" y="${Y(i) + 4}" text-anchor="end">τ${i + 1}</text><line x1="${X(b - r.sd[i])}" x2="${X(b + r.sd[i])}" y1="${Y(i)}" y2="${Y(i)}" stroke="${c}" stroke-width="2"/><circle cx="${X(b)}" cy="${Y(i)}" r="4.5" fill="${c}"/>`;
    if (S.reveal) s += `<path d="M${X(EM.BETA_TRUE[i]) - 5},${Y(i) - 5} l10,10 m0,-10 l-10,10" stroke="${css('--warn')}" stroke-width="2"/>`; });
  $('bhatChart').innerHTML = s + '</svg>' + (S.reveal ? '<p class="note">' + T.bhatTrue + '</p>' : '');
}
function arrow(x1, y1, x2, y2, c, w, op = 1) {
  const L = Math.hypot(x2 - x1, y2 - y1); if (L < 1) return ''; const ux = (x2 - x1) / L, uy = (y2 - y1) / L, hs = 5 + w * 1.6;
  const bx = x2 - ux * hs, by = y2 - uy * hs, px = -uy * hs * 0.55, py = ux * hs * 0.55;
  return `<g opacity="${op}"><line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}" stroke="${c}" stroke-width="${w}"/><polygon points="${x2},${y2} ${bx + px},${by + py} ${bx - px},${by - py}" fill="${c}"/></g>`;
}
function vecChart(g, dirs) {
  const W = 360, H = 300, cx = W / 2, cy = H / 2; const m = Math.max(1e-6, ...g.flatMap(v => [Math.abs(v[0]), Math.abs(v[1])]), ...dirs.flatMap(d => [Math.abs(d[0][0]), Math.abs(d[0][1])])); const k = (Math.min(W, H) / 2 - 22) / m;
  let s = svgEl(W, H);
  s += `<line x1="10" x2="${W - 10}" y1="${cy}" y2="${cy}" stroke="${css('--line')}"/><line x1="${cx}" x2="${cx}" y1="10" y2="${H - 10}" stroke="${css('--line')}"/><text x="${W - 12}" y="${cy - 6}" text-anchor="end">${T.ax1}</text><text x="${cx + 6}" y="20">${T.ax2}</text>`;
  g.forEach((v, i) => { s += arrow(cx, cy, cx + v[0] * k, cy - v[1] * k, i === S.sel ? css('--accent') : css('--muted'), i === S.sel ? 2 : 1.2, i === S.sel ? 1 : 0.55); });
  dirs.forEach(([d, c, lab], j) => { s += arrow(cx, cy, cx + d[0] * k, cy - d[1] * k, c, 3.2) + `<text x="${cx + d[0] * k + 6}" y="${cy - d[1] * k + 12 * (j - 1)}" style="fill:${c};font-weight:700">${lab}</text>`; });
  $('vecChart').innerHTML = s + '</svg>';
}
function pathChart() {
  const pts = S.hist.map(x => x.th); const all = pts.concat(S.reveal ? [EM.THETA_TRUE] : []);
  const xs = all.map(v => v[0]), ys = all.map(v => v[1]); let x0 = Math.min(...xs, -0.2) - 0.2, x1 = Math.max(...xs, 0) + 0.2, y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys, 0) + 0.3;
  const W = 420, H = 300, L = 44, R = 12, T0 = 12, B = 34; const X = v => L + (v - x0) / (x1 - x0) * (W - L - R), Y = v => T0 + (1 - (v - y0) / (y1 - y0)) * (H - T0 - B);
  let s = svgEl(W, H) + `<rect x="${L}" y="${T0}" width="${W - L - R}" height="${H - T0 - B}" fill="none" stroke="${css('--line')}"/>`;
  niceTicks(x0, x1, 4).forEach(v => { s += `<text x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`; }); niceTicks(y0, y1, 4).forEach(v => { s += `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`; });
  s += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">${T.pathX}</text>`;
  s += `<polyline fill="none" stroke="${css('--accent')}" stroke-width="2" points="${pts.map(v => X(v[0]) + ',' + Y(v[1])).join(' ')}"/>`;
  s += `<circle cx="${X(pts[0][0])}" cy="${Y(pts[0][1])}" r="4" fill="${css('--muted')}"/><text x="${X(pts[0][0]) + 6}" y="${Y(pts[0][1]) - 6}">${T.start}</text>`;
  const e = pts[pts.length - 1]; s += `<circle cx="${X(e[0])}" cy="${Y(e[1])}" r="5" fill="${css('--accent')}"/>`;
  if (S.reveal) s += `<path d="M${X(EM.THETA_TRUE[0]) - 6},${Y(EM.THETA_TRUE[1]) - 6} l12,12 m0,-12 l-12,12" stroke="${css('--warn')}" stroke-width="2.5"/><text x="${X(EM.THETA_TRUE[0]) + 8}" y="${Y(EM.THETA_TRUE[1]) + 4}" class="reveal" style="fill:${css('--warn')}">θ⋆</text>`;
  $('pathChart').innerHTML = s + '</svg>';
}

// ---------- events ----------
$('btnStep').onclick = () => { stop(); step(); };
$('btnPlay').onclick = () => { if (S.playing) { stop(); return; } $('btnPlay').setAttribute('aria-pressed', 'true'); $('btnPlay').textContent = T.pause;
  S.playing = setInterval(() => { step(); if (converged() || S.hist.length > 600) stop(); }, window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 400 : 90); };
$('btnConv').onclick = () => { stop(); for (let k = 0; k < 800 && !converged(); k++) { const r = EM.iterate(cur().th, data, P(), G); cur().elbo = r.elbo; S.hist.push({ th: r.next }); } render(); };
$('btnReset').onclick = reset;
$('btnReveal').onclick = () => { S.reveal = !S.reveal; render(); };
['th1', 'th2', 'lam', 'gam', 'kSel', 'opSel', 'bhatOnly', 'alpha'].forEach(k => $(k).addEventListener('input', reset));
['bSel', 'bTarget'].forEach(k => $(k).addEventListener('input', render));
$('btnGen').onclick = () => { data = EM.generate(+$('seed').value || 1, Math.min(40, Math.max(5, +$('steps').value || 15)), P()); reset(); };
$('btnDefault').onclick = () => { data = EM.DATA.map(tr => tr.map(c => c.slice())); $('seed').value = 12; $('steps').value = 15; reset(); };
reset();
try { window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render); new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] }); } catch (e) {}
})();
