/*
 * EM Dispatch Lab: computation core.
 * Teaching-scale version of variational EM for risk-sensitive inverse reinforcement learning
 * (three situations, two actions, two features). All quantities are computed exactly as in
 * the full method: immediate-risk mean-variance Q-function, softmax choice model, trajectory
 * likelihood, discrete posterior over a log-spaced beta grid, per-trajectory gradients that use
 * the whole posterior table, gradient aggregation (mean / PCGrad / CAGrad), and Phase B rule
 * generation. With two actions the M-step derivative can be written in closed form.
 */
const EM = (() => {
  const SIT = ['A', 'B', 'C'];
  const COST = [[0.20, 0.55], [0.20, 0.45], [0.15, 0.40]];   // dispatch cost (divided by 100)
  const PFIX = [[0.45, 0.95], [0.70, 0.95], [0.95, 0.95]];   // probability of a first-time fix
  const PS = [0.3, 0.4, 0.3];                                  // probability that the next job is A, B, C (independent of the action)
  const THETA_TRUE = [-2, -3];
  const BETA_TRUE = [0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.9, 2.4, 3.0];
  // Default data: 10 trajectories of 15 steps generated with numpy seed 12; [situation, action], action 0 = lowest cost, 1 = skill first
  const DATA = [[[0,1],[2,0],[0,0],[0,1],[1,1],[0,0],[1,0],[0,0],[2,0],[2,1],[0,0],[1,0],[0,0],[0,1],[1,1]],[[2,0],[1,0],[2,0],[0,1],[0,1],[2,0],[0,1],[2,0],[0,0],[1,0],[1,0],[1,0],[0,0],[2,0],[2,1]],[[2,0],[1,0],[1,1],[1,1],[1,1],[1,1],[0,1],[2,0],[2,0],[0,1],[1,0],[2,0],[2,0],[2,0],[0,1]],[[0,1],[2,1],[0,1],[1,0],[0,0],[0,1],[1,0],[2,0],[1,0],[2,0],[2,0],[1,0],[2,0],[0,1],[0,1]],[[0,1],[2,1],[0,1],[0,1],[0,1],[0,1],[0,1],[2,0],[1,0],[1,1],[0,0],[2,1],[1,1],[2,0],[1,1]],[[2,1],[1,1],[1,0],[2,1],[1,1],[1,1],[1,1],[0,1],[0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,0]],[[1,1],[1,1],[1,0],[1,1],[2,0],[1,1],[0,1],[0,1],[0,1],[0,1],[1,1],[0,1],[2,0],[2,0],[0,1]],[[2,0],[1,1],[2,0],[2,1],[2,0],[2,0],[1,1],[0,1],[1,1],[0,1],[1,1],[0,1],[0,1],[2,0],[2,0]],[[0,1],[0,1],[0,1],[1,1],[0,1],[0,1],[0,1],[2,0],[2,0],[0,1],[1,1],[1,1],[0,1],[2,0],[2,0]],[[2,0],[2,0],[2,0],[2,0],[0,1],[2,0],[2,0],[0,1],[2,1],[2,0],[1,1],[1,1],[1,1],[0,1],[1,1]]];

  const sig = x => 1 / (1 + Math.exp(-x));
  const lsig = x => (x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)));   // numerically stable log sigmoid

  function grid(K, lo = 0.02, hi = 10, sd = 0.75) {
    const nodes = K === 7 ? [0.25, 0.4, 0.63, 1.0, 1.58, 2.5, 4.0] : Array.from({ length: K }, (_, n) => Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * n / (K - 1)));
    const lp = nodes.map(b => -0.5 * (Math.log(b) / sd) ** 2); const m = Math.max(...lp); let z = 0; const pr = lp.map(v => { const e = Math.exp(v - m); z += e; return e; });
    return { nodes, prior: pr.map(v => v / z) };
  }
  // Mean-variance Q: Q_beta(s,a) = E[R] - 1/2 (beta - 1) Var[R] / R0 + gamma E[V(s')]; R = theta1 * cost + theta2 * rework
  function qTable(th, beta, P) {
    const base = [0, 1, 2].map(s => [0, 1].map(a => {
      const ER = th[0] * COST[s][a] + th[1] * (1 - PFIX[s][a]);
      const VR = th[1] * th[1] * PFIX[s][a] * (1 - PFIX[s][a]);
      return { ER, VR, pen: 0.5 * (beta - 1) * VR / P.R0, base: ER - 0.5 * (beta - 1) * VR / P.R0 };
    }));
    const M = [0, 1, 2].reduce((acc, s) => acc + PS[s] * Math.max(base[s][0].base, base[s][1].base), 0);
    const C = P.gamma * M / (1 - P.gamma);          // gamma E[V(s')]: the next situation does not depend on the action, so both actions share it
    return base.map(row => row.map(c => ({ ...c, cont: C, Q: c.base + C })));
  }
  // Q difference between the two actions (skill first minus lowest cost) and its derivative with respect to theta
  function dQ(th, beta, P) {
    return [0, 1, 2].map(s => {
      const dmu = [COST[s][1] - COST[s][0], PFIX[s][0] - PFIX[s][1]];
      const dv = PFIX[s][1] * (1 - PFIX[s][1]) - PFIX[s][0] * (1 - PFIX[s][0]);
      const d = th[0] * dmu[0] + th[1] * dmu[1] - 0.5 * (beta - 1) * th[1] * th[1] * dv / P.R0;
      const grad = [dmu[0], dmu[1] - (beta - 1) * th[1] * dv / P.R0];     // d(dQ)/d(theta)
      return { d, grad, dmu, dv };
    });
  }
  function counts(data) { return data.map(tr => { const c = [[0, 0], [0, 0], [0, 0]]; tr.forEach(([s, a]) => c[s][a]++); return c; }); }

  // One full EM iteration (E-step + M-step); returns every intermediate quantity for display
  function iterate(th, data, P, G) {
    const N = data.length, K = G.nodes.length, cnt = counts(data);
    const dq = G.nodes.map(b => dQ(th, b, P));                                   // K x 3
    const pi = dq.map(row => row.map(x => sig(P.lambda * x.d)));                 // softmax choice: P(skill first | s)
    const LL = cnt.map(c => dq.map(row => row.reduce((acc, x, s) => acc + c[s][1] * lsig(P.lambda * x.d) + c[s][0] * lsig(-P.lambda * x.d), 0)));   // trajectory log-likelihood
    const W = [], bhat = [], sd = [], ml = [];
    for (let i = 0; i < N; i++) {                                                 // posterior table and its summary
      const lw = LL[i].map((v, n) => Math.log(G.prior[n]) + v); const m = Math.max(...lw);
      const e = lw.map(v => Math.exp(v - m)); const z = e.reduce((a, b) => a + b, 0);
      const w = e.map(v => v / z); W.push(w); ml.push(m + Math.log(z));
      const bh = w.reduce((a, v, n) => a + v * G.nodes[n], 0); bhat.push(bh);
      sd.push(Math.sqrt(w.reduce((a, v, n) => a + v * (G.nodes[n] - bh) ** 2, 0)));
    }
    const elbo = ml.reduce((a, b) => a + b, 0);                                   // ELBO after the E-step
    const suggest = (c, row, lam) => row.reduce((acc, x, s) => {                  // inner sum: sum_t lambda (1[a = skill] - pi) d(dQ)/d(theta)
      const r = c[s][1] - (c[s][0] + c[s][1]) * sig(lam * x.d); return [acc[0] + lam * r * x.grad[0], acc[1] + lam * r * x.grad[1]];
    }, [0, 0]);
    const Gn = cnt.map(c => dq.map(row => suggest(c, row, P.lambda)));          // N x K x 2: suggestion if the trajectory had the n-th beta
    let g;
    if (P.bhatOnly) g = cnt.map((c, i) => suggest(c, dQ(th, bhat[i], P), P.lambda));   // wrong shortcut: use beta-hat only
    else g = Gn.map((row, i) => row.reduce((acc, v, n) => [acc[0] + W[i][n] * v[0], acc[1] + W[i][n] * v[1]], [0, 0]));   // whole posterior table
    let conflicts = 0; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (g[i][0] * g[j][0] + g[i][1] * g[j][1] < 0) conflicts++;
    const d = aggregate(g, P.op);                                                 // PCGrad / CAGrad / mean
    const next = [th[0] + P.alpha * d[0], th[1] + P.alpha * d[1]];               // gradient step
    return { th, dq, pi, LL, W, bhat, sd, ml, elbo, Gn, g, d, next, conflicts };
  }
  function aggregate(g, op) {
    const N = g.length, mean = [0, 1].map(k => g.reduce((a, v) => a + v[k], 0) / N);
    if (op === 'pcgrad') {                                                        // PCGrad with a fixed order j = 1..N
      const gp = g.map(v => v.slice());
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        if (i === j) continue; const dot = gp[i][0] * g[j][0] + gp[i][1] * g[j][1];
        if (dot < 0) { const nn = g[j][0] ** 2 + g[j][1] ** 2 + 1e-12; gp[i][0] -= dot / nn * g[j][0]; gp[i][1] -= dot / nn * g[j][1]; }
      }
      return [0, 1].map(k => gp.reduce((a, v) => a + v[k], 0) / N);
    }
    if (op === 'cagrad') {                                                        // CAGrad with c = 0.5, 200 projected-gradient steps on the simplex
      const c = 0.5, M = g.map(a => g.map(b => a[0] * b[0] + a[1] * b[1])); const n0 = Math.hypot(mean[0], mean[1]) + 1e-12;
      const m0 = M.map(r => r.reduce((a, b) => a + b, 0) / N); let w = Array(N).fill(1 / N);
      const proj = v => { const u = v.slice().sort((a, b) => b - a); let cs = 0, rho = 0, t = 0; for (let k = 0; k < N; k++) { cs += u[k]; if (u[k] * (k + 1) > cs - 1) { rho = k; t = (cs - 1) / (k + 1); } } return v.map(x => Math.max(x - t, 0)); };
      for (let it = 0; it < 200; it++) {
        const Mw = M.map(r => r.reduce((a, b, k) => a + b * w[k], 0)); const nw = Math.sqrt(Math.max(w.reduce((a, b, k) => a + b * Mw[k], 0), 1e-12));
        const gr = m0.map((v, k) => v + c * n0 * Mw[k] / nw); const gn = Math.hypot(...gr) + 1e-12; w = proj(w.map((v, k) => v - 0.5 * gr[k] / gn));
      }
      const gw = [0, 1].map(k => g.reduce((a, v, i) => a + w[i] * v[k], 0)); const ng = Math.hypot(gw[0], gw[1]) + 1e-12;
      return [0, 1].map(k => mean[k] + c * n0 / ng * gw[k]);
    }
    return mean;
  }
  // Phase B: use theta-hat and a chosen beta to set a rule per situation, then score it with the true reward theta*
  function phaseB(thHat, beta, P) {
    const rule = dQ(thHat, beta, P).map(x => (x.d > 0 ? 1 : 0));
    const perStep = [0, 1, 2].reduce((acc, s) => { const a = rule[s]; return acc + PS[s] * (THETA_TRUE[0] * COST[s][a] + THETA_TRUE[1] * (1 - PFIX[s][a])); }, 0);
    const rework = [0, 1, 2].reduce((acc, s) => acc + PS[s] * (1 - PFIX[s][rule[s]]), 0);
    const cost = [0, 1, 2].reduce((acc, s) => acc + PS[s] * COST[s][rule[s]], 0);
    return { rule, J: perStep / (1 - P.gamma), perStep, rework, cost };
  }
  function mulberry(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function generate(seed, steps, P) {
    const r = mulberry(seed);
    return BETA_TRUE.map(b => { const d = dQ(THETA_TRUE, b, P); return Array.from({ length: steps }, () => { const u = r(); const s = u < PS[0] ? 0 : (u < PS[0] + PS[1] ? 1 : 2); return [s, r() < sig(P.lambda * d[s].d) ? 1 : 0]; }); });
  }
  return { SIT, COST, PFIX, PS, THETA_TRUE, BETA_TRUE, DATA, grid, qTable, dQ, iterate, aggregate, phaseB, generate, sig };
})();
if (typeof module !== 'undefined') module.exports = EM;
