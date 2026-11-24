/**
 * The example charts on the landing page. The numbers come from
 * website/src/data/landing-examples.json, written by
 * `python scripts/landing_examples.py` (ts-shape run on its simulated data).
 *
 * Plain SVG: one axis per chart, 2px lines, <=24px bars with 4px rounded
 * data ends, hairline grid, legend for two or more series, hover tooltips
 * (crosshair on the line, per mark on bars), and a data table for each.
 */
import {useState, type ReactNode, type PointerEvent} from 'react';
import data from '@site/src/data/landing-examples.json';

import styles from './LandingCharts.module.css';

const W = 640;

const time = (iso: string): number => new Date(`${iso}Z`).getTime();
const hhmm = (iso: string): string => iso.slice(11, 16);
const fmt = (v: number, digits = 1): string =>
  v.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});

function Tooltip({x, y, children}: {x: number; y: number; children: ReactNode}): ReactNode {
  // x, y in percent of the chart box; flipped to stay inside it
  const style = {
    left: x > 60 ? undefined : `${x}%`,
    right: x > 60 ? `${100 - x}%` : undefined,
    top: `${Math.max(0, y)}%`,
  };
  return (
    <div className={styles.tooltip} style={style} role="status">
      {children}
    </div>
  );
}

function Legend({items}: {items: {label: string; className: string; shape?: 'line' | 'dot' | 'tri' | 'box'}[]}): ReactNode {
  return (
    <ul className={styles.legend}>
      {items.map((it) => (
        <li key={it.label}>
          <svg width="18" height="12" aria-hidden="true">
            {it.shape === 'line' && <line x1="1" y1="6" x2="17" y2="6" className={it.className} />}
            {it.shape === 'dot' && <circle cx="9" cy="6" r="4" className={it.className} />}
            {it.shape === 'tri' && <path d="M9 1.5 L13.5 10 L4.5 10 Z" className={it.className} />}
            {(it.shape ?? 'box') === 'box' && <rect x="2" y="1" width="14" height="10" rx="2" className={it.className} />}
          </svg>
          {it.label}
        </li>
      ))}
    </ul>
  );
}

/* ---------------------------------------------------------------- SPC -- */

export function SpcChart(): ReactNode {
  const d = data.spc;
  const H = 260;
  const m = {l: 44, r: 84, t: 12, b: 28};
  const t0 = time(d.points[0][0] as string);
  const t1 = time(d.points[d.points.length - 1][0] as string);
  const values = d.points.map((p) => p[1] as number);
  const lo = Math.floor(Math.min(...values, d.lcl) - 0.5);
  const hi = Math.ceil(Math.max(...values, d.ucl) + 0.5);
  const x = (iso: string) => m.l + ((time(iso) - t0) / (t1 - t0)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const path = d.points.map((p, i) => `${i ? 'L' : 'M'}${x(p[0] as string).toFixed(1)},${y(p[1] as number).toFixed(1)}`).join('');
  const flagged = new Map<string, string>();
  d.run_rules.forEach((r) => flagged.set(r[0] as string, `run rule ${String(r[2]).replace('rule_', '')}`));
  d.out_of_limits.forEach((r) => flagged.set(r[0] as string, 'beyond 3σ (rule 1)'));
  const [hover, setHover] = useState<number | null>(null);

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const frac = (px - m.l) / (W - m.l - m.r);
    const i = Math.round(Math.min(1, Math.max(0, frac)) * (d.points.length - 1));
    setHover(i);
  };
  const ticks = Array.from({length: 13}, (_, i) => ({
    h: i,
    iso: new Date(t0 + i * 30 * 60e3).toISOString().slice(0, 19),
  })).filter((tk) => time(tk.iso) <= t1);
  const hp = hover === null ? null : d.points[hover];

  return (
    <figure className={styles.figure}>
      <Legend
        items={[
          {label: 'Oven temperature', className: styles.mLine, shape: 'line'},
          {label: 'Beyond 3σ', className: styles.mCritical, shape: 'dot'},
          {label: 'Run rules flag a drift', className: styles.mBand},
        ]}
      />
      <div className={styles.plot}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Oven temperature with control limits and flagged points">
          {d.run_spans.map(([a, b]) => (
            <rect
              key={a}
              x={x(a) - 2}
              y={m.t}
              width={Math.max(4, x(b) - x(a) + 4)}
              height={H - m.t - m.b}
              className={styles.band}
            />
          ))}
          {[d.lcl, d.mean, d.ucl].map((v, i) => (
            <g key={v}>
              <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} className={i === 1 ? styles.grid : styles.limit} />
              <text x={W - m.r + 6} y={y(v) + 4} className={styles.axisLabel}>
                {['LCL', 'mean', 'UCL'][i]} {fmt(v)}
              </text>
            </g>
          ))}
          <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} className={styles.axis} />
          {ticks.map((tk) => (
            <text key={tk.h} x={x(tk.iso)} y={H - 8} textAnchor="middle" className={styles.axisLabel}>
              {hhmm(tk.iso)}
            </text>
          ))}
          {Array.from({length: Math.floor((hi - lo) / 2) + 1}, (_, i) => lo + i * 2).map((v) => (
            <text key={v} x={m.l - 8} y={y(v) + 4} textAnchor="end" className={styles.axisLabel}>
              {v}
            </text>
          ))}
          <path d={path} className={styles.line} />
          {d.out_of_limits.map((r) => (
            <circle key={r[0] as string} cx={x(r[0] as string)} cy={y(r[1] as number)} r={4.5} className={styles.critical} />
          ))}
          {hp && (
            <g>
              <line x1={x(hp[0] as string)} x2={x(hp[0] as string)} y1={m.t} y2={H - m.b} className={styles.crosshair} />
              <circle cx={x(hp[0] as string)} cy={y(hp[1] as number)} r={4} className={styles.hoverDot} />
            </g>
          )}
          <rect
            x={m.l}
            y={m.t}
            width={W - m.l - m.r}
            height={H - m.t - m.b}
            className={styles.hit}
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {hp && (
          <Tooltip x={(x(hp[0] as string) / W) * 100} y={4}>
            <strong>{fmt(hp[1] as number, 2)} °C</strong>
            <span>{hhmm(hp[0] as string)}</span>
            {flagged.has(hp[0] as string) && <span className={styles.flag}>{flagged.get(hp[0] as string)}</span>}
          </Tooltip>
        )}
      </div>
      <details className={styles.table}>
        <summary>Data: {d.out_of_limits.length} points beyond 3σ, {d.run_rules.length} run-rule points</summary>
        <table>
          <thead>
            <tr><th>Time</th><th>°C</th><th>Rule</th></tr>
          </thead>
          <tbody>
            {[...d.out_of_limits.map((r) => [r[0], r[1], 'rule 1 (beyond 3σ)']), ...d.run_rules.map((r) => [r[0], r[1], String(r[2]).replace('_', ' ')])]
              .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
              .map((r) => (
                <tr key={`${r[0]}${r[2]}`}><td>{hhmm(r[0] as string)}</td><td>{fmt(r[1] as number, 2)}</td><td>{r[2]}</td></tr>
              ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

/* ------------------------------------------------------- line timeline -- */

export function LineTimeline(): ReactNode {
  const d = data.line;
  const H = 96;
  const m = {l: 22, r: 22, t: 12, b: 28};
  const t0 = time(d.intervals[0][0]);
  const t1 = time(d.intervals[d.intervals.length - 1][1]);
  const x = (iso: string) => m.l + ((time(iso) - t0) / (t1 - t0)) * (W - m.l - m.r);
  const [hover, setHover] = useState<number | null>(null);
  const ticks = Array.from({length: 10}, (_, i) => new Date(t0 + i * 2 * 3600e3).toISOString().slice(0, 19)).filter(
    (iso) => time(iso) <= t1,
  );
  const minutes = (a: string, b: string) => (time(b) - time(a)) / 60e3;
  const h = hover === null ? null : d.intervals[hover];
  const barY = m.t + 8;
  const barH = 28;

  return (
    <figure className={styles.figure}>
      <div className={styles.tiles}>
        <div className={styles.tile}><span>OEE</span><strong>{fmt(d.oee_pct)}%</strong></div>
        <div className={styles.tile}><span>Availability</span><strong>{fmt(d.availability_pct)}%</strong></div>
        <div className={styles.tile}><span>Performance</span><strong>{fmt(d.performance_pct)}%</strong></div>
        <div className={styles.tile}><span>Stops</span><strong>{d.stop_count}</strong><em>{Math.round(d.stopped_minutes)} min, longest {Math.round(d.longest_stop_minutes)}</em></div>
      </div>
      <Legend
        items={[
          {label: 'Running', className: styles.mRun},
          {label: 'Stopped', className: styles.mIdle},
        ]}
      />
      <div className={styles.plot}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Line running and stopped over one day">
          {d.intervals.map((iv, i) => {
            const x0 = x(iv[0]);
            const x1 = x(iv[1]);
            return (
              <rect
                key={iv[0]}
                x={x0 + 1}
                y={barY}
                width={Math.max(1, x1 - x0 - 2)}
                height={barH}
                rx={4}
                className={`${iv[2] === 'run' ? styles.run : styles.idle} ${hover === i ? styles.active : ''}`}
                tabIndex={0}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              />
            );
          })}
          {ticks.map((iso) => (
            <text key={iso} x={x(iso)} y={H - 8} textAnchor="middle" className={styles.axisLabel}>
              {hhmm(iso)}
            </text>
          ))}
        </svg>
        {h && (
          <Tooltip x={(((x(h[0]) + x(h[1])) / 2) / W) * 100} y={0}>
            <strong>{fmt(minutes(h[0], h[1]), 0)} min</strong>
            <span>{h[2] === 'run' ? 'Running' : 'Stopped'}, {hhmm(h[0])}–{hhmm(h[1])}</span>
          </Tooltip>
        )}
      </div>
      <details className={styles.table}>
        <summary>Data: {d.stop_count} stops</summary>
        <table>
          <thead><tr><th>From</th><th>To</th><th>Minutes</th></tr></thead>
          <tbody>
            {d.intervals.filter((iv) => iv[2] === 'idle').map((iv) => (
              <tr key={iv[0]}><td>{hhmm(iv[0])}</td><td>{hhmm(iv[1])}</td><td>{fmt(minutes(iv[0], iv[1]))}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

/* ------------------------------------------------------- loop responses -- */

export function LoopResponses(): ReactNode {
  const d = data.loops;
  const panelW = 196;
  const gap = 26;
  const H = 200;
  const m = {l: 34, r: 6, t: 40, b: 26};
  const tMin = -60;
  const tMax = 600;
  const lo = -20;
  const hi = 140;
  const [hover, setHover] = useState<{loop: number; i: number} | null>(null);
  const px = (k: number) => k * (panelW + gap);
  const x = (k: number, t: number) => px(k) + m.l + ((t - tMin) / (tMax - tMin)) * (panelW - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const total = 3 * panelW + 2 * gap;
  const needTuning = d.loops.filter((l) => l.diagnosis !== 'ok').length;
  const verdict = (l: (typeof d.loops)[number]) =>
    l.diagnosis === 'ok' ? 'well tuned' : l.diagnosis === 'oscillating' ? 'oscillates' : 'sluggish';
  const onMove = (k: number) => (e: PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const vx = ((e.clientX - box.left) / box.width) * total;
    const frac = (vx - px(k) - m.l) / (panelW - m.l - m.r);
    const r = d.loops[k].response;
    setHover({loop: k, i: Math.round(Math.min(1, Math.max(0, frac)) * (r.length - 1))});
  };
  const hp = hover === null ? null : d.loops[hover.loop].response[hover.i];

  return (
    <figure className={styles.figure}>
      <div className={styles.tiles}>
        <div className={styles.tile}><span>Loops found</span><strong>{d.loops.length}</strong><em>no tag mapping</em></div>
        <div className={styles.tile}><span>Need tuning</span><strong>{needTuning}</strong><em>{d.loops.filter((l) => l.diagnosis !== 'ok').map((l) => l.name).join(', ')}</em></div>
        <div className={styles.tile}><span>Setpoint steps assessed</span><strong>{d.loops.reduce((a, l) => a + l.steps, 0)}</strong></div>
      </div>
      <Legend
        items={[
          {label: 'Process value', className: styles.mLine, shape: 'line'},
          {label: 'New setpoint', className: styles.mSetpoint, shape: 'line'},
        ]}
      />
      <div className={styles.plot}>
        <svg viewBox={`0 0 ${total} ${H}`} role="img" aria-label="Setpoint step response of three control loops">
          {d.loops.map((l, k) => {
            const path = l.response.map((p, i) => `${i ? 'L' : 'M'}${x(k, p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join('');
            return (
              <g key={l.name}>
                <text x={px(k) + m.l} y={14} className={styles.panelTitle}>{l.name}</text>
                <text x={px(k) + m.l} y={30} className={l.diagnosis === 'ok' ? styles.axisLabel : styles.flagLabel}>
                  {verdict(l)}
                </text>
                {[0, 100].map((v) => (
                  <line key={v} x1={px(k) + m.l} x2={px(k) + panelW - m.r} y1={y(v)} y2={y(v)} className={v === 100 ? styles.setpoint : styles.grid} />
                ))}
                {k === 0 && [0, 100].map((v) => (
                  <text key={v} x={m.l - 6} y={y(v) + 4} textAnchor="end" className={styles.axisLabel}>{v}%</text>
                ))}
                <line x1={x(k, 0)} x2={x(k, 0)} y1={m.t} y2={H - m.b} className={styles.grid} />
                <line x1={px(k) + m.l} x2={px(k) + panelW - m.r} y1={H - m.b} y2={H - m.b} className={styles.axis} />
                {[0, 5, 10].map((mins) => (
                  <text key={mins} x={x(k, mins * 60)} y={H - 8} textAnchor="middle" className={styles.axisLabel}>
                    {mins} min
                  </text>
                ))}
                <path d={path} className={styles.line} />
                {hover?.loop === k && hp && (
                  <g>
                    <line x1={x(k, hp[0])} x2={x(k, hp[0])} y1={m.t} y2={H - m.b} className={styles.crosshair} />
                    <circle cx={x(k, hp[0])} cy={y(hp[1])} r={4} className={styles.hoverDot} />
                  </g>
                )}
                <rect
                  x={px(k) + m.l}
                  y={m.t}
                  width={panelW - m.l - m.r}
                  height={H - m.t - m.b}
                  className={styles.hit}
                  onPointerMove={onMove(k)}
                  onPointerLeave={() => setHover(null)}
                />
              </g>
            );
          })}
        </svg>
        {hover && hp && (
          <Tooltip x={(x(hover.loop, hp[0]) / total) * 100} y={12}>
            <strong>{fmt(hp[1], 0)}% of the step</strong>
            <span>{d.loops[hover.loop].name}, {hp[0] < 0 ? `${-hp[0]} s before` : `${hp[0]} s after`} the step</span>
          </Tooltip>
        )}
      </div>
      <details className={styles.table}>
        <summary>Data: one row per loop, medians over its setpoint steps</summary>
        <table>
          <thead><tr><th>Loop</th><th>Diagnosis</th><th>Steps</th><th>Overshoot</th><th>Settle time</th><th>Steps settled</th></tr></thead>
          <tbody>
            {d.loops.map((l) => (
              <tr key={l.name}><td>{l.name}</td><td>{l.diagnosis}</td><td>{l.steps}</td><td>{fmt(l.overshoot_pct)}%</td><td>{l.settle_seconds} s</td><td>{l.settled_pct}%</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

/* ------------------------------------------------------- energy per part -- */

type EnergyWindow = [start: string, kwhPerPart: number, parts: number, kwh: number];

export function EnergyPerPart(): ReactNode {
  const d = {...data.energy, windows: data.energy.windows as EnergyWindow[]};
  const H = 220;
  const m = {l: 44, r: 92, t: 12, b: 28};
  const n = d.windows.length;
  const band = (W - m.l - m.r) / n;
  const bw = Math.min(24, band - 4);
  const lo = 0.25;
  const hi = Math.ceil(d.worst_kwh_per_part * 20) / 20;
  const y = (v: number) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const [hover, setHover] = useState<number | null>(null);
  const h = hover === null ? null : d.windows[hover];
  const ticks: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += 0.05) ticks.push(+v.toFixed(2));

  return (
    <figure className={styles.figure}>
      <div className={styles.tiles}>
        <div className={styles.tile}><span>Median</span><strong>{fmt(d.median_kwh_per_part, 2)} kWh</strong><em>per part</em></div>
        <div className={styles.tile}><span>Best half hour</span><strong>{fmt(d.best_kwh_per_part, 2)} kWh</strong><em>per part</em></div>
        <div className={styles.tile}><span>Above best rate</span><strong>{fmt(d.excess_kwh, 0)} kWh</strong><em>of {fmt(d.kwh, 0)} kWh, {d.parts} parts</em></div>
      </div>
      <div className={styles.plot}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Energy per part, per half hour">
          {ticks.map((v) => (
            <g key={v}>
              <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} className={styles.grid} />
              <text x={m.l - 8} y={y(v) + 4} textAnchor="end" className={styles.axisLabel}>{fmt(v, 2)}</text>
            </g>
          ))}
          {d.windows.map((wdw, i) => {
            const cx = m.l + band * i + band / 2;
            const top = y(wdw[1]);
            const bh = y(lo) - top;
            const worst = wdw[0] === d.worst_window;
            return (
              <g
                key={wdw[0]}
                tabIndex={0}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              >
                <rect x={cx - band / 2} y={m.t} width={band} height={H - m.t - m.b} className={styles.hit} />
                <path
                  d={`M${cx - bw / 2},${y(lo)} v${-(bh - 4)} a4,4 0 0 1 4,-4 h${bw - 8} a4,4 0 0 1 4,4 v${bh - 4} Z`}
                  className={`${worst ? styles.barAlert : styles.bar} ${hover === i ? styles.active : ''}`}
                />
                {worst && (
                  <text x={cx} y={top - 6} textAnchor="middle" className={styles.valueLabel}>
                    {fmt(wdw[1], 2)}
                  </text>
                )}
                {i % 4 === 0 && (
                  <text x={cx} y={H - 8} textAnchor="middle" className={styles.axisLabel}>{hhmm(wdw[0])}</text>
                )}
              </g>
            );
          })}
          <line x1={m.l} x2={W - m.r} y1={y(d.median_kwh_per_part)} y2={y(d.median_kwh_per_part)} className={styles.median} />
          <text x={W - m.r + 6} y={y(d.median_kwh_per_part) + 4} className={styles.axisLabel}>median {fmt(d.median_kwh_per_part, 2)}</text>
          <line x1={m.l} x2={W - m.r} y1={y(lo)} y2={y(lo)} className={styles.axis} />
        </svg>
        {h && (
          <Tooltip x={((m.l + band * (hover ?? 0) + band / 2) / W) * 100} y={0}>
            <strong>{fmt(h[1], 3)} kWh per part</strong>
            <span>{hhmm(h[0])}–{hhmm(new Date(time(h[0]) + 30 * 60e3).toISOString().slice(0, 19))}</span>
            <span>{h[2]} parts, {fmt(h[3])} kWh</span>
          </Tooltip>
        )}
      </div>
      <details className={styles.table}>
        <summary>Data: {n} half hours</summary>
        <table>
          <thead><tr><th>From</th><th>kWh / part</th><th>Parts</th><th>kWh</th></tr></thead>
          <tbody>
            {d.windows.map((wdw) => (
              <tr key={wdw[0]}><td>{hhmm(wdw[0])}</td><td>{fmt(wdw[1], 3)}</td><td>{wdw[2]}</td><td>{fmt(wdw[3])}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

/* --------------------------------------------------------------- drivers -- */

export function DriversChart(): ReactNode {
  const d = data.drivers;
  const rowH = 36;
  const m = {l: 150, r: 150, t: 4, b: 28};
  const H = m.t + m.b + d.signals.length * rowH;
  const decades = [0, 1, 2, 3, 4, 5]; // F statistic on a log scale, 1 .. 100 000
  const x = (f: number) => m.l + (Math.log10(Math.max(1, f)) / 5) * (W - m.l - m.r);
  const [hover, setHover] = useState<number | null>(null);
  const h = hover === null ? null : d.signals[hover];

  return (
    <figure className={styles.figure}>
      <Legend
        items={[
          {label: 'Drives the temperature (significant)', className: styles.mBar},
          {label: 'No effect', className: styles.mMuted},
        ]}
      />
      <div className={styles.plot}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Granger causality F statistic per candidate signal">
          {decades.map((k) => (
            <g key={k}>
              <line x1={x(10 ** k)} x2={x(10 ** k)} y1={m.t} y2={H - m.b} className={styles.grid} />
              <text x={x(10 ** k)} y={H - 8} textAnchor="middle" className={styles.axisLabel}>
                {(10 ** k).toLocaleString('en-US')}
              </text>
            </g>
          ))}
          {d.signals.map((sg, i) => {
            const cy = m.t + i * rowH + rowH / 2;
            const w = Math.max(6, x(sg.f_statistic) - m.l);
            const bh = 20;
            return (
              <g
                key={sg.uuid}
                tabIndex={0}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              >
                <rect x={0} y={cy - rowH / 2} width={W} height={rowH} className={styles.hit} />
                <text x={m.l - 10} y={cy + 4} textAnchor="end" className={styles.rowLabel}>{sg.label}</text>
                <path
                  d={`M${m.l},${cy - bh / 2} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} Z`}
                  className={`${sg.causal ? styles.bar : styles.barMuted} ${hover === i ? styles.active : ''}`}
                />
                <text x={m.l + w + 8} y={cy + 4} className={sg.causal ? styles.valueLabel : styles.axisLabel}>
                  {sg.causal ? `drives it, ${sg.lag * 2} s ahead` : 'no effect'}
                </text>
              </g>
            );
          })}
        </svg>
        {h && (
          <Tooltip x={(m.l / W) * 100} y={((m.t + ((hover ?? 0) + 1) * rowH) / H) * 100}>
            <strong>F = {h.f_statistic.toLocaleString('en-US')}</strong>
            <span>{h.label} → process temperature</span>
            <span>adjusted p = {h.p_value_adjusted < 1e-6 ? '< 0.000001' : h.p_value_adjusted}</span>
          </Tooltip>
        )}
      </div>
      <details className={styles.table}>
        <summary>Data: Granger test of each signal against the process temperature</summary>
        <table>
          <thead><tr><th>Signal</th><th>F statistic</th><th>Adjusted p</th><th>Best lag</th><th>Causal</th></tr></thead>
          <tbody>
            {d.signals.map((sg) => (
              <tr key={sg.uuid}><td>{sg.label}</td><td>{sg.f_statistic.toLocaleString('en-US')}</td><td>{sg.p_value_adjusted}</td><td>{sg.lag * 2} s</td><td>{sg.causal ? 'yes' : 'no'}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

export const exampleCode = {
  line: data.line.code,
  loops: data.loops.code,
  spc: data.spc.code,
  energy: data.energy.code,
  drivers: data.drivers.code,
};

export const exampleFacts = data;
