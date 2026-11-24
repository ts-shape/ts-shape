/**
 * The three example charts on the landing page. The numbers come from
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

/* ------------------------------------------------------- cycle potential -- */

export function CyclePotential(): ReactNode {
  const d = data.cycles;
  const rowH = 36;
  const m = {l: 220, r: 56, t: 4, b: 28};
  const H = m.t + m.b + d.segments.length * rowH;
  const max = Math.max(...d.segments.map((s) => s.potential_seconds));
  const scaleMax = Math.ceil(max * 5) / 5;
  const x = (v: number) => m.l + (v / scaleMax) * (W - m.l - m.r);
  const [hover, setHover] = useState<number | null>(null);
  const ticks = Array.from({length: Math.round(scaleMax / 0.2) + 1}, (_, i) => +(i * 0.2).toFixed(1));
  const h = hover === null ? null : d.segments[hover];

  return (
    <figure className={styles.figure}>
      <div className={styles.tiles}>
        <div className={styles.tile}><span>Median cycle (variant {d.variant})</span><strong>{fmt(d.median_seconds)} s</strong></div>
        <div className={styles.tile}><span>Best demonstrated</span><strong>{fmt(d.target_seconds)} s</strong></div>
        <div className={styles.tile}><span>Throughput to gain</span><strong>+{fmt(d.throughput_gain_pct)}%</strong></div>
      </div>
      <div className={styles.plot}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Seconds to gain per cycle step">
          {ticks.map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={m.t} y2={H - m.b} className={styles.grid} />
              <text x={x(v)} y={H - 8} textAnchor="middle" className={styles.axisLabel}>{fmt(v)} s</text>
            </g>
          ))}
          {d.segments.map((s, i) => {
            const cy = m.t + i * rowH + rowH / 2;
            const w = x(s.potential_seconds) - m.l;
            const bh = 20;
            return (
              <g
                key={s.label}
                tabIndex={0}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              >
                <rect x={0} y={cy - rowH / 2} width={W} height={rowH} className={styles.hit} />
                <text x={m.l - 10} y={cy + 4} textAnchor="end" className={styles.rowLabel}>{s.label}</text>
                <path
                  d={`M${m.l},${cy - bh / 2} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} Z`}
                  className={`${styles.bar} ${hover === i ? styles.active : ''}`}
                />
                {i === 0 && (
                  <text x={x(s.potential_seconds) + 6} y={cy + 4} className={styles.valueLabel}>{fmt(s.potential_seconds, 2)} s</text>
                )}
              </g>
            );
          })}
        </svg>
        {h && (
          <Tooltip x={(m.l / W) * 100} y={((m.t + ((hover ?? 0) + 1) * rowH) / H) * 100}>
            <strong>{fmt(h.potential_seconds, 2)} s to gain</strong>
            <span>{h.label}</span>
            <span>median {fmt(h.median_seconds, 2)} s, best 10% {fmt(h.p10_seconds, 2)} s</span>
          </Tooltip>
        )}
      </div>
      <details className={styles.table}>
        <summary>Data: top {d.segments.length} steps of {d.cycle_count} cycles</summary>
        <table>
          <thead><tr><th>Step</th><th>To gain (s)</th><th>Median (s)</th><th>Best 10% (s)</th></tr></thead>
          <tbody>
            {d.segments.map((s) => (
              <tr key={s.label}><td>{s.label}</td><td>{fmt(s.potential_seconds, 2)}</td><td>{fmt(s.median_seconds, 2)}</td><td>{fmt(s.p10_seconds, 2)}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

export const exampleCode = {
  spc: data.spc.code,
  line: data.line.code,
  cycles: data.cycles.code,
};

export const exampleFacts = {
  spc: data.spc,
  line: data.line,
  cycles: data.cycles,
};
