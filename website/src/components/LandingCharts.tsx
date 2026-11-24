/**
 * The example charts on the landing page: one insight each, readable at a
 * glance. The numbers come from website/src/data/landing-examples.json,
 * written by `python scripts/landing_examples.py` (ts-shape run on its
 * simulated data; the page shows the code that produced them).
 *
 * Plain SVG: one axis per chart, 2px lines, rounded data ends, hairline
 * grid, direct labels on the marks that matter, hover tooltips, and a data
 * table for each chart.
 */
import {useEffect, useRef, useState, type ReactNode, type PointerEvent, type RefObject} from 'react';
import data from '@site/src/data/landing-examples.json';

import styles from './LandingCharts.module.css';

/**
 * Draw at the plot's real width so 12px text stays 12px; phones get a
 * minimum width and scroll sideways. Before hydration: a desktop width.
 */
function usePlotWidth(min = 640): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1060);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = () => setWidth(Math.max(min, Math.round(el.clientWidth)));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return [ref, width];
}

const time = (iso: string): number => new Date(`${iso}Z`).getTime();
const hhmm = (iso: string): string => iso.slice(11, 16);
const fmt = (v: number, digits = 1): string =>
  v.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
const hours = (v: number): string => `${fmt(v)} h`;

function Tooltip({x, y, children}: {x: number; y: number; children: ReactNode}): ReactNode {
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

function DataTable({summary, head, rows}: {summary: string; head: string[]; rows: (string | number)[][]}): ReactNode {
  return (
    <details className={styles.table}>
      <summary>{summary}</summary>
      <table>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.join('|')}>
              {r.map((c, i) => (
                <td key={i}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

/* ------------------------------------------------ production: the day -- */

export function ProductionChart(): ReactNode {
  const d = data.line;
  const parts = [
    {key: 'productive', label: 'Productive', value: d.productive_hours, cls: styles.s1},
    {key: 'slow', label: 'Running slow', value: d.slow_hours, cls: styles.s3},
    {key: 'stopped', label: 'Stopped', value: d.stopped_hours, cls: styles.s2},
  ];
  const total = parts.reduce((a, p) => a + p.value, 0);
  const [hover, setHover] = useState<number | null>(null);
  const [stopHover, setStopHover] = useState<number | null>(null);
  const [plotRef, W] = usePlotWidth(280);
  const tickHours = W < 600 ? 4 : 2;

  // Thin stop timeline under the split
  const t0 = time(d.intervals[0][0]);
  const t1 = time(d.intervals[d.intervals.length - 1][1]);
  const top = 26;
  const barH = 12;
  const H = 64;
  const x = (iso: string) => ((time(iso) - t0) / (t1 - t0)) * W;
  const longestMid = (x(d.longest_stop[0]) + x(d.longest_stop[1])) / 2;
  const ticks = Array.from({length: 10}, (_, i) => new Date(t0 + i * tickHours * 3600e3).toISOString().slice(0, 19)).filter(
    (iso) => time(iso) <= t1,
  );
  const sh = stopHover === null ? null : d.intervals[stopHover];
  const minutes = (a: string, b: string) => (time(b) - time(a)) / 60e3;

  return (
    <figure className={styles.figure}>
      <div
        className={styles.split}
        role="img"
        aria-label={`Of ${d.planned_hours} planned hours: ${parts
          .map((p) => `${p.value} h ${p.label.toLowerCase()}`)
          .join(', ')}`}
      >
        {parts.map((p, i) => (
          <div
            key={p.key}
            className={`${styles.segment} ${p.cls} ${hover !== null && hover !== i ? styles.dim : ''}`}
            style={{flexGrow: p.value / total}}
            tabIndex={0}
            title={`${hours(p.value)} ${p.label.toLowerCase()} (${fmt((p.value / total) * 100, 0)}%)`}
            onPointerEnter={() => setHover(i)}
            onPointerLeave={() => setHover(null)}
            onFocus={() => setHover(i)}
            onBlur={() => setHover(null)}
          >
            <span className={styles.segmentValue}>{hours(p.value)}</span>
            <span className={styles.segmentLabel}>{p.label}</span>
          </div>
        ))}
      </div>
      <div className={styles.timelineHead}>
        <span>When the line stopped</span>
        <span>
          {d.stop_count} stops · longest {Math.round(d.longest_stop_minutes)} min
        </span>
      </div>
      <div className={styles.plot} ref={plotRef}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Stops through the day">
          <rect x={0} y={top} width={W} height={barH} rx={3} className={styles.track} />
          {d.intervals.map((iv, i) =>
            iv[2] === 'idle' ? (
              <rect
                key={iv[0]}
                x={x(iv[0])}
                y={top}
                width={Math.max(3, x(iv[1]) - x(iv[0]))}
                height={barH}
                rx={2}
                className={styles.stop}
                tabIndex={0}
                onPointerEnter={() => setStopHover(i)}
                onPointerLeave={() => setStopHover(null)}
                onFocus={() => setStopHover(i)}
                onBlur={() => setStopHover(null)}
              />
            ) : null,
          )}
          <line x1={longestMid} x2={longestMid} y1={16} y2={top - 2} className={styles.callout} />
          <text x={longestMid} y={12} textAnchor="middle" className={styles.calloutText}>
            longest stop {hhmm(d.longest_stop[0])}–{hhmm(d.longest_stop[1])}
          </text>
          {ticks.map((iso) => (
            <text
              key={iso}
              x={Math.min(W - 16, Math.max(16, x(iso)))}
              y={H - 4}
              textAnchor="middle"
              className={styles.axisLabel}
            >
              {hhmm(iso)}
            </text>
          ))}
        </svg>
        {sh && (
          <Tooltip x={((x(sh[0]) + x(sh[1])) / 2 / W) * 100} y={70}>
            <strong>{fmt(minutes(sh[0], sh[1]), 0)} min stopped</strong>
            <span>
              {hhmm(sh[0])}–{hhmm(sh[1])}
            </span>
          </Tooltip>
        )}
      </div>
      <DataTable
        summary="Data"
        head={['', 'Value']}
        rows={[
          ...parts.map((p) => [p.label, hours(p.value)]),
          ['Planned', hours(d.planned_hours)],
          ['Parts made / possible at ideal speed', `${d.parts.toLocaleString('en-US')} / ${d.parts_possible.toLocaleString('en-US')}`],
          ['Availability · Performance · Quality', `${d.availability_pct}% · ${d.performance_pct}% · ${d.quality_pct}%`],
        ]}
      />
    </figure>
  );
}

/* --------------------------------------------- cycle time: what to fix -- */

export function CycleChart(): ReactNode {
  const d = data.cycles;
  const [plotRef, W] = usePlotWidth(280);
  const narrow = W < 600; // labels above the bars instead of beside them
  const rowH = narrow ? 50 : 40;
  const m = {l: narrow ? 0 : 250, r: 64, t: 4, b: 26};
  const H = m.t + m.b + d.segments.length * rowH;
  const max = Math.max(...d.segments.map((s) => s.potential_seconds));
  const x = (v: number) => m.l + (v / max) * (W - m.l - m.r);
  const [hover, setHover] = useState<number | null>(null);
  const h = hover === null ? null : d.segments[hover];

  return (
    <figure className={styles.figure}>
      <div className={styles.plot} ref={plotRef}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Seconds per cycle to gain, per step">
          {!narrow && <line x1={m.l} x2={m.l} y1={m.t} y2={H - m.b} className={styles.axis} />}
          {d.segments.map((s, i) => {
            const rowTop = m.t + i * rowH;
            const cy = narrow ? rowTop + 34 : rowTop + rowH / 2;
            const w = Math.max(6, x(s.potential_seconds) - m.l);
            const bh = i === 0 ? 24 : 18;
            return (
              <g
                key={s.label}
                className={hover !== null && hover !== i ? styles.dim : ''}
                tabIndex={0}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              >
                <rect x={0} y={rowTop} width={W} height={rowH} className={styles.hit} />
                <text
                  x={narrow ? 0 : m.l - 12}
                  y={narrow ? rowTop + 15 : cy + 4}
                  textAnchor={narrow ? 'start' : 'end'}
                  className={i === 0 ? styles.rowLabelStrong : styles.rowLabel}
                >
                  {s.label}
                </text>
                <path
                  d={`M${m.l},${cy - bh / 2} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} Z`}
                  className={i === 0 ? styles.barTop : styles.bar}
                />
                <text x={m.l + w + 8} y={cy + 4} className={i === 0 ? styles.valueStrong : styles.valueLabel}>
                  {fmt(s.potential_seconds, 2)} s
                </text>
              </g>
            );
          })}
          <text x={m.l} y={H - 6} className={styles.axisLabel}>
            {narrow ? 'seconds per cycle to gain' : 'seconds per cycle to gain (median − best 10% of cycles)'}
          </text>
        </svg>
        {h && (
          <Tooltip x={(m.l / W) * 100} y={((m.t + ((hover ?? 0) + 1) * rowH) / H) * 100}>
            <strong>{fmt(h.potential_seconds, 2)} s per cycle</strong>
            <span>{h.label}</span>
            <span>
              median {fmt(h.median_seconds, 2)} s, best 10% {fmt(h.p10_seconds, 2)} s
            </span>
          </Tooltip>
        )}
      </div>
      <DataTable
        summary="Data"
        head={['Step', 'To gain (s)', 'Median (s)', 'Best 10% (s)']}
        rows={d.segments.map((s) => [
          s.label,
          fmt(s.potential_seconds, 2),
          fmt(s.median_seconds, 2),
          fmt(s.p10_seconds, 2),
        ])}
      />
    </figure>
  );
}

/* ------------------------------------------------- quality: the drift -- */

export function DriftChart(): ReactNode {
  const d = data.spc;
  const [plotRef, W] = usePlotWidth(600);
  const H = 280;
  const m = {l: 40, r: 76, t: 34, b: 26};
  const t0 = time(d.points[0][0] as string);
  const t1 = time(d.points[d.points.length - 1][0] as string);
  const values = d.points.map((p) => p[1] as number);
  const lo = Math.floor(Math.min(...values, d.lcl) - 0.5);
  const hi = Math.ceil(Math.max(...values, d.ucl) + 0.5);
  const x = (iso: string) => m.l + ((time(iso) - t0) / (t1 - t0)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const pathOf = (pts: (string | number)[][]) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0] as string).toFixed(1)},${y(p[1] as number).toFixed(1)}`).join('');
  const alarmX = x(d.first_alarm);
  const outX = x(d.first_out);
  const [hover, setHover] = useState<number | null>(null);
  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const frac = (((e.clientX - box.left) / box.width) * W - m.l) / (W - m.l - m.r);
    setHover(Math.round(Math.min(1, Math.max(0, frac)) * (d.smooth.length - 1)));
  };
  const hp = hover === null ? null : d.smooth[hover];
  const ticks = Array.from({length: 9}, (_, i) => new Date(t0 + i * 30 * 60e3).toISOString().slice(0, 19)).filter(
    (iso) => time(iso) <= t1,
  );

  return (
    <figure className={styles.figure}>
      <ul className={styles.legend}>
        <li>
          <svg width="18" height="12" aria-hidden="true">
            <line x1="1" y1="6" x2="17" y2="6" className={styles.mLine} />
          </svg>
          Rolling average
        </li>
        <li>
          <svg width="18" height="12" aria-hidden="true">
            <line x1="1" y1="6" x2="17" y2="6" className={styles.mRaw} />
          </svg>
          Readings
        </li>
        <li>
          <svg width="18" height="12" aria-hidden="true">
            <line x1="1" y1="6" x2="17" y2="6" className={styles.mLimit} />
          </svg>
          Control limits (3σ)
        </li>
      </ul>
      <div className={styles.plot} ref={plotRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`} width={W} height={H}
          role="img"
          aria-label="Oven temperature drifting up; the drift alarm fires before readings leave the control limits"
        >
          <rect x={alarmX} y={m.t} width={outX - alarmX} height={H - m.t - m.b} className={styles.leadBand} />
          {[d.lcl, d.ucl].map((v, i) => (
            <g key={v}>
              <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} className={styles.limit} />
              <text x={W - m.r + 6} y={y(v) + 4} className={styles.axisLabel}>
                {i ? 'UCL' : 'LCL'} {fmt(v)}
              </text>
            </g>
          ))}
          <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} className={styles.axis} />
          {Array.from({length: Math.floor((hi - lo) / 2) + 1}, (_, i) => lo + i * 2).map((v) => (
            <text key={v} x={m.l - 8} y={y(v) + 4} textAnchor="end" className={styles.axisLabel}>
              {v}
            </text>
          ))}
          {ticks.map((iso) => (
            <text key={iso} x={x(iso)} y={H - 8} textAnchor="middle" className={styles.axisLabel}>
              {hhmm(iso)}
            </text>
          ))}
          <path d={pathOf(d.points)} className={styles.raw} />
          <path d={pathOf(d.smooth)} className={styles.line} />
          <line x1={alarmX} x2={alarmX} y1={m.t - 10} y2={H - m.b} className={styles.alarm} />
          <text x={alarmX + 6} y={m.t - 16} className={styles.calloutStrong}>
            Drift alarm {hhmm(d.first_alarm)}
          </text>
          <line x1={outX} x2={outX} y1={m.t - 10} y2={H - m.b} className={styles.callout} />
          <text x={outX + 6} y={m.t - 16} className={styles.calloutText}>
            first reading over the limit {hhmm(d.first_out)}
          </text>
          <text x={(alarmX + outX) / 2} y={H - m.b - 10} textAnchor="middle" className={styles.calloutStrong}>
            {d.lead_minutes} min warning
          </text>
          {hp && (
            <g>
              <line
                x1={x(hp[0] as string)}
                x2={x(hp[0] as string)}
                y1={m.t}
                y2={H - m.b}
                className={styles.crosshair}
              />
              <circle cx={x(hp[0] as string)} cy={y(hp[1] as number)} r={4} className={styles.hoverDot} />
            </g>
          )}
          <rect
            x={m.l}
            y={m.t}
            width={W - m.l - m.r}
            height={H - m.t - m.b}
            className={styles.hitCross}
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
        {hp && (
          <Tooltip x={(x(hp[0] as string) / W) * 100} y={16}>
            <strong>{fmt(hp[1] as number, 2)} °C</strong>
            <span>rolling average at {hhmm(hp[0] as string)}</span>
          </Tooltip>
        )}
      </div>
      <DataTable
        summary="Data"
        head={['', 'Value']}
        rows={[
          ['Drift alarm (run rule: 9 readings in a row on one side of the mean)', hhmm(d.first_alarm)],
          ['First reading beyond 3σ', hhmm(d.first_out)],
          ['Warning time', `${d.lead_minutes} min`],
          ['Readings beyond 3σ', String(d.out_of_limits.length)],
          ['Control limits (LCL · mean · UCL)', `${fmt(d.lcl)} · ${fmt(d.mean)} · ${fmt(d.ucl)}`],
        ]}
      />
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
