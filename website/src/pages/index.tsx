import {useEffect, useState, type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';

import {
  DriversChart,
  EnergyPerPart,
  LineTimeline,
  LoopResponses,
  SpcChart,
  exampleCode,
  exampleFacts,
} from '@site/src/components/LandingCharts';
import styles from './index.module.css';

type Domain = {title: string; to: string; description: string};

const DOMAINS: Domain[] = [
  {
    title: 'Production and OEE',
    to: '/guides/production/',
    description: 'Machine states, cycle times, downtime, part counts, OEE and shift reports.',
  },
  {
    title: 'Quality and SPC',
    to: '/guides/quality/',
    description: 'Outliers, control-chart rules, capability, drift and gauge R&R.',
  },
  {
    title: 'Process engineering',
    to: '/guides/engineering/',
    description: 'Setpoint response, steady state, process windows and control-loop health.',
  },
  {
    title: 'Maintenance',
    to: '/guides/maintenance/',
    description: 'Degradation, failure prediction, MTBF / MTTR, Weibull and vibration.',
  },
  {
    title: 'Energy',
    to: '/guides/energy/',
    description: 'Power integration, consumption baselines, EnPI, idle energy and carbon.',
  },
  {
    title: 'Event logs (OCEL / XES)',
    to: '/guides/eventlog/',
    description: 'Turn any detector output into an event log for process mining.',
  },
];

// Outcomes the green part of the headline cycles through, one per domain.
const OUTCOMES = [
  'continuous improvement.',
  'higher OEE.',
  'stable processes.',
  'less downtime.',
  'lower energy use.',
  'process mining.',
];

function RotatingOutcome(): ReactNode {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    // Reduced motion only drops the slide (see the CSS); the words still change.
    const id = window.setInterval(() => setIndex((i) => (i + 1) % OUTCOMES.length), 2800);
    return () => window.clearInterval(id);
  }, []);
  return (
    <>
      <span className={styles.srOnly}>{OUTCOMES[0]}</span>
      <span className={styles.rotator} aria-hidden="true">
        {OUTCOMES.map((outcome, i) => (
          <span
            key={outcome}
            className={
              i === index
                ? styles.rotatorActive
                : i === (index - 1 + OUTCOMES.length) % OUTCOMES.length
                  ? styles.rotatorLeaving
                  : undefined
            }>
            {outcome}
          </span>
        ))}
      </span>
    </>
  );
}

function Hero(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={`container ${styles.heroInner}`}>
        <div className={styles.heroText}>
          <p className={styles.eyebrow}>Timeseries · Manufacturing · {'Continuous\u00a0Improvement'}</p>
          <h1 className={styles.title}>
            Turn machine signals into <RotatingOutcome />
          </h1>
          <p className={styles.tagline}>{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link className="button button--primary button--lg" to="/user_guide/quick_start/">
              Get started
            </Link>
            <Link className="button button--secondary button--lg" to="/modules/">
              Browse detectors
            </Link>
          </div>
        </div>
        <div className={styles.heroInstall}>
          <CodeBlock language="bash">pip install ts-shape</CodeBlock>
          <ul className={styles.heroFacts}>
            <li>
              <strong>90+</strong> detectors for production, quality, engineering, maintenance and energy
            </li>
            <li>
              <strong>Long-format</strong> signals in (<code>systime</code>, <code>uuid</code>,{' '}
              <code>value_*</code>), plain DataFrames out
            </li>
            <li>
              <strong>OCEL 2.0 / XES</strong> event logs from any result, for process mining
            </li>
          </ul>
        </div>
      </div>
    </header>
  );
}

type ExampleProps = {
  kicker: string;
  question: string;
  answer: ReactNode;
  code: string;
  link: {to: string; label: string};
  chart: ReactNode;
};

function Example({kicker, question, answer, code, link, chart}: ExampleProps): ReactNode {
  return (
    <article className={styles.example}>
      <div className={styles.exampleText}>
        <p className={styles.kicker}>{kicker}</p>
        <h3 className={styles.question}>{question}</h3>
        <p className={styles.answer}>{answer}</p>
        <details className={styles.code}>
          <summary>The code behind this chart</summary>
          <CodeBlock language="python">{code.trimEnd()}</CodeBlock>
        </details>
        <Link to={link.to} className={styles.more}>
          {link.label} →
        </Link>
      </div>
      <div className={styles.exampleChart}>{chart}</div>
    </article>
  );
}

function Examples(): ReactNode {
  const {line, loops, spc, energy, drivers} = exampleFacts;
  const bad = loops.loops.filter((l) => l.diagnosis !== 'ok');
  const causal = drivers.signals.filter((sg) => sg.causal);
  return (
    <section className={styles.section}>
      <div className="container">
        <h2 className={styles.sectionTitle}>Questions from the plant, answered in a few lines</h2>
        <p className={styles.sectionLead}>
          Five everyday plant situations: a film extrusion line, dryer temperature loops, an extruder, an injection
          moulding machine. The data comes from ts-shape's plant simulators, tagged and timed the way a historian
          logs it; every chart is real ts-shape output, and the code is under each chart.
        </p>
        <Example
          kicker="Production · OEE"
          question="Why did film line 2 stop on Monday and Tuesday?"
          answer={
            <>
              Two days of one extrusion line, six shifts, from its run signal, part counter and alarm tags, with no
              mapping of which alarm means what: {line.stop_count} stops cost {line.lost_hours} hours and OEE was{' '}
              {line.oee_pct}%. ts-shape finds the alarm that rises before each stop: die-pressure alarms cause{' '}
              {line.causes[0].share_pct}% of the lost time ({line.causes[0].mttr_minutes} min per stop), web breaks{' '}
              {line.causes[1].share_pct}%.
            </>
          }
          code={exampleCode.line}
          link={{to: '/modules/production/oee-calculator/', label: 'OEE and machine states'}}
          chart={<LineTimeline />}
        />
        <Example
          kicker="Engineering · control loops"
          question="Which dryer temperature loops need retuning?"
          answer={
            <>
              Hand it the raw PLC tags of the dryer: it pairs setpoint, process value and controller output itself,
              measures every setpoint change in the shift and diagnoses each loop.{' '}
              {bad.map((l) => `${l.name.split(' ')[0]} ${l.diagnosis === 'oscillating' ? `overshoots by ${Math.round(l.overshoot_pct)}%` : `settles in only ${l.settled_pct}% of its setpoint changes`}`).join(', ')}.
            </>
          }
          code={exampleCode.loops}
          link={{to: '/modules/engineering/control-loop-health/', label: 'Control loop health'}}
          chart={<LoopResponses />}
        />
        <Example
          kicker="Quality · SPC"
          question="Is the extruder melt temperature drifting before parts go out of spec?"
          answer={
            <>
              Last week's in-control shift sets the control limits ({spc.lcl.toFixed(1)}–{spc.ucl.toFixed(1)} °C).
              Today the melt temperature creeps up through the shift: the Nelson run rules flag the shifted mean
              (shaded), without a break from {spc.drift_from.slice(11, 16)} on, and {spc.out_of_limits.length}{' '}
              readings end up beyond the upper limit. One call returns both, ready to alarm on.
            </>
          }
          code={exampleCode.spc}
          link={{to: '/modules/quality/spc/', label: 'Statistical process control'}}
          chart={<SpcChart />}
        />
        <Example
          kicker="Energy · kWh per part"
          question="What does a moulded part cost in energy, and when is it wasted?"
          answer={
            <>
              Injection moulding machine IMM-07 over two shifts, from its cumulative kWh meter and shot counter: a
              median of {energy.median_kwh_per_part.toFixed(2)} kWh per part, up to{' '}
              {energy.worst_kwh_per_part.toFixed(2)} in half hours with an interruption, while the machine keeps
              drawing idle power. At the best half hour's rate the {energy.parts} parts would have needed{' '}
              {Math.round(energy.excess_kwh)} kWh less.
            </>
          }
          code={exampleCode.energy}
          link={{to: '/modules/energy/efficiency-tracking/', label: 'Energy efficiency'}}
          chart={<EnergyPerPart />}
        />
        <Example
          kicker="Data science · causality"
          question="What actually drives the dryer temperature?"
          answer={
            <>
              A Granger test of every tag logged near the loop against its temperature, with a Bonferroni-corrected
              p-value: {causal.map((sg) => sg.label).join(' and ')} drive it; the look-alikes (
              {drivers.signals.filter((sg) => !sg.causal).map((sg) => sg.label.toLowerCase()).join(', ')}) do not.
              Transfer entropy and lead-lag are in the same class.
            </>
          }
          code={exampleCode.drivers}
          link={{to: '/reference/ts_shape/features/cross_signal/', label: 'Cross-signal analytics'}}
          chart={<DriversChart />}
        />
      </div>
    </section>
  );
}

function Domains(): ReactNode {
  return (
    <section className={`${styles.section} ${styles.sectionAlt}`}>
      <div className="container">
        <h2 className={styles.sectionTitle}>One DataFrame in, one DataFrame out</h2>
        <p className={styles.sectionLead}>
          Loaders, filters, feature extraction and event detectors for long-format signal data. Pick a domain:
        </p>
        <div className={styles.grid}>
          {DOMAINS.map((d) => (
            <Link key={d.title} to={d.to} className={styles.card}>
              <h3>{d.title}</h3>
              <p>{d.description}</p>
              <span className={styles.cardMore}>Read the guide →</span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Manufacturing time-series analytics for pandas" description={siteConfig.tagline}>
      <Hero />
      <main>
        <Examples />
        <Domains />
      </main>
    </Layout>
  );
}
