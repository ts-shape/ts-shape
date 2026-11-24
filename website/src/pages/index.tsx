import type {ReactNode} from 'react';
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

function Hero(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={`container ${styles.heroInner}`}>
        <div className={styles.heroText}>
          <p className={styles.eyebrow}>Python · pandas · open source</p>
          <h1 className={styles.title}>
            Answers from your machine data, <span>not just plots.</span>
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
          Every chart below is real ts-shape output on its simulated plant data; open the code to see the calls
          that produced it.
        </p>
        <Example
          kicker="Production · OEE"
          question="Where did the day go?"
          answer={
            <>
              From one boolean run signal and a part counter: {line.stop_count} stops cost{' '}
              {Math.round(line.stopped_minutes)} minutes, the longest {Math.round(line.longest_stop_minutes)}.
              Availability {line.availability_pct}% and performance {line.performance_pct}% give an OEE of{' '}
              {line.oee_pct}%.
            </>
          }
          code={exampleCode.line}
          link={{to: '/modules/production/oee-calculator/', label: 'OEE and machine states'}}
          chart={<LineTimeline />}
        />
        <Example
          kicker="Engineering · control loops"
          question="Which control loops need retuning?"
          answer={
            <>
              Hand it the raw PLC tags: it pairs setpoint, process value and output itself, measures every
              setpoint step and diagnoses each loop. {bad.map((l) => `${l.name} ${l.diagnosis === 'oscillating' ? `overshoots by ${Math.round(l.overshoot_pct)}%` : `settles in only ${l.settled_pct}% of its steps`}`).join(', ')}.
            </>
          }
          code={exampleCode.loops}
          link={{to: '/modules/engineering/control-loop-health/', label: 'Control loop health'}}
          chart={<LoopResponses />}
        />
        <Example
          kicker="Quality · SPC"
          question="Is the oven drifting before parts go out of spec?"
          answer={
            <>
              Control limits come from an in-control reference run. As the temperature creeps up, the Nelson run
              rules flag the drift ({spc.run_rules.length} readings, shaded), and {spc.out_of_limits.length}{' '}
              readings cross the 3σ limit. One call returns both, ready to alarm on.
            </>
          }
          code={exampleCode.spc}
          link={{to: '/modules/quality/spc/', label: 'Statistical process control'}}
          chart={<SpcChart />}
        />
        <Example
          kicker="Energy · kWh per part"
          question="What does a part cost in energy, and when is it wasted?"
          answer={
            <>
              A cumulative kWh meter and the part counter give the specific energy per half hour: a median of{' '}
              {energy.median_kwh_per_part.toFixed(2)} kWh per part, up to {energy.worst_kwh_per_part.toFixed(2)} when
              the station waits while drawing idle power. Running every half hour at the best rate would save{' '}
              {Math.round(energy.excess_kwh)} kWh this shift.
            </>
          }
          code={exampleCode.energy}
          link={{to: '/modules/energy/efficiency-tracking/', label: 'Energy efficiency'}}
          chart={<EnergyPerPart />}
        />
        <Example
          kicker="Data science · causality"
          question="Which signals actually drive the process temperature?"
          answer={
            <>
              A Granger test of every signal against the temperature, with a Bonferroni-corrected p-value:{' '}
              {causal.map((sg) => sg.label).join(' and ')} drive it; the look-alikes (
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
