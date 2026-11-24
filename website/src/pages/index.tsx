import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';

import {
  CyclePotential,
  LineTimeline,
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
  const {spc, line, cycles} = exampleFacts;
  return (
    <section className={styles.section}>
      <div className="container">
        <h2 className={styles.sectionTitle}>Questions from the shop floor, answered in a few lines</h2>
        <p className={styles.sectionLead}>
          Every chart below is real ts-shape output on its simulated plant data; open the code to see
          the calls that produced it.
        </p>
        <Example
          kicker="Quality · SPC"
          question="Is the oven drifting before parts go out of spec?"
          answer={
            <>
              Control limits come from an in-control reference run. As the temperature creeps up, the
              Nelson run rules flag the drift from {spc.first_run_rule.slice(11, 16)} on ({spc.run_rules.length}{' '}
              readings, shaded), and {spc.out_of_limits.length} readings cross the 3σ limit. One call returns
              both, ready to alarm on or to turn into an event log.
            </>
          }
          code={exampleCode.spc}
          link={{to: '/modules/quality/spc/', label: 'Statistical process control'}}
          chart={<SpcChart />}
        />
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
          kicker="Cycle time · profiling"
          question="Which step should we fix first to shorten the cycle?"
          answer={
            <>
              Give it every raw PLC signal, no mapping: it finds the cycle, learns the steps and ranks them by
              the time a step already achieves in its best 10% of cycles. Variant {cycles.variant} could run{' '}
              {cycles.throughput_gain_pct}% faster; the top step alone is worth{' '}
              {cycles.segments[0].potential_seconds.toFixed(2)} s per cycle.
            </>
          }
          code={exampleCode.cycles}
          link={{to: '/modules/profiling/cycle-profiling/', label: 'Cycle time profiling'}}
          chart={<CyclePotential />}
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
