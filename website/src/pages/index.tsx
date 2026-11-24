import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';

import {
  CycleChart,
  DriftChart,
  ProductionChart,
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
        <div>
          <p className={styles.eyebrow}>Python · pandas · open source</p>
          <h1 className={styles.title}>
            Answers from your machine data, <span>not just plots.</span>
          </h1>
          <p className={styles.tagline}>{siteConfig.tagline}</p>
        </div>
        <div className={styles.heroSide}>
          <CodeBlock language="bash">pip install ts-shape</CodeBlock>
          <div className={styles.buttons}>
            <Link className="button button--primary" to="/user_guide/quick_start/">
              Get started
            </Link>
            <Link className="button button--secondary" to="/modules/">
              Browse 90+ detectors
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

type ExampleProps = {
  kicker: string;
  headline: ReactNode;
  text: ReactNode;
  stat: {value: string; label: string};
  code: string;
  link: {to: string; label: string};
  chart: ReactNode;
};

function Example({kicker, headline, text, stat, code, link, chart}: ExampleProps): ReactNode {
  return (
    <article className={styles.example}>
      <div className={styles.exampleHead}>
        <div>
          <p className={styles.kicker}>{kicker}</p>
          <h3 className={styles.headline}>{headline}</h3>
          <p className={styles.text}>{text}</p>
        </div>
        <div className={styles.stat}>
          <strong>{stat.value}</strong>
          <span>{stat.label}</span>
        </div>
      </div>
      <div className={styles.exampleChart}>{chart}</div>
      <div className={styles.exampleFoot}>
        <details className={styles.code}>
          <summary>The code behind this chart</summary>
          <CodeBlock language="python">{code.trimEnd()}</CodeBlock>
        </details>
        <Link to={link.to} className={styles.more}>
          {link.label} →
        </Link>
      </div>
    </article>
  );
}

function Examples(): ReactNode {
  const {spc, line, cycles} = exampleFacts;
  const top = cycles.segments[0];
  return (
    <section className={styles.section}>
      <div className="container">
        <h2 className={styles.sectionTitle}>What the plant data says, in a few lines of Python</h2>
        <p className={styles.sectionLead}>
          Real ts-shape output on its simulated plant data. Open the code under each chart to see the calls.
        </p>
        <Example
          kicker="Production · OEE"
          headline={
            <>
              {line.stopped_hours} h lost to stops, {line.slow_hours} h to running slow
            </>
          }
          text={
            <>
              From one run signal and a part counter: of {line.planned_hours} planned hours,{' '}
              {line.productive_hours} were productive. {line.parts.toLocaleString('en-US')} parts made,{' '}
              {line.parts_possible.toLocaleString('en-US')} possible at ideal speed.
            </>
          }
          stat={{value: `${line.oee_pct}%`, label: 'OEE'}}
          code={exampleCode.line}
          link={{to: '/modules/production/oee-calculator/', label: 'OEE and machine states'}}
          chart={<ProductionChart />}
        />
        <Example
          kicker="Cycle time · profiling"
          headline={
            <>
              Fix {top.label} first: {top.potential_seconds.toFixed(2)} s per cycle
            </>
          }
          text={
            <>
              Raw PLC signals in, no mapping: ts-shape finds the cycle, learns its steps and compares each step
              with its own best 10% of {cycles.cycle_count} cycles.
            </>
          }
          stat={{value: `+${cycles.throughput_gain_pct}%`, label: 'throughput within reach'}}
          code={exampleCode.cycles}
          link={{to: '/modules/profiling/cycle-profiling/', label: 'Cycle time profiling'}}
          chart={<CycleChart />}
        />
        <Example
          kicker="Quality · SPC"
          headline={
            <>
              Drift caught at {spc.first_alarm.slice(11, 16)}, before the first bad reading
            </>
          }
          text={
            <>
              Control limits from an in-control reference run; the run rules flag the creeping oven temperature
              long before a reading crosses 3σ.
            </>
          }
          stat={{value: `${spc.lead_minutes} min`, label: 'early warning'}}
          code={exampleCode.spc}
          link={{to: '/modules/quality/spc/', label: 'Statistical process control'}}
          chart={<DriftChart />}
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
