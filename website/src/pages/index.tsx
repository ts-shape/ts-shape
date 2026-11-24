import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';
import ThemedImage from '@theme/ThemedImage';

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

const EXAMPLE = `import ts_shape

# Synthetic data in the standard long format; use a loader or pandas for your own
df = ts_shape.make_timeseries(["press:temp"], n_points=2000, n_outliers=5)

print(ts_shape.check_signals(df).summary())   # signal kinds, gaps, resets, duplicates

outliers = ts_shape.OutlierDetectionEvents(df, value_column="value_double")
events = outliers.detect_outliers_zscore(n_sigma=4.0)

log = ts_shape.to_event_log(events, detector="OutlierDetectionEvents.detect_outliers_zscore")
print(log.events[["ocel:activity", "ocel:timestamp", "ts_shape:value"]])`;

function Hero(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className="container">
        <ThemedImage
          className={styles.heroLogo}
          alt="ts-shape"
          sources={{
            light: useBaseUrl('/img/logo-light.png'),
            dark: useBaseUrl('/img/logo-dark.png'),
          }}
          width={342}
          height={96}
        />
        <p className={styles.tagline}>{siteConfig.tagline}</p>
        <div className={styles.install}>
          <CodeBlock language="bash">pip install ts-shape</CodeBlock>
        </div>
        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="/user_guide/quick_start/">
            Get started
          </Link>
          <Link className="button button--secondary button--lg" to="/reference/">
            API reference
          </Link>
        </div>
      </div>
    </header>
  );
}

function Domains(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <h2 className={styles.sectionTitle}>One DataFrame in, one DataFrame out</h2>
        <p className={styles.sectionLead}>
          Loaders, filters, feature extraction and event detectors for long-format signal data
          (<code>systime</code>, <code>uuid</code>, <code>value_*</code>).
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

function Example(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.example}>
          <div className={styles.exampleText}>
            <h2 className={styles.sectionTitle}>From raw signals to an event log</h2>
            <p>
              Profile the signals, run a detector and hand the result to process mining, in a
              few lines of pandas.
            </p>
            <ul>
              <li>
                <Link to="/concept/">Data model and layers</Link>
              </li>
              <li>
                <Link to="/modules/">Module reference</Link>: one page per detector
              </li>
              <li>
                <Link to="/examples/">Example scripts</Link>
              </li>
            </ul>
          </div>
          <div className={styles.exampleCode}>
            <CodeBlock language="python">{EXAMPLE}</CodeBlock>
          </div>
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
        <Domains />
        <Example />
      </main>
    </Layout>
  );
}
