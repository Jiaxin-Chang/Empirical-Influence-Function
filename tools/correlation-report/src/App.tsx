import { useEffect, useState } from 'react';
import './App.css';
import { NewView } from './components/NewView';

interface AllTokensMeta {
    taskId: string;
    label: string;
    fileName: string;
}

interface Manifest {
    allTokensExperiments: AllTokensMeta[];
}

function App() {
    const [manifest, setManifest] = useState<Manifest | null>(null);

    useEffect(() => {
        fetch('/data/index.json')
            .then(r => { if (!r.ok) throw new Error(); return r.json(); })
            .then((m: Manifest) => setManifest(m))
            .catch(() => {
                setManifest({ allTokensExperiments: [] });
            });
    }, []);

    if (!manifest) {
        return (
            <div className="app-root">
                <header className="app-header">
                    <h1>Attribution Analysis</h1>
                </header>
                <section className="analysis-section">
                    <p className="no-data" style={{ marginTop: '16px' }}>Loading…</p>
                </section>
            </div>
        );
    }

    return (
        <div className="app-root">
            <header className="app-header">
                <h1>Attribution Analysis</h1>
            </header>
            <NewView metas={manifest.allTokensExperiments ?? []} />
        </div>
    );
}

export default App;
