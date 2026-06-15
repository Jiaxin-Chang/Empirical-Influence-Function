import { useEffect, useState } from 'react';
import './App.css';
import { NewView } from './components/NewView';
import { ModelCompareView, type ModelCompareMeta } from './components/ModelCompareView';

// ─── Manifest types ───────────────────────────────────────────────────────────

interface AllTokensMeta {
    taskId: string;
    label: string;
    fileName: string;
}

interface Manifest {
    allTokensExperiments: AllTokensMeta[];
    modelCompare: ModelCompareMeta | null;
}

// ─── App ──────────────────────────────────────────────────────────────────────

type ViewMode = 'new' | 'compare';

function App() {
    const [viewMode, setViewMode] = useState<ViewMode>('new');
    const [manifest, setManifest] = useState<Manifest | null>(null);
    const [manifestError, setManifestError] = useState(false);

    useEffect(() => {
        fetch('/data/index.json')
            .then(r => { if (!r.ok) throw new Error(); return r.json(); })
            .then((m: Manifest) => setManifest(m))
            .catch(() => {
                setManifest({ allTokensExperiments: [], modelCompare: null });
                setManifestError(true);
            });
    }, []);

    if (!manifest) {
        return (
            <div className="app-root">
                <header className="app-header"><h1>Attribution Analysis</h1></header>
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
                <div className="view-toggle">
                    <button
                        className={`view-toggle-btn${viewMode === 'new' ? ' active' : ''}`}
                        onClick={() => setViewMode('new')}
                    >
                        New View
                    </button>
                    {manifest.modelCompare && (
                        <button
                            className={`view-toggle-btn${viewMode === 'compare' ? ' active' : ''}`}
                            onClick={() => setViewMode('compare')}
                        >
                            Compare
                        </button>
                    )}
                </div>
            </header>

            {manifestError && (
                <div className="manifest-warning">
                    Failed to load <code>/data/index.json</code>. Bundled samples are unavailable, but JSON import still works.
                </div>
            )}

            {viewMode === 'new' && (
                <NewView metas={manifest.allTokensExperiments ?? []} />
            )}

            {viewMode === 'compare' && manifest.modelCompare && (
                <ModelCompareView meta={manifest.modelCompare} />
            )}
        </div>
    );
}

export default App;
