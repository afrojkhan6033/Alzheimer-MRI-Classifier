import { useState, useCallback } from 'react'
import DropZone from './components/DropZone'
import ResultCard from './components/ResultCard'
import ConfidenceBar from './components/ConfidenceBar'

export default function App() {
  const [file, setFile]                   = useState(null)
  const [preview, setPreview]             = useState(null)
  const [results, setResults]             = useState(null)
  const [loading, setLoading]             = useState(false)
  const [gradcamLoading, setGradcamLoading] = useState(false)
  const [error, setError]                 = useState(null)

  /* ── file handling ─────────────────────────────────────────────────────── */
  const handleFile = useCallback((f) => {
    setFile(f)
    setResults(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const reset = useCallback(() => {
    setFile(null)
    setPreview(null)
    setResults(null)
    setError(null)
  }, [])

  /* ── API call ──────────────────────────────────────────────────────────── */
  const callPredict = useCallback(async (withGradcam = false) => {
    if (!file) return
    const setter = withGradcam ? setGradcamLoading : setLoading
    setter(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`/api/predict?gradcam=${withGradcam}`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      setResults(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setter(false)
    }
  }, [file])

  const handlePredict = useCallback(() => callPredict(false), [callPredict])
  const handleGradcam = useCallback(() => callPredict(true),  [callPredict])

  /* ── render ────────────────────────────────────────────────────────────── */
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* ─── Header ──────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800/60 bg-slate-900/70 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center gap-3">
          <span className="text-3xl select-none">🧠</span>
          <div>
            <h1 className="text-xl font-bold gradient-text">
              Alzheimer MRI Classifier
            </h1>
            <p className="text-xs sm:text-sm text-slate-400">
              Dual-model inference · ResNet-50 &amp; VGG-16 · Ensemble prediction
            </p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        {/* ─── Upload row ────────────────────────────────────────────────── */}
        <section className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* Left: upload + button */}
          <div className="lg:col-span-2 space-y-4">
            <DropZone onFile={handleFile} preview={preview} hasFile={!!file} onReset={reset} />

            {file && !loading && (
              <button onClick={handlePredict}
                className="w-full py-3 px-6 rounded-xl font-semibold text-white
                  bg-gradient-to-r from-teal-600 to-blue-600
                  hover:from-teal-500 hover:to-blue-500
                  active:scale-[0.98] transition-all duration-200
                  shadow-lg shadow-teal-900/30 cursor-pointer">
                Analyze MRI Scan
              </button>
            )}

            {loading && (
              <div className="w-full py-3 text-center text-slate-400 flex items-center justify-center gap-2">
                <Spinner /> Running inference on both models…
              </div>
            )}
          </div>

          {/* Right: placeholder or quick summary */}
          <div className="lg:col-span-3">
            {!results && (
              <div className="h-full min-h-[320px] bg-slate-900/40 rounded-2xl border border-slate-800/60
                flex flex-col items-center justify-center text-slate-500 p-8">
                <svg className="h-16 w-16 mb-4 opacity-25" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <p className="text-lg font-medium">Upload an MRI scan to begin</p>
                <p className="text-sm mt-1">Both models will analyze the image simultaneously</p>
              </div>
            )}

            {/* Quick ensemble summary (appears instantly with results) */}
            {results && (
              <div className="animate-slide-up bg-gradient-to-br from-slate-900 to-slate-900/60
                rounded-2xl border border-slate-700/50 p-8">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium text-slate-400 uppercase tracking-wider">Ensemble Prediction</span>
                  {results.agreement ? (
                    <span className="ml-auto text-xs px-2.5 py-0.5 rounded-full bg-emerald-900/40 border border-emerald-700/50 text-emerald-400 flex items-center gap-1">
                      <CheckIcon /> Models Agree
                    </span>
                  ) : (
                    <span className="ml-auto text-xs px-2.5 py-0.5 rounded-full bg-amber-900/40 border border-amber-700/50 text-amber-400 flex items-center gap-1">
                      <WarnIcon /> Models Disagree
                    </span>
                  )}
                </div>

                <div className="flex items-baseline gap-4 mb-6">
                  <span className="inline-block w-4 h-4 rounded-full mt-1"
                    style={{ backgroundColor: classColor(results.ensemble.prediction) }} />
                  <span className="text-3xl font-bold text-white">
                    {classLabel(results.ensemble.prediction)}
                  </span>
                  <span className="text-3xl font-mono font-bold"
                    style={{ color: classColor(results.ensemble.prediction) }}>
                    {(results.ensemble.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="space-y-3">
                  {Object.entries(results.ensemble.probabilities).map(([cls, prob], i) => (
                    <ConfidenceBar key={cls}
                      label={classLabel(cls)} probability={prob}
                      color={classColor(cls)}
                      isTop={cls === results.ensemble.prediction}
                      delay={i * 80} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ─── Error ─────────────────────────────────────────────────────── */}
        {error && (
          <div className="p-4 rounded-xl bg-red-900/30 border border-red-800/60 text-red-300 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* ─── Warnings ──────────────────────────────────────────────────── */}
        {results?.warnings?.length > 0 && (
          <div className="p-4 rounded-xl bg-amber-900/25 border border-amber-800/50 text-amber-200 text-sm space-y-1 animate-fade-in">
            <p className="font-semibold text-amber-300">⚠ Advisory Warnings</p>
            {results.warnings.map((w, i) => <p key={i}>{w}</p>)}
          </div>
        )}

        {/* ─── Per-model result cards ────────────────────────────────────── */}
        {results && (
          <section className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-slide-up"
            style={{ animationDelay: '0.15s', animationFillMode: 'both' }}>
            <ResultCard title="ResNet-50" subtitle="25 M params · 90 MB" data={results.resnet50} />
            <ResultCard title="VGG-16"   subtitle="134 M params · 512 MB" data={results.vgg16} />
          </section>
        )}

        {/* ─── Grad-CAM section ──────────────────────────────────────────── */}
        {results && (
          <section className="animate-slide-up bg-slate-900/40 rounded-2xl border border-slate-800/60 p-6"
            style={{ animationDelay: '0.25s', animationFillMode: 'both' }}>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
              <div>
                <h3 className="text-lg font-semibold">Grad-CAM Explainability</h3>
                <p className="text-sm text-slate-400">
                  See which brain regions each model focuses on
                </p>
              </div>
              <button onClick={handleGradcam} disabled={gradcamLoading}
                className="px-5 py-2 rounded-lg text-sm font-medium
                  bg-slate-800 border border-slate-700
                  hover:bg-slate-700 hover:border-slate-600
                  disabled:opacity-50 disabled:cursor-not-allowed
                  transition-all duration-200 cursor-pointer flex items-center gap-2">
                {gradcamLoading
                  ? <><Spinner small /> Generating…</>
                  : results.gradcam ? '↻ Regenerate Heatmaps' : '🔬 Generate Heatmaps'}
              </button>
            </div>

            {results.gradcam ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 animate-fade-in">
                {[['resnet50', 'ResNet-50'], ['vgg16', 'VGG-16']].map(([key, label]) => (
                  <div key={key} className="bg-slate-800/50 rounded-xl p-4">
                    <p className="text-sm font-medium text-slate-300 mb-2">{label}</p>
                    <img
                      src={`data:image/png;base64,${results.gradcam[key]}`}
                      alt={`${label} Grad-CAM`}
                      className="w-full rounded-lg"
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-10 text-slate-600 text-sm">
                Click "Generate Heatmaps" to visualise model attention
              </div>
            )}
          </section>
        )}
      </main>

      {/* ─── Footer ──────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800/40 mt-12 py-6 text-center text-xs text-slate-600">
        Alzheimer MRI Classifier · ResNet-50 &amp; VGG-16 · For research purposes only · Not a medical device
      </footer>
    </div>
  )
}


/* ── tiny inline helper components ─────────────────────────────────────────── */

function Spinner({ small = false }) {
  const sz = small ? 'h-4 w-4' : 'h-5 w-5'
  return (
    <svg className={`animate-spin ${sz} text-teal-400`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 20 20">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
    </svg>
  )
}

function WarnIcon() {
  return (
    <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 20 20">
      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
    </svg>
  )
}

/* ── class-colour lookup ───────────────────────────────────────────────────── */

const COLORS = {
  Non_Demented:       '#10b981',
  Very_Mild_Demented: '#eab308',
  Mild_Demented:      '#f97316',
  Moderate_Demented:  '#ef4444',
}
const LABELS = {
  Non_Demented:       'Non Demented',
  Very_Mild_Demented: 'Very Mild',
  Mild_Demented:      'Mild',
  Moderate_Demented:  'Moderate',
}

function classColor(cls) { return COLORS[cls] || '#94a3b8' }
function classLabel(cls) { return LABELS[cls] || cls }
