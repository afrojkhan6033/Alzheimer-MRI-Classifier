import ConfidenceBar from './ConfidenceBar'

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

export default function ResultCard({ title, subtitle, data }) {
  if (!data) return null

  const predColor = COLORS[data.prediction] || '#94a3b8'
  const predLabel = LABELS[data.prediction] || data.prediction

  return (
    <div className="bg-slate-900/50 rounded-2xl border border-slate-800/60 p-6
      hover:border-slate-700/60 transition-colors duration-200">

      {/* header */}
      <div className="flex items-baseline justify-between mb-4">
        <h3 className="font-semibold text-slate-200">{title}</h3>
        {subtitle && (
          <span className="text-xs text-slate-500">{subtitle}</span>
        )}
      </div>

      {/* prediction + confidence */}
      <div className="mb-5">
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Prediction</p>
        <div className="flex items-center gap-2.5">
          <span className="w-3 h-3 rounded-full shrink-0"
            style={{ backgroundColor: predColor }} />
          <span className="text-xl font-bold text-white">{predLabel}</span>
        </div>
        <p className="text-2xl font-mono font-bold mt-1"
          style={{ color: predColor }}>
          {(data.confidence * 100).toFixed(1)}%
        </p>
      </div>

      {/* bars */}
      <div className="space-y-2.5">
        {Object.entries(data.probabilities).map(([cls, prob], i) => (
          <ConfidenceBar key={cls}
            label={LABELS[cls] || cls}
            probability={prob}
            color={COLORS[cls] || '#94a3b8'}
            isTop={cls === data.prediction}
            delay={i * 80}
          />
        ))}
      </div>
    </div>
  )
}
