import { useEffect, useState } from 'react'

/**
 * A single animated horizontal bar for one class probability.
 *
 * The bar grows from 0 → actual width on mount (via CSS transition),
 * with a staggered delay for a cascade effect.
 */
export default function ConfidenceBar({ label, probability, color, isTop, delay = 0 }) {
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const timer = setTimeout(() => setWidth(probability * 100), delay + 50)
    return () => clearTimeout(timer)
  }, [probability, delay])

  const pct = (probability * 100).toFixed(1)

  return (
    <div className={isTop ? 'opacity-100' : 'opacity-60 hover:opacity-80 transition-opacity'}>
      <div className="flex items-center justify-between text-sm mb-1">
        <span className={isTop ? 'text-white font-medium' : 'text-slate-400'}>
          {label}
        </span>
        <span className="font-mono text-xs tabular-nums text-slate-300">
          {pct}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${width}%`,
            backgroundColor: color,
            boxShadow: isTop ? `0 0 10px ${color}50` : 'none',
          }}
        />
      </div>
    </div>
  )
}
