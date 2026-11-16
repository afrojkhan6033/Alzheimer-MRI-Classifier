import { useCallback, useRef, useState } from 'react'

export default function DropZone({ onFile, preview, hasFile, onReset }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    const f = e.dataTransfer?.files?.[0]
    if (f) onFile(f)
  }, [onFile])

  const handleChange = useCallback((e) => {
    const f = e.target.files?.[0]
    if (f) onFile(f)
  }, [onFile])

  /* ── preview state ──────────────────────────────────────────────────────── */
  if (hasFile && preview) {
    return (
      <div className="relative bg-slate-900/50 rounded-2xl border border-slate-700/60 overflow-hidden group">
        <img src={preview} alt="MRI Preview"
          className="w-full h-72 object-contain bg-black/80" />
        <button onClick={onReset}
          className="absolute top-3 right-3 p-2 rounded-full
            bg-slate-800/80 border border-slate-600
            text-slate-300 hover:text-white hover:bg-slate-700
            opacity-0 group-hover:opacity-100 transition-all duration-200 cursor-pointer"
          title="Remove image">
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    )
  }

  /* ── drop zone ──────────────────────────────────────────────────────────── */
  return (
    <div
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        h-72 rounded-2xl border-2 border-dashed cursor-pointer
        flex flex-col items-center justify-center gap-4
        transition-all duration-200 select-none
        ${dragActive
          ? 'border-teal-400 bg-teal-900/20 scale-[1.02]'
          : 'border-slate-700 bg-slate-900/30 hover:border-slate-500 hover:bg-slate-900/50'}
      `}>
      <input ref={inputRef} type="file" accept="image/*"
        onChange={handleChange} className="hidden" />

      <svg className={`h-14 w-14 transition-colors duration-200
            ${dragActive ? 'text-teal-400' : 'text-slate-600'}`}
        fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>

      <div className="text-center">
        <p className={`text-lg font-medium transition-colors
              ${dragActive ? 'text-teal-300' : 'text-slate-400'}`}>
          {dragActive ? 'Drop your MRI scan here' : 'Drag & drop an MRI scan'}
        </p>
        <p className="text-sm text-slate-500 mt-1">
          or click to browse · JPEG, PNG, BMP, TIFF
        </p>
      </div>
    </div>
  )
}
