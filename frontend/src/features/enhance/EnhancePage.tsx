import { useCallback, useRef, useState } from 'react'
import {
  Upload,
  Download,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Circle,
  RefreshCw,
  ArrowRight,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { FadeIn } from '@/components/shared/PageTransition'
import { enhanceInfraredUpload, type IrEnhanceResult } from '@/api'
import { cn } from '@/lib/utils'

const HOW_TO = [
  { n: 1, title: 'Upload', desc: 'Choose a satellite IR photo (JPG/PNG) or a Landsat .TIF band.' },
  { n: 2, title: 'Enhance', desc: 'Click Start enhancement and wait for it to finish.' },
  { n: 3, title: 'Compare & download', desc: 'View before/after and save your images.' },
]

function canBrowserPreview(file: File): boolean {
  const name = file.name.toLowerCase()
  if (name.endsWith('.tif') || name.endsWith('.tiff')) return false
  return file.type.startsWith('image/') || /\.(jpe?g|png|webp|gif|bmp)$/i.test(name)
}

function withCacheBust(url: string, jobId: string): string {
  if (!url) return url
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}t=${jobId}`
}

export function EnhancePage() {
  const inputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewNote, setPreviewNote] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [statusText, setStatusText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<IrEnhanceResult | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [compareMode, setCompareMode] = useState<'split' | 'grid'>('split')
  const [detectObjects, setDetectObjects] = useState(false)

  const onSelect = useCallback((f: File | null) => {
    setFile(f)
    setResult(null)
    setError(null)
    setProgress(0)
    setStatusText('')
    setPreviewNote(null)
    setPreviewUrl(null)

    if (!f) return

    if (!canBrowserPreview(f)) {
      setPreviewNote('Preview not shown for TIFF in the browser — processing still works. Prefer JPG/PNG to see a preview.')
      return
    }

    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') setPreviewUrl(reader.result)
    }
    reader.onerror = () => {
      setPreviewNote('Could not build preview, but you can still run enhancement.')
    }
    reader.readAsDataURL(f)
  }, [])

  async function run() {
    if (!file || running) return
    setRunning(true)
    setError(null)
    setResult(null)
    setProgress(5)
    setStatusText('Sending image to the server…')

    const messages = [
      'Reading image…',
      'Resizing / preprocessing…',
      'Enhancing infrared…',
      'Colorizing…',
      'Finishing up…',
    ]
    let i = 0
    const tick = window.setInterval(() => {
      setProgress((p) => Math.min(p + 6, 90))
      setStatusText(messages[Math.min(i, messages.length - 1)])
      i += 1
    }, 700)

    try {
      const data = await enhanceInfraredUpload(file, 'SatelliteIR', detectObjects)
      window.clearInterval(tick)
      setResult(data)
      setProgress(100)
      setStatusText('Done — results are below')
      window.setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    } catch (e) {
      window.clearInterval(tick)
      setError(e instanceof Error ? e.message : 'Something went wrong. Is the API running on port 8000?')
      setStatusText('')
    } finally {
      setRunning(false)
    }
  }

  function reset() {
    onSelect(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const step = result ? 3 : file ? 2 : 1
  const jobId = result?.job_id ?? ''

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <FadeIn>
        <div className="rounded-2xl border border-border bg-card p-6 sm:p-8">
          <p className="text-[11px] font-medium uppercase tracking-widest text-primary mb-2">
            How to use this app
          </p>
          <h2 className="text-2xl font-semibold tracking-tight">Enhance an infrared image</h2>
          <p className="mt-2 text-sm text-muted max-w-xl">
            Follow the three steps. Upload infrared / Landsat imagery only.
            Normal color photos are detected and <strong className="text-foreground">rejected</strong> (not processed).
          </p>

          <ol className="mt-6 grid gap-3 sm:grid-cols-3">
            {HOW_TO.map((item) => (
              <li
                key={item.n}
                className={cn(
                  'rounded-xl border p-4 transition-colors',
                  step === item.n
                    ? 'border-primary/40 bg-primary/5'
                    : step > item.n
                      ? 'border-success/30 bg-success/5'
                      : 'border-border bg-surface'
                )}
              >
                <div className="flex items-center gap-2 mb-2">
                  {step > item.n ? (
                    <CheckCircle2 className="h-4 w-4 text-success" />
                  ) : step === item.n ? (
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-[11px] font-bold text-white">
                      {item.n}
                    </span>
                  ) : (
                    <Circle className="h-4 w-4 text-muted-dim" />
                  )}
                  <span className="text-sm font-semibold">{item.title}</span>
                </div>
                <p className="text-xs text-muted leading-relaxed">{item.desc}</p>
              </li>
            ))}
          </ol>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <Card
          className={cn('border-2 border-dashed', dragOver && 'border-primary bg-primary/5')}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            const f = e.dataTransfer.files?.[0]
            if (f) onSelect(f)
          }}
        >
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
              <div className="flex-1 space-y-4">
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-white">
                    1
                  </span>
                  <h3 className="text-base font-semibold">Select your infrared image</h3>
                </div>
                <p className="text-sm text-muted">
                  Drag and drop here, or click <strong className="text-foreground">Browse files</strong>.
                </p>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/jpg,.jpg,.jpeg,.png,.tif,.tiff"
                  className="hidden"
                  onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
                />
                <div className="flex flex-wrap gap-2">
                  <Button type="button" onClick={() => inputRef.current?.click()} variant="secondary">
                    <Upload className="h-4 w-4" />
                    Browse files
                  </Button>
                  {file && (
                    <Button type="button" variant="ghost" onClick={reset}>
                      <RefreshCw className="h-4 w-4" />
                      Clear
                    </Button>
                  )}
                </div>
                {file && (
                  <p className="text-sm text-foreground">
                    Selected: <span className="font-mono text-primary break-all">{file.name}</span>
                    <span className="text-muted"> ({(file.size / 1024).toFixed(0)} KB)</span>
                  </p>
                )}
                {previewNote && (
                  <p className="text-xs text-warning flex gap-2 items-start">
                    <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                    {previewNote}
                  </p>
                )}
              </div>

              <div className="w-full lg:w-64 shrink-0">
                <div className="aspect-square rounded-xl border border-border bg-black/60 flex items-center justify-center overflow-hidden">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Upload preview"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <div className="text-center p-4">
                      <Upload className="h-8 w-8 text-muted-dim mx-auto mb-2" />
                      <p className="text-xs text-muted">
                        {file ? 'No browser preview' : 'Preview appears here'}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="mt-8 pt-6 border-t border-border">
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-white">
                  2
                </span>
                <h3 className="text-base font-semibold">Run enhancement</h3>
              </div>
              <label className="flex items-center gap-2 text-sm text-muted mb-4 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={detectObjects}
                  onChange={(e) => setDetectObjects(e.target.checked)}
                  className="rounded border-border"
                />
                Also run object detection (optional, needs YOLO weights)
              </label>
              <Button
                type="button"
                size="lg"
                className="w-full sm:w-auto min-w-[220px] h-12 text-base"
                disabled={!file || running}
                onClick={run}
              >
                {running ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Working… please wait
                  </>
                ) : (
                  <>
                    Start enhancement
                    <ArrowRight className="h-5 w-5" />
                  </>
                )}
              </Button>
              {!file && (
                <p className="text-xs text-muted-dim mt-2">Upload an image in step 1 first.</p>
              )}
            </div>

            {running && (
              <div className="mt-6 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted">{statusText}</span>
                  <span className="font-mono text-xs">{progress}%</span>
                </div>
                <Progress value={progress} />
              </div>
            )}

            {error && (
              <div className="mt-6 flex gap-3 rounded-xl border border-error/30 bg-error/10 p-4 text-sm text-error">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium">Could not process the image</p>
                  <p className="mt-1 whitespace-pre-wrap opacity-90">{error}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </FadeIn>

      {result && (
        <div ref={resultsRef}>
          <FadeIn delay={0.08}>
            <div className="space-y-6">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-success text-xs font-bold text-white">
                    3
                  </span>
                  <div>
                    <h3 className="text-base font-semibold">Your results</h3>
                    <p className="text-xs text-muted">Compare before and after, then download.</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant={compareMode === 'split' ? 'default' : 'secondary'}
                    onClick={() => setCompareMode('split')}
                  >
                    Before / After
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={compareMode === 'grid' ? 'default' : 'secondary'}
                    onClick={() => setCompareMode('grid')}
                  >
                    All 4 views
                  </Button>
                </div>
              </div>

              {result.warnings?.length ? (
                <div className="flex gap-3 rounded-xl border border-primary/25 bg-primary/10 p-4 text-sm text-foreground">
                  <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-primary" />
                  <div>
                    <p className="font-medium text-primary mb-1">Note</p>
                    <ul className="space-y-1 text-muted">
                      {result.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : null}

              {compareMode === 'split' ? (
                <div className="grid gap-4 sm:grid-cols-2">
                  <ResultPanel title="Before (original)" url={withCacheBust(result.artifacts.original, jobId)} />
                  <ResultPanel title="After (colorized)" url={withCacheBust(result.artifacts.colorized, jobId)} highlight />
                </div>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2">
                  <ResultPanel title="1. Original" url={withCacheBust(result.artifacts.original, jobId)} />
                  <ResultPanel title="2. Preprocessed" url={withCacheBust(result.artifacts.preprocessed, jobId)} />
                  <ResultPanel title="3. Enhanced infrared" url={withCacheBust(result.artifacts.enhanced, jobId)} />
                  <ResultPanel title="4. Colorized" url={withCacheBust(result.artifacts.colorized, jobId)} highlight />
                </div>
              )}

              <div className="rounded-xl border border-border bg-card p-5">
                <p className="text-sm font-semibold mb-3">Download</p>
                <div className="flex flex-wrap gap-2">
                  <a href={result.downloads.enhanced} download>
                    <Button type="button">
                      <Download className="h-4 w-4" />
                      Enhanced infrared
                    </Button>
                  </a>
                  <a href={result.downloads.colorized} download>
                    <Button type="button">
                      <Download className="h-4 w-4" />
                      Colorized image
                    </Button>
                  </a>
                  <a href={result.downloads.comparison} download>
                    <Button type="button" variant="secondary">
                      <Download className="h-4 w-4" />
                      Comparison sheet
                    </Button>
                  </a>
                </div>
              </div>

              <div className="rounded-xl border border-border bg-card p-5 space-y-4">
                <div>
                  <p className="text-sm font-semibold">Objects found</p>
                  <p className="text-xs text-muted mt-1">
                    {detectObjects
                      ? 'From the optional detection step.'
                      : 'Turn on “Also run object detection” in step 2 to enable this.'}
                  </p>
                </div>
                {result.artifacts.annotated && (
                  <img
                    src={withCacheBust(result.artifacts.annotated, jobId)}
                    alt="Objects marked on image"
                    className="w-full max-h-[420px] object-contain rounded-lg bg-black border border-border"
                  />
                )}
                {result.detections.length === 0 ? (
                  <p className="text-sm text-muted">
                    {result.detect_message || 'No objects detected.'}
                  </p>
                ) : (
                  <ul className="space-y-2">
                    {result.detections.map((d, i) => (
                      <li
                        key={i}
                        className="flex items-center justify-between rounded-lg border border-border bg-surface px-4 py-3 text-sm"
                      >
                        <span className="font-medium capitalize">{d.label}</span>
                        <span className="font-mono text-primary">
                          {(d.confidence * 100).toFixed(0)}% sure
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <Button type="button" variant="secondary" onClick={reset}>
                <RefreshCw className="h-4 w-4" />
                Process another image
              </Button>
            </div>
          </FadeIn>
        </div>
      )}
    </div>
  )
}

function ResultPanel({
  title,
  url,
  highlight,
}: {
  title: string
  url: string
  highlight?: boolean
}) {
  const [failed, setFailed] = useState(false)

  return (
    <div
      className={cn(
        'overflow-hidden rounded-xl border bg-card',
        highlight ? 'border-primary/40 ring-1 ring-primary/20' : 'border-border'
      )}
    >
      <div className="px-4 py-2.5 border-b border-border text-sm font-medium">{title}</div>
      {failed ? (
        <div className="aspect-square flex items-center justify-center text-xs text-muted bg-black p-4 text-center">
          Image failed to load. Check that the API is running, then try again.
        </div>
      ) : (
        <img
          src={url}
          alt={title}
          className="w-full aspect-square object-contain bg-black"
          onError={() => setFailed(true)}
        />
      )}
    </div>
  )
}
