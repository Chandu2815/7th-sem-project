import type {
  Scene,
  SceneFilters,
  DashboardStats,
  ActivityItem,
  SystemHealth,
  PreprocessingStep,
  TrainingMetrics,
  EvaluationMetrics,
  OutputItem,
  ExportHistoryItem,
  AppSettings,
  PipelineNodeDetail,
} from '@/types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

async function withMockFallback<T>(primary: () => Promise<T>, mock: () => Promise<T>): Promise<T> {
  try {
    return await primary()
  } catch {
    return mock()
  }
}

export async function fetchScenes(filters?: Partial<SceneFilters>): Promise<Scene[]> {
  return withMockFallback(
    async () => {
      const params = new URLSearchParams()
      if (filters?.search) params.set('search', filters.search)
      if (filters?.satellite) params.set('satellite', filters.satellite)
      if (filters?.status) params.set('status', filters.status)
      if (filters?.cloudMax !== undefined) params.set('cloudMax', String(filters.cloudMax))
      const q = params.toString()
      return apiFetch<Scene[]>(`/scenes${q ? `?${q}` : ''}`)
    },
    async () => {
      const { scenes } = await import('@/data/mockData')
      let result = [...scenes]
      if (filters?.search) {
        const q = filters.search.toLowerCase()
        result = result.filter(
          (s) => s.sceneId.toLowerCase().includes(q) || s.location.toLowerCase().includes(q)
        )
      }
      if (filters?.satellite && filters.satellite !== 'all') {
        result = result.filter((s) => s.satellite === filters.satellite)
      }
      if (filters?.status && filters.status !== 'all') {
        result = result.filter((s) => s.status === filters.status)
      }
      if (filters?.cloudMax !== undefined && filters.cloudMax < 100) {
        result = result.filter((s) => s.cloudCover <= filters.cloudMax!)
      }
      return result
    }
  )
}

export async function fetchScene(id: string): Promise<Scene | null> {
  return withMockFallback(
    async () => {
      try {
        return await apiFetch<Scene>(`/scenes/${encodeURIComponent(id)}`)
      } catch {
        return null
      }
    },
    async () => {
      const { scenes } = await import('@/data/mockData')
      return scenes.find((s) => s.id === id) ?? null
    }
  )
}

export async function fetchDashboard(): Promise<{
  stats: DashboardStats
  activities: ActivityItem[]
  health: SystemHealth[]
}> {
  return withMockFallback(
    () => apiFetch('/dashboard'),
    async () => {
      const { dashboardStats, activities, systemHealth } = await import('@/data/mockData')
      return { stats: dashboardStats, activities, health: systemHealth }
    }
  )
}

export async function fetchPreprocessingSteps(sceneId?: string): Promise<PreprocessingStep[]> {
  return withMockFallback(
    async () => {
      if (!sceneId) {
        const scenes = await apiFetch<Scene[]>('/scenes')
        sceneId = scenes[0]?.sceneId
      }
      if (!sceneId) throw new Error('no scene')
      const data = await apiFetch<{ steps: PreprocessingStep[] }>(
        `/preprocessing/${encodeURIComponent(sceneId)}`
      )
      return data.steps
    },
    async () => {
      const { preprocessingSteps } = await import('@/data/mockData')
      return preprocessingSteps
    }
  )
}

export async function fetchPreprocessingLogs(sceneId?: string): Promise<string[]> {
  return withMockFallback(
    async () => {
      if (!sceneId) {
        const scenes = await apiFetch<Scene[]>('/scenes')
        sceneId = scenes[0]?.sceneId
      }
      if (!sceneId) throw new Error('no scene')
      const data = await apiFetch<{ logs: string[] }>(
        `/preprocessing/${encodeURIComponent(sceneId)}`
      )
      return data.logs
    },
    async () => {
      const { preprocessingLogs } = await import('@/data/mockData')
      return preprocessingLogs
    }
  )
}

export async function fetchTrainingMetrics(): Promise<TrainingMetrics> {
  return withMockFallback(
    () => apiFetch('/training'),
    async () => {
      const { trainingMetrics } = await import('@/data/mockData')
      return trainingMetrics
    }
  )
}

export async function fetchEvaluation(sceneId?: string): Promise<EvaluationMetrics[]> {
  return withMockFallback(
    async () => {
      const q = sceneId ? `?scene_id=${encodeURIComponent(sceneId)}` : ''
      return apiFetch(`/evaluation${q}`)
    },
    async () => {
      const { evaluationMetrics } = await import('@/data/mockData')
      if (sceneId) return evaluationMetrics.filter((m) => m.sceneId === sceneId)
      return evaluationMetrics
    }
  )
}

export async function fetchOutputs(): Promise<OutputItem[]> {
  return withMockFallback(
    () => apiFetch('/outputs'),
    async () => {
      const { outputs } = await import('@/data/mockData')
      return outputs
    }
  )
}

export async function fetchExportHistory(): Promise<ExportHistoryItem[]> {
  return withMockFallback(
    () => apiFetch('/exports'),
    async () => {
      const { exportHistory } = await import('@/data/mockData')
      return exportHistory
    }
  )
}

export async function fetchSettings(): Promise<AppSettings> {
  return withMockFallback(
    () => apiFetch('/settings'),
    async () => {
      const { defaultSettings } = await import('@/data/mockData')
      return { ...defaultSettings }
    }
  )
}

export async function fetchPipelineNodes(): Promise<PipelineNodeDetail[]> {
  return withMockFallback(
    () => apiFetch('/pipeline/nodes'),
    async () => {
      const { pipelineNodes } = await import('@/data/mockData')
      return pipelineNodes
    }
  )
}

export async function runPipeline(sceneId: string, scale = 4): Promise<unknown> {
  return apiFetch('/pipeline/run', {
    method: 'POST',
    body: JSON.stringify({ scene_id: sceneId, scale, sentinel_available: false }),
  })
}

export async function login(username: string, role: 'user' | 'admin' = 'user'): Promise<unknown> {
  return apiFetch('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, role }),
  })
}

export async function detectObjects(sceneId: string): Promise<unknown> {
  return apiFetch('/detect', {
    method: 'POST',
    body: JSON.stringify({ scene_id: sceneId }),
  })
}

export async function importExternalDataset(_file: File): Promise<{ success: boolean; message: string }> {
  return { success: true, message: 'Place GeoTIFF bands under dataset/raw/<scene_id>/ then refresh' }
}

export interface IrDetection {
  label: string
  confidence: number
  bbox: number[]
}

export interface IrEnhanceResult {
  job_id: string
  status: string
  artifacts: {
    original: string
    preprocessed: string
    enhanced: string
    colorized: string
    comparison: string
    annotated?: string | null
  }
  downloads: {
    enhanced: string
    colorized: string
    comparison: string
  }
  detections: IrDetection[]
  warnings: string[]
  detect_message?: string
  meta?: Record<string, unknown>
}

export async function enhanceInfraredUpload(
  file: File,
  sourceKind: 'SatelliteIR' | 'CCTVIR' = 'SatelliteIR',
  runDetection = true,
): Promise<IrEnhanceResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('source_kind', sourceKind)
  form.append('run_detection', String(runDetection))

  const res = await fetch(`${API_BASE}/ir/enhance`, {
    method: 'POST',
    body: form,
  })

  const raw = await res.text()
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    if (raw) {
      try {
        const body = JSON.parse(raw) as { detail?: unknown }
        if (typeof body.detail === 'string') detail = body.detail
        else if (body.detail != null) detail = JSON.stringify(body.detail)
        else detail = raw
      } catch {
        detail = raw.slice(0, 300)
      }
    }
    throw new Error(detail)
  }

  try {
    return JSON.parse(raw) as IrEnhanceResult
  } catch {
    throw new Error('Server returned an invalid response. Is the API running on port 8000?')
  }
}
