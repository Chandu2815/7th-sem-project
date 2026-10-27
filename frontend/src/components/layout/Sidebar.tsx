import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  LayoutDashboard,
  Sparkles,
  Database,
  Layers,
  Wrench,
  GitBranch,
  Brain,
  BarChart3,
  Image,
  Settings,
  Orbit,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const primaryNav = [
  { to: '/enhance', icon: Sparkles, label: 'Enhance IR', hint: 'Start here' },
]

const extraNav = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/datasets', icon: Database, label: 'Datasets' },
  { to: '/scenes', icon: Layers, label: 'Scenes' },
  { to: '/preprocessing', icon: Wrench, label: 'Preprocess' },
  { to: '/models', icon: GitBranch, label: 'Models' },
  { to: '/training', icon: Brain, label: 'Training' },
  { to: '/evaluation', icon: BarChart3, label: 'Evaluation' },
  { to: '/outputs', icon: Image, label: 'Outputs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

export function Sidebar() {
  const location = useLocation()

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-[200px] flex-col border-r border-border bg-surface">
      <div className="flex h-14 items-center gap-2.5 border-b border-border px-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10 shrink-0">
          <Orbit className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-semibold truncate">ThermaScope</p>
          <p className="text-[10px] text-muted truncate">IR Enhance</p>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto p-3 space-y-4">
        <div>
          <p className="px-2 mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-dim">
            Main
          </p>
          <div className="space-y-1">
            {primaryNav.map(({ to, icon: Icon, label, hint }) => {
              const active = location.pathname.startsWith(to)
              return (
                <NavLink key={to} to={to} className="relative block">
                  {active && (
                    <motion.div
                      layoutId="sidebar-active"
                      className="absolute inset-0 rounded-xl bg-primary/15"
                      transition={{ duration: 0.2 }}
                    />
                  )}
                  <div
                    className={cn(
                      'relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors',
                      active ? 'text-primary font-medium' : 'text-muted hover:text-foreground hover:bg-card'
                    )}
                  >
                    <Icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
                    <span className="truncate">{label}</span>
                    {hint && !active && (
                      <span className="ml-auto text-[10px] text-primary/80">{hint}</span>
                    )}
                  </div>
                </NavLink>
              )
            })}
          </div>
        </div>

        <div>
          <p className="px-2 mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-dim">
            Advanced
          </p>
          <div className="space-y-0.5">
            {extraNav.map(({ to, icon: Icon, label, end }) => {
              const active = end ? location.pathname === to : location.pathname.startsWith(to)
              return (
                <NavLink key={to} to={to} end={end} className="relative block">
                  <div
                    className={cn(
                      'relative flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition-colors',
                      active ? 'text-foreground bg-card' : 'text-muted hover:text-foreground hover:bg-card'
                    )}
                  >
                    <Icon className="h-[16px] w-[16px] shrink-0" strokeWidth={1.75} />
                    <span className="truncate">{label}</span>
                  </div>
                </NavLink>
              )
            })}
          </div>
        </div>
      </nav>

      <div className="border-t border-border p-3">
        <p className="text-[10px] text-muted-dim leading-relaxed px-1">
          Tip: open <span className="text-muted">Enhance IR</span> and upload one image to begin.
        </p>
      </div>
    </aside>
  )
}
