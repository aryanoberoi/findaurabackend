import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [deferredPrompt, setDeferredPrompt] = useState(null)
  const [installable, setInstallable] = useState(false)

  useEffect(() => {
    const handler = (e) => {
      e.preventDefault()
      setDeferredPrompt(e)
      setInstallable(true)
    }
    window.addEventListener('beforeinstallprompt', handler)
    return () => window.removeEventListener('beforeinstallprompt', handler)
  }, [])

  const handleInstall = async () => {
    if (!deferredPrompt) return
    deferredPrompt.prompt()
    const { outcome } = await deferredPrompt.userChoice
    if (outcome === 'accepted') setInstallable(false)
    setDeferredPrompt(null)
  }

  return (
    <div className="page">
      <nav className="nav">
        <div className="nav-brand">
          <span className="nav-logo">✦</span>
          <span className="nav-name">FindAura</span>
        </div>
        {installable && (
          <button className="btn btn-outline" onClick={handleInstall}>
            Install App
          </button>
        )}
      </nav>

      <section className="hero">
        <div className="hero-inner">
          <div className="badge">Now in beta</div>
          <h1 className="hero-title">
            Discover the aura<br />of your documents
          </h1>
          <p className="hero-sub">
            FindAura lets you upload any document and instantly search,
            summarize, and chat with your content — powered by AI.
          </p>
          <div className="hero-actions">
            <button className="btn btn-primary">Get started free</button>
            <button className="btn btn-ghost">See how it works</button>
          </div>
        </div>
        <div className="hero-glow" />
      </section>

      <section className="features">
        <div className="features-inner">
          <h2 className="section-title">Everything you need</h2>
          <div className="grid">
            <div className="card">
              <div className="card-icon">📄</div>
              <h3>Upload any format</h3>
              <p>PDF, DOCX, TXT and more. Drag, drop, done.</p>
            </div>
            <div className="card">
              <div className="card-icon">🔍</div>
              <h3>Semantic search</h3>
              <p>Find exactly what you mean, not just what you typed.</p>
            </div>
            <div className="card">
              <div className="card-icon">💬</div>
              <h3>Chat with docs</h3>
              <p>Ask questions and get answers grounded in your files.</p>
            </div>
            <div className="card">
              <div className="card-icon">🌐</div>
              <h3>Multi-language</h3>
              <p>Works across 23+ languages including regional Indian languages.</p>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <span className="nav-logo">✦</span>
        <span>© {new Date().getFullYear()} FindAura. All rights reserved.</span>
      </footer>
    </div>
  )
}

export default App
