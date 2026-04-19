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
    <div className="app">
      <header className="header">
        <div className="logo">✦</div>
        <h1>FindAura</h1>
        <p className="tagline">Intelligent document discovery</p>
        {installable && (
          <button className="install-btn" onClick={handleInstall}>
            Install App
          </button>
        )}
      </header>
      <main className="main">
        <p className="placeholder">Coming soon.</p>
      </main>
    </div>
  )
}

export default App
