import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const handleScan = async () => {
    if (!file) {
      alert("Please select an image first!")
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Connecting to your FastAPI local server
      const response = await fetch('http://127.0.0.1:8000/api/scan', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || "Failed to connect to the server.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <h1>📄 AI Document Scanner</h1>
        <p>Upload an image to extract context and text</p>
      </header>
      
      <div className="upload-card">
        <input 
          type="file" 
          onChange={handleFileChange} 
          accept="image/*" 
          capture="environment"  // This triggers the rear camera on mobile!
          className="file-input"
        />
        <button 
          onClick={handleScan} 
          disabled={loading || !file}
          className={loading ? "scanning-btn" : "scan-btn"}
        >
          {loading ? 'Scanning Document...' : 'Scan Document'}
        </button>
      </div>

      {error && <div className="error-box">⚠️ {error}</div>}

  {result && result.results && (
    <div className="results-card">
      <h2>Scan Results</h2>
      
      <div className="result-group">
        <span className="label">Category:</span>
        <span className="value badge">{result.results.category || "N/A"}</span>
      </div>
      
      <div className="result-group">
        <span className="label">Summary:</span>
        <p className="value summary-text">{result.results.summary || "No summary provided."}</p>
      </div>
      
      <div className="result-group">
        <span className="label">Key Information Found:</span>
        <ul className="key-info-list">
          {/* The ?. prevents the crash if key_information is missing */}
          {result.results.key_information?.map((info, index) => (
            <li key={index}>{info}</li>
          )) || <li>No specific data extracted.</li>}
        </ul>
      </div>

      <button 
        className="copy-btn"
        onClick={() => {
          // Safe check for the join operation
          const text = result.results.key_information?.join('\n') || "";
          if (text) {
            navigator.clipboard.writeText(text);
            alert("Copied to clipboard!");
          }
        }}
      >
        Copy Info to Clipboard
      </button>
    </div>
  )}
    </div>
  )
}

export default App