import { useEffect, useRef, useState } from 'react'
import './App.css'

const LOADING_STEPS = [
  'Detecting document edges…',
  'Cropping & correcting perspective…',
  'Running OCR on the image…',
  'Extracting text…',
  'Categorising content…',
  'Almost done…',
]

const WORKFLOW_STEPS = [
  'Detect and crop only the document from a wide-angle image.',
  'Extract readable OCR text from the cleaned document area.',
  'Output text as editable content you can copy or refine.',
  'Categorize content into clear topic headings for any document type.',
]

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [editableText, setEditableText] = useState('')
  const [aiRetrying, setAiRetrying] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)

  useEffect(() => {
    if (!loading && !aiRetrying) { setLoadingStep(0); return }
    const id = setInterval(() => {
      setLoadingStep(s => Math.min(s + 1, LOADING_STEPS.length - 1))
    }, 2200)
    return () => clearInterval(id)
  }, [loading, aiRetrying])
  const MAX_UPLOAD_MB = 10
  const API_BASE = import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:8000`
  const cameraInputRef = useRef(null)
  const galleryInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (!selectedFile) return

    const sizeMb = selectedFile.size / (1024 * 1024)
    if (sizeMb > MAX_UPLOAD_MB) {
      setError(`File too large (${sizeMb.toFixed(1)} MB). Please use an image under ${MAX_UPLOAD_MB} MB.`)
      setFile(null)
      return
    }

    setError(null)
    setFile(selectedFile)
    // Clear previous result so the "Retry with AI" button doesn't carry over
    // to the newly selected file.
    setResult(null)
    setEditableText('')
  }

  const handleScan = async (forceVision = false) => {
    if (!file) {
      alert("Please select an image first!")
      return
    }

    if (forceVision) setAiRetrying(true)
    else setLoading(true)
    setError(null)
    if (!forceVision) { setResult(null); setEditableText('') }

    const formData = new FormData()
    formData.append('file', file)
    if (forceVision) formData.append('force_vision', 'true')
    let timeoutId

    try {
      const controller = new AbortController()
      timeoutId = setTimeout(() => controller.abort(), 60000)
      const response = await fetch(`${API_BASE}/api/scan`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })

      if (!response.ok) {
        let detail = `Server error: ${response.status}`
        try {
          const errData = await response.json()
          if (errData?.detail) detail = errData.detail
        } catch {
          // keep default message if backend didn't return JSON
        }
        throw new Error(detail)
      }

      const data = await response.json()
      setResult(data)
      setEditableText(data?.results?.editable_text || '')
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Try a smaller or clearer image.')
      } else {
        setError(err.message || "Failed to connect to the server.")
      }
    } finally {
      if (timeoutId) clearTimeout(timeoutId)
      setLoading(false)
      setAiRetrying(false)
    }
  }

  return (
    <div className="app-container">

      {(loading || aiRetrying) && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <div className="loading-steps">
            <p className="loading-title">
              {aiRetrying ? 'Asking AI Vision…' : 'Processing Document'}
            </p>
            <p key={loadingStep} className="loading-step">
              {LOADING_STEPS[loadingStep]}
            </p>
          </div>
        </div>
      )}

      <header>
        <h1>📄 OCR Document Categorizer</h1>
        <p>Crop documents, extract editable OCR text, and categorize into topic headings.</p>
      </header>

      <section className="workflow-card">
        <h2>How It Works</h2>
        <ul>
          {WORKFLOW_STEPS.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ul>
      </section>
      
      <div className="upload-card">
        <div className="source-row">
          <button
            type="button"
            className="secondary-btn"
            disabled={loading}
            onClick={() => cameraInputRef.current?.click()}
          >
            Take Photo (Camera)
          </button>
          <button
            type="button"
            className="secondary-btn"
            disabled={loading}
            onClick={() => galleryInputRef.current?.click()}
          >
            Choose from Gallery
          </button>
        </div>

        <input
          ref={cameraInputRef}
          type="file"
          onChange={handleFileChange}
          accept="image/*"
          capture="environment"
          className="file-input-hidden"
        />
        <input
          ref={galleryInputRef}
          type="file"
          onChange={handleFileChange}
          accept="image/*"
          className="file-input-hidden"
        />

        {file && (
          <div className="selected-file">
            Selected: <span className="selected-file-name">{file.name}</span>
          </div>
        )}
        <button 
          onClick={() => handleScan()}
          disabled={loading || !file}
          className={loading ? "scanning-btn" : "scan-btn"}
        >
          {loading ? 'Scanning Document...' : 'Scan Document'}
        </button>
      </div>

      {error && <div className="error-box">⚠️ {error}</div>}

  {result && result.results && (
    <div className="results-card">
      <h2>Document Output</h2>

      {result.scan_id && result.artifacts?.cropped_image && (
        <div className="result-group">
          <span className="label">Cropped Preview:</span>
          <img
            className="preview-image"
            src={`${API_BASE}${result.artifacts.cropped_image}?t=${Date.now()}`}
            alt="Cropped document preview"
          />
        </div>
      )}
      
      <div className="result-group">
        <span className="label">Category:</span>
        <span className="value badge">{result.results.category || "N/A"}</span>
      </div>

      <div className="result-group">
        <span className="label">Subcategory:</span>
        <span className="value badge">{result.results.subcategory || "N/A"}</span>
      </div>
      
      <div className="result-group">
        <span className="label">Summary:</span>
        <p className="value summary-text">{result.results.summary || "No summary provided."}</p>
      </div>

      <div className="result-group">
        <span className="label">Editable Extracted Text:</span>
        <textarea
          className="editable-textarea"
          value={editableText}
          onChange={(e) => setEditableText(e.target.value)}
          rows={10}
          placeholder="Extracted OCR text will appear here..."
        />
      </div>
      
      <div className="result-group">
        <span className="label">Desired Topic Headings:</span>
        <ul className="key-info-list">
          {result.results.key_information?.map((info, index) => (
            <li key={index}>{info}</li>
          )) || <li>No headings extracted.</li>}
        </ul>
      </div>

      <button 
        className="copy-btn"
        onClick={() => {
          const text = editableText || ""
          if (text) {
            navigator.clipboard.writeText(text);
            alert("Extracted text copied to clipboard!");
          }
        }}
      >
        Copy Extracted Text
      </button>

      {/* AI Vision button — at the bottom so it's never accidentally tapped */}
      <div className={
        (result.results.subcategory === 'UnreadableContent' || result.results._classification_method === 'fallback_quota')
          ? 'ai-retry-box ai-retry-box--warn'
          : 'ai-retry-box ai-retry-box--subtle'
      }>
        {(result.results.subcategory === 'UnreadableContent' || result.results._classification_method === 'fallback_quota')
          ? <p>⚠️ Document could not be read clearly. AI Vision can extract text directly from the image.</p>
          : <p>Not satisfied? AI Vision re-reads the image for better accuracy.</p>
        }
        <button
          className="ai-retry-btn"
          disabled={aiRetrying}
          onClick={() => handleScan(true)}
        >
          {aiRetrying ? '⏳ Asking AI Vision...' : '✨ Improve with AI Vision'}
        </button>
      </div>

      {result.scan_id && result.artifacts && (
        <div className="download-row">
          {result.artifacts.ocr_text && (
            <a className="download-link" href={`${API_BASE}${result.artifacts.ocr_text}`} target="_blank" rel="noreferrer">
              Download OCR Text
            </a>
          )}
          {result.artifacts.result_json && (
            <a className="download-link" href={`${API_BASE}${result.artifacts.result_json}`} target="_blank" rel="noreferrer">
              Download JSON
            </a>
          )}
        </div>
      )}

      {result.meta && (
        <div className="result-group meta-group">
          <span className="label">Processing Metadata:</span>
          <p className="meta-text">
            Document detected: {result.meta.document_detected ? 'Yes' : 'No'} | OCR words: {result.meta.ocr_word_count} | Avg confidence: {result.meta.ocr_avg_confidence} | Method: {result.meta.classification_method}
          </p>
          {result.meta.durations_ms && (
            <p className="meta-text">
              Timing (ms): preprocess {result.meta.durations_ms.preprocess}, ocr {result.meta.durations_ms.ocr}, categorize {result.meta.durations_ms.categorize}, total {result.meta.durations_ms.total}
            </p>
          )}
        </div>
      )}
    </div>
  )}
    </div>
  )
}

export default App