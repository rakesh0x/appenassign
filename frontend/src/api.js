const API_BASE = import.meta.env.VITE_API_URL ?? ''

export const SIMILARITY_THRESHOLD = 0.4

async function readJson(response) {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`)
  const data = await readJson(response)
  if (!response.ok || data?.status !== 'ok') throw new Error('Backend is not ready')
  return data
}

export async function askQuestion(question) {
  let response
  try {
    response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  } catch {
    throw new Error('Cannot reach the backend. Is uvicorn running on port 8000?')
  }

  const data = await readJson(response)
  if (!response.ok) throw new Error(data?.detail ?? `The backend returned ${response.status}`)
  if (!data) throw new Error('The backend returned an empty response.')
  return data
}