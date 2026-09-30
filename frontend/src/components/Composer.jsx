const OFFLINE_HINT = 'Start the backend with: uvicorn app.main:app --reload'

export default function Composer({ value, onChange, onSubmit, pending, health }) {
  const offline = health === 'offline'

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSubmit()
    }
  }

  const label = offline ? 'Offline' : pending ? 'Waiting' : 'Send'

  return (
    <>
      {offline && <p className="offline-hint">{OFFLINE_HINT}</p>}
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit()
        }}
      >
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            offline ? 'Waiting for the backend to come online…' : 'Ask a question about the Agentic AI ebook…'
          }
          rows={1}
          disabled={pending || offline}
          aria-label="Your question"
        />
        <button type="submit" disabled={pending || offline || !value.trim()}>
          {label}
        </button>
      </form>
    </>
  )
}