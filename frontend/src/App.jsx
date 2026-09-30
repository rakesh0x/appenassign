import { useEffect, useRef, useState } from 'react'
import { askQuestion, checkHealth, SIMILARITY_THRESHOLD } from './api.js'
import { OUT_OF_SCOPE_QUESTIONS, SAMPLE_QUESTIONS } from './constants.js'
import Message from './components/Message.jsx'
import Composer from './components/Composer.jsx'

const HEALTH_POLL_MS = 5000

export default function App() {
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState(false)
  const [health, setHealth] = useState('checking')
  const logRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        await checkHealth()
        if (!cancelled) setHealth('online')
      } catch {
        if (!cancelled) setHealth('offline')
      }
    }

    poll()
    const timer = setInterval(poll, HEALTH_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    const log = logRef.current
    if (log) log.scrollTo({ top: log.scrollHeight, behavior: 'smooth' })
  }, [messages])

  async function send(question) {
    const text = question.trim()
    if (!text || pending) return

    setDraft('')
    setPending(true)
    setMessages((current) => [...current, { role: 'user', answer: text }])

    try {
      const data = await askQuestion(text)
      setMessages((current) => [
        ...current,
        { role: 'bot', answer: data.answer, confidence: data.confidence, sources: data.sources },
      ])
    } catch (error) {
      setMessages((current) => [...current, { role: 'bot', error: error.message }])
    } finally {
      setPending(false)
    }
  }

  const empty = messages.length === 0

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>Agentic AI Ebook</h1>
          <p>RAG chatbot grounded in the Konverge AI ebook</p>
        </div>
        <div className="header-right">
          <span className={`health health-${health}`}>
            <span className="dot" />
            {health === 'checking'
              ? 'Connecting'
              : health === 'online'
                ? 'Backend online'
                : 'Backend offline'}
          </span>
          {!empty && (
            <button className="ghost" onClick={() => setMessages([])}>
              Clear
            </button>
          )}
        </div>
      </header>

      <main className="log" ref={logRef}>
        <div className="log-inner">
          {empty && <Intro onPick={send} pending={pending} />}

          {messages.map((message, index) => (
            <Message key={index} message={message} />
          ))}

          {pending && <Thinking />}
        </div>
      </main>

      <Composer
        value={draft}
        onChange={setDraft}
        onSubmit={() => send(draft)}
        pending={pending}
        health={health}
      />

      <footer className="footer">
        Answers come only from the ebook. Anything else is refused below a similarity of{' '}
        {SIMILARITY_THRESHOLD.toFixed(2)}.
      </footer>
    </div>
  )
}

function Thinking() {
  return (
    <article className="message from-bot">
      <div className="avatar" aria-hidden="true">
        AI
      </div>
      <div className="bubble thinking">
        <span />
        <span />
        <span />
      </div>
    </article>
  )
}

function Intro({ onPick, pending }) {
  return (
    <section className="intro">
      <h2>Ask the ebook</h2>
      <p>
        Every answer is generated from chunks retrieved out of the PDF. You will see the page numbers
        it came from, and questions the ebook cannot answer are refused.
      </p>

      <h3>Answerable from the ebook</h3>
      <div className="chips">
        {SAMPLE_QUESTIONS.map((question) => (
          <button key={question} className="chip" disabled={pending} onClick={() => onPick(question)}>
            {question}
          </button>
        ))}
      </div>

      <h3>Should be refused</h3>
      <div className="chips">
        {OUT_OF_SCOPE_QUESTIONS.map((question) => (
          <button
            key={question}
            className="chip chip-muted"
            disabled={pending}
            onClick={() => onPick(question)}
          >
            {question}
          </button>
        ))}
      </div>
    </section>
  )
}