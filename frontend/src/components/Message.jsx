import { SIMILARITY_THRESHOLD } from '../api.js'

function confidenceTone(confidence) {
  if (confidence < SIMILARITY_THRESHOLD) return 'low'
  if (confidence < 0.6) return 'mid'
  return 'high'
}

export default function Message({ message }) {
  const { role, answer, confidence, sources, error } = message
  const isUser = role === 'user'

  return (
    <article className={`message ${isUser ? 'from-user' : 'from-bot'}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? 'You' : 'AI'}
      </div>

      <div className="bubble">
        {isUser ? (
          <p className="text">{answer}</p>
        ) : error ? (
          <p className="text error-text">{error}</p>
        ) : (
          <>
            <p className="text">
              <InlineText text={answer} />
            </p>
            <div className={`score score-${confidenceTone(confidence)}`}>
              <span className="score-label">Retrieval score</span>
              <span className="score-value">{confidence.toFixed(2)}</span>
              {confidence < SIMILARITY_THRESHOLD && (
                <span className="score-note">
                  below the {SIMILARITY_THRESHOLD.toFixed(2)} threshold, nothing relevant found
                </span>
              )}
            </div>
            <Sources sources={sources} />
          </>
        )}
      </div>
    </article>
  )
}

function InlineText({ text }) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith('**') && part.endsWith('**') ? (
      <strong key={index}>{part.slice(2, -2)}</strong>
    ) : (
      part
    ),
  )
}

function Sources({ sources }) {
  if (!sources?.length) return null

  return (
    <details className="sources">
      <summary>
        {sources.length} source chunk{sources.length === 1 ? '' : 's'} from the ebook
      </summary>
      <ul>
        {sources.map((source, index) => (
          <li key={source.page + index}>
            <div className="source-head">
              <span className="source-page">Page {source.page}</span>
              <span className="source-score">{source.score.toFixed(2)}</span>
            </div>
            <p>{source.text}</p>
          </li>
        ))}
      </ul>
    </details>
  )
}