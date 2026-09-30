# agentic ai ebook rag chatbot

## about

This is a chatbot that answers questions about one specific book, the Konverge AI ebook on
[Agentic AI](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf). You ask a question in plain English
and it replies using text pulled out of that PDF, then shows you which pages it used so you can
check the answer yourself.

If the book does not cover what you asked, the chatbot says so instead of making something up.

That is the whole point of the project. A normal chatbot can talk about agentic AI in general
because it was trained on the internet. This one can only talk about what is actually written in
these 60 pages, and it will tell you when it does not know.

It was built as a take home assignment for an AI engineering internship. The brief asked for a
RAG pipeline built with LangGraph and a vector database, a simple HTTP API, and answers that are
strictly grounded in the source document. Everything here follows that brief and nothing more.

**try it live:** https://agentic-ai-rag-chatbot-tc3p.onrender.com

![the chat interface](docs/ui-chat.png)

The screenshot shows the two things that matter. The first question is answered from the ebook
with a retrieval score of 0.81. The second one, about the capital of France, scores 0.10, falls
under the relevance threshold, and is refused without the language model ever being called.

<p align="center">
  <img src="docs/ui-home.png" width="700" alt="the landing screen with sample questions to click">
</p>

On the landing screen there are six questions the book can answer and two it cannot, so you can
see both behaviours without typing anything.

---

## architecture

When someone asks a question, this is what happens:

```
react ui (frontend/, port 5173 in dev)  or  curl / any http client
  |
  |  POST /chat  {"question": "what is a multi-agent system?"}
  v
fastapi  (app/main.py, port 8000)
  |
  v
langgraph  (app/rag.py)          START
  |
  |---> retrieve ---> embed the question ---> pinecone, cosine similarity, top 4 chunks
  |                            |
  |                            +---> 4 chunks of the ebook, each with a similarity score
  v
  |---> generate ---> question + those 4 chunks ---> qwen 3.8 27b via groq
  |                            |
  |                            +---> a grounded answer                    END
  v
json: answer + confidence + sources (page, score, text)
  |
  v
the ui draws the answer, the score, and a list of the source chunks you can expand
```

The same pipeline runs once at the start to build the index:

```
pdf url --> data/Ebook-Agentic-AI.pdf --> text extracted with pymupdf
                                           |
                                           v
                            split into 113 chunks, about 1000 characters each
                            with 150 characters of overlap
                                           |
                                           v
                            embedded with minilm-l6-v2, 384 dimensions, on the cpu
                                           |
                                           v
                                   pinecone index, cosine metric
```

In development Vite forwards `/chat` and `/health` to port 8000, so the browser only ever talks
to one origin and there is no CORS configuration anywhere. In production FastAPI serves the built
React app itself, so the whole thing runs on a single port.

---

## how it works

### embeddings in plain words

An embedding is a list of 384 numbers that captures what a piece of text means. A model is
trained so that texts with similar meanings end up pointing in roughly the same direction, and
unrelated texts point in different directions. You can then compare two pieces of text just by
measuring the angle between them.

This is why keyword search is not enough. Somebody asking about "the cost of an agent" would not
match a paragraph that says "how much does an agent cost", because the words barely overlap. The
embeddings still find it, because the meanings are close.

### building the index

1. Download the PDF and pull the text off each page with PyMuPDF.
2. Slide a 1000 character window across each page, overlapping by 150 characters. That gives 113
   chunks. The overlap exists so that a sentence sitting across a chunk boundary still appears
   whole somewhere.
3. Turn every chunk into a 384 number vector using a local sentence transformers model.
4. Upload each vector to Pinecone, storing the text, the page number, and a chunk id as metadata.

### answering a question

1. Turn the question into a vector using the same model.
2. Ask Pinecone for the 4 chunks whose vectors point closest to the question's.
3. Build a prompt holding the system instructions, those 4 chunks with their page numbers, and the
   question.
4. Send that to the language model. It answers using nothing else.
5. Return the answer, the chunks, and the similarity score of the best chunk.

The important part is step 4. The model never sees the whole book. It sees four paragraphs. So it
can only build its answer out of those paragraphs, and the API can show you exactly which ones.

---

## tech stack

| part | choice | why |
|---|---|---|
| language | Python 3.12 | the brief asked for Python |
| API | FastAPI | about 60 lines, gives you `/docs` for free, easy to explain |
| orchestration | LangGraph | the brief asked for it, and two nodes with three edges keeps the whole workflow visible |
| vector database | Pinecone, serverless in `us-east-1`, cosine | the brief asked for it, and the free tier easily holds 113 vectors |
| embeddings | `all-MiniLM-L6-v2` through `fastembed`, 384 dimensions, runs locally | no API key, no cost, nothing leaves the machine, and it performs well on this book |
| language model | `qwen/qwen3.8-27b` through Groq | free tier, OpenAI compatible API, quick answers, follows instructions well |
| pdf parsing | PyMuPDF | fast, maintained, no external binaries |
| ui | React 19 with Vite | the brief allowed React. No UI framework, no component library, no styling library |
| tests | pytest with httpx | `TestClient`, so no server has to be running |

### why these two providers

Groq exposes an OpenAI compatible endpoint, which means it is the same `openai` Python package
with a different `base_url`. There is no extra dependency and no hand written HTTP calls. It is
also fast, which makes the demo feel responsive, and the free tier is enough to run the whole
project without spending anything.

`openai/gpt-oss-120b` is available on the same account if you want to try it, by setting
`GROQ_MODEL`. It is not the default because it is a reasoning model that spends part of its
output budget thinking before it replies.

Groq does not offer an embeddings endpoint, and I did not want to need a second paid provider
just to embed 113 chunks. `fastembed` runs the MiniLM model through ONNX Runtime on the CPU, so
embeddings are free, private, and need no key. The model is about 90 MB and embeds the entire
book in a few seconds.

If you would rather use a hosted embedding model, `app/embeddings.py` is the only file you need
to change. Update `EMBEDDING_DIMENSIONS` in `app/config.py` to match, then run
`python scripts/ingest.py --rebuild` so Pinecone builds the index at the new size.

LangChain is not used anywhere. The brief asked for LangGraph, LangGraph does not need LangChain,
and calling the Pinecone and Groq SDKs directly keeps the retrieval step readable.

---

## setup

```bash
git clone https://github.com/rakesh0x/appenassign.git
cd appenassign

python3 -m venv .venv
source .venv/bin/activate          # on windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then open .env and fill it in
```

You need three things in `.env`:

| variable | where to get it |
|---|---|
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) then API Keys, then create a key. The free plan is plenty |
| `PINECONE_INDEX_NAME` | any name you like. The ingestion script creates the index if it is missing. Default is `agentic-ai-ebook` |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) then Create API Key. Free, and no card needed |
| `GROQ_MODEL` | optional, defaults to `qwen/qwen3.8-27b` |

You do not need an OpenAI key. Embeddings run locally and generation goes to Groq. The `.env`
file is in `.gitignore`, so never commit real keys.

The first time you ingest, the embedding model downloads once, about 90 MB, and then gets cached.

### a note on the frontend

`frontend/dist` is committed, so you can run the whole app with nothing but Python and you never
have to touch Node. To work on the UI itself you also want:

```bash
cd frontend
npm install
```

Python 3.10 or newer and Node 18 or newer.

---

## ingest the pdf

```bash
python scripts/ingest.py
```

The script downloads the PDF into `data/` unless it is already there, reads the text of each page
with PyMuPDF, splits it into chunks of about 1000 characters with 150 characters of overlap,
embeds every chunk locally, then creates the Pinecone index if needed and uploads all 113 chunks.

You should see something like this:

```
PDF already present: .../data/Ebook-Agentic-AI.pdf
Extracted 59 pages -> 113 chunks
  uploaded 96/113
  uploaded 113/113
Done. 113 chunks are in Pinecone.
```

To start again from scratch, which also deletes the index first:

```bash
python scripts/ingest.py --rebuild
```

Run this again whenever you change the chunk size or swap the embedding model. The chat API only
reads what is already sitting in Pinecone, it never ingests anything itself.

---

## run it

You need two terminals.

Terminal one, the backend:

```bash
uvicorn app.main:app --reload
```

Terminal two, the frontend:

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173**. The badge in the top right says **Backend online** once the API
answers. If the backend is not running the badge turns red, the input box disables itself, and it
tells you what to start. It re enables itself within five seconds of the backend coming back, so
you never have to reload the page.

There are two other endpoints worth knowing:

- **http://localhost:8000/docs** has a Swagger page where you can try `POST /chat` by hand
- **http://localhost:8000/health** returns `{"status": "ok"}`

If you would rather run everything from one process, build the frontend once and then let
FastAPI serve it:

```bash
cd frontend && npm run build
cd .. && uvicorn app.main:app
```

Now **http://localhost:8000** is the entire app, UI and API together, on a single port.

---

## example requests

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

A real question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a multi-agent system and how can agents be organised?"}'
```

```json
{
  "answer": "A Multi-Agent System (MAS) is an agentic system that shines in tasks requiring diverse feedback and parallel task execution. It allows for independent agent operation, facilitating dynamic task allocation and parallel processing. Agents in these systems can be organized into two main architectures: 1. Hierarchical, where an agent operates and communicates within a particular domain of influence. 2. Peer-to-peer, where an agent can talk to any other agent.",
  "confidence": 0.8053,
  "sources": [
    { "page": 31, "score": 0.8053, "text": "Multi Agentic systems could be organized into hierarchical ..." },
    { "page": 29, "score": 0.7521, "text": "In this section, we explore Multi-Agent Systems (MAS) and ..." },
    { "page": 30, "score": 0.7104, "text": "Agentic systems can be categorized into single and multi-agent systems (MAS) ..." },
    { "page": 34, "score": 0.6888, "text": "..." }
  ]
}
```

That answer and those scores are the real output of this pipeline running against the live
deployment. The `text` values are shortened here to keep the example readable, the API returns
the whole chunk.

In python:

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What challenges arise when orchestrating complex agentic systems?"},
).json()

print(response["answer"])
print("retrieval score:", response["confidence"])
for source in response["sources"]:
    print(f"  p{source['page']} ({source['score']}): {source['text'][:100]}...")
```

---

## sample questions

Every question below was run against the deployed version. The score is the real retrieval score
and the page is the real page the best matching chunk came from.

| question | score | page | where it is in the book |
|---|---|---|---|
| What is Agentic AI? | 0.82 | 3 | chapter 1, pages 8 to 11 |
| How are AI agents different from traditional LLM applications? | 0.74 | 10 | the comparison table on pages 10 and 11 |
| What are the core pillars of an agentic AI system, from perception to execution? | 0.77 | 19 | section 2.1, pages 19 and 20 |
| What is a multi-agent system and how can agents be organised? | 0.81 | 31 | chapter 3, pages 30 and 31 |
| What challenges arise when orchestrating complex agentic systems? | 0.71 | 37 | sections 3.4 and 4.1, pages 36 and 39 |
| How should an organisation assess its readiness for Agentic AI? | 0.81 | 48 | chapter 5, pages 48 to 53 |

And questions that get refused on purpose:

| question | score | what comes back |
|---|---|---|
| What is the capital of France? | 0.10 | refused |
| Who won the 2018 FIFA World Cup? | 0.07 | refused |
| How do I bake sourdough bread? | 0.09 | refused |

---

## how it stays grounded

Two separate things stop the chatbot from inventing content.

**A threshold on the retrieval score.** The book covers one narrow subject, so questions about it
score far higher than questions about anything else. I measured this by embedding the 113 real
chunks and comparing eight questions from inside the book against eight unrelated ones:

| | best match score |
|---|---|
| questions about the book | 0.60 to 0.82 |
| unrelated questions | 0.07 to 0.18 |

`SIMILARITY_THRESHOLD = 0.40` in `app/rag.py` sits in the gap between those two groups. If the
best chunk scores under it, the graph returns

```
I couldn't find this information in the provided ebook.
```

and the language model is never called at all. No request, no cost, no chance of a confident
guess. The refused questions in the table above score between 0.07 and 0.10, which is a wide
margin under the threshold.

**A system prompt that forbids guessing.** The prompt is written out in full in `app/rag.py`. It
tells the model to use only the supplied context, to say the exact refusal sentence when the
context does not contain the answer, and not to guess or reach for outside knowledge.
`temperature=0` keeps the wording stable between runs.

On their own, the threshold stops most bad questions before they reach the model, and the prompt
catches the rest.

### what the confidence number actually means

`confidence` is simply the cosine similarity of the best matching chunk, exactly as Pinecone
reports it. It is **not** a probability that the answer is correct. A high score means these
chunks look like a good match for your question, nothing more. Read it as an indicator of how
good the retrieval was, and look at the `sources` beside it, which you can check yourself. That
is also why the interface labels it "retrieval score" rather than "confidence".

---

## project structure

```
appenassign/
├── app/                            the python backend
│   ├── __init__.py                 empty, makes app a package
│   ├── config.py                   env variables and the chunking constants
│   ├── embeddings.py               the only place text becomes a vector
│   ├── vector_store.py             pinecone: create the index, upsert chunks, search chunks
│   ├── rag.py                      the langgraph, the prompt and the threshold
│   └── main.py                     fastapi: /chat, /health, and serving the built ui
│
├── frontend/                       the react ui
│   ├── src/
│   │   ├── App.jsx                 holds the messages, the draft and the backend health
│   │   ├── api.js                  the only file that talks to the backend
│   │   ├── constants.js            the sample questions on the landing screen
│   │   ├── index.css               all the styling, plain css
│   │   └── components/
│   │       ├── Message.jsx         one bubble: answer, score, source chunks
│   │       └── Composer.jsx        the text box and the send button
│   ├── vite.config.js              dev server, and the proxy to port 8000
│   ├── index.html
│   └── dist/                       the build, committed so the ui runs without node
│
├── scripts/
│   └── ingest.py                   pdf to chunks to vectors to pinecone, run once
├── tests/
│   └── test_api.py                 six tests, two of which need a live index
├── docs/                           the screenshots used in this readme
├── data/                           where the pdf lands, git ignored
├── .env.example                    template for the environment variables
├── .gitignore
├── render.yaml                     the render service definition
├── .python-version                 pins python 3.12
├── requirements.txt
└── README.md
```

Every file has one job. If you are reading the backend, start with `app/rag.py`, that is the whole
pipeline in about 80 lines. If you are reading the frontend, start with `frontend/src/api.js`,
that is the entire network layer in 37 lines.

### about the frontend in a bit more detail

| file | lines | what it does |
|---|---|---|
| `src/App.jsx` | 163 | Keeps the messages, the draft, the loading flag and the backend health in `useState`, calls `askQuestion`, renders the transcript and the landing screen |
| `src/api.js` | 37 | The only place that calls the backend. Safe json parsing, so an empty response never crashes the page |
| `src/constants.js` | 12 | The sample questions |
| `src/components/Message.jsx` | 76 | One bubble: the answer, a colour coded score, and a collapsible list of source chunks with page numbers |
| `src/components/Composer.jsx` | 41 | Text box and send button. Enter sends, Shift and Enter makes a new line. Shows a hint when the backend is down |
| `src/index.css` | 472 | All the styling. Plain css custom properties, no framework |
| `vite.config.js` | 13 | The React plugin, port 5173, and the proxy for `/chat` and `/health` |

Data only flows one way. `App` owns the state, and `Message` and `Composer` get props and call
back through props. There is no context, no store and no global object.

The backend can be down, restarting, or not started yet, so the frontend treats it as something
that might not be there rather than assuming it always is. It checks `/health` on load and again
every five seconds. While it is offline the input box is disabled and says how to start the
backend. And `api.js` never calls `response.json()` blindly, it reads the body as text first, so
an empty response from a stopped backend produces a readable message instead of a raw
`Unexpected end of JSON input` error.

---

## design decisions

**Why this is so small.** The point of the exercise is showing that the retrieval loop is
understood, so every stage is a plain function you can read from top to bottom. No classes beyond
one `TypedDict`, no interfaces, no plugin registries, no design patterns.

- No LangChain. LangGraph is what the brief asked for and it does not need LangChain. Adding
  LangChain would mean a large dependency and it would hide the retrieval step behind
  abstractions.
- Direct calls to the Pinecone and Groq SDKs. More code than the LangChain equivalents, but the
  code you read is the code that runs.
- Local embeddings. One provider instead of two, no key to manage, and nothing leaves the
  machine. The book is small enough that a 384 dimension MiniLM model is more than enough.
- Chunking by characters, not tokens. Easier to read and easier to explain, and fine for a
  document of 78,000 characters.
- Chunking per page rather than across the whole document. It keeps the page numbers exact, so
  every source in the response is a real page you can open in the PDF. A sentence split across a
  page boundary is a small cost and an acceptable one.
- A fixed threshold instead of a reranker. In domain and out of domain scores separate cleanly on
  this book, so a reranker would add complexity and measurably nothing here.
- React for the ui and nothing else. No Tailwind, no component library, no data fetching library,
  no state manager. It is `useState`, `fetch` and some jsx, all readable in one sitting.
- The Vite dev server proxies to the backend rather than switching on permissive CORS. The browser
  stays on one origin and the API needs no CORS configuration.
- No conversation memory, no auth, no database, no streaming. None of them were asked for, and
  each would need its own paragraph of justification in an interview.
- No cache, no retries, no custom exception hierarchy. Errors become plain HTTP status codes at
  the API boundary and plain red bubbles in the UI.

### things i know are weak

- The chunk size and the threshold are tuned for this one book. Another document would need both
  re measured.
- A question whose answer is in the book but worded very differently can fall under the
  threshold and get refused. That is a false negative. Raising `TOP_K` helps at the cost of more
  noise.
- Retrieval is purely semantic. A question that hinges on one particular table row or figure can
  be missed, and keyword search alongside vector search would help there.
- The `sources` list always shows the 4 retrieved chunks, even when the answer is a refusal. That
  is useful when debugging but it does mean a refusal still returns text.
- There is no conversation memory, so a follow up like "and what about healthcare?" has no
  context. The UI keeps the conversation on screen but the API is stateless.
- Answers are not streamed. The whole reply appears after the typing indicator. Simpler than a
  streaming endpoint and good enough at this document size.
- There is no evaluation set. Answer quality is judged by reading, not measured against ground
  truth.

---

## deploy

It runs on Render as a **native Python service, with no Dockerfile**, because `frontend/dist` is
committed and FastAPI serves the UI from the same process. `render.yaml` in the repo root
describes the whole service and `.python-version` pins Python 3.12.

**Live:** https://agentic-ai-rag-chatbot-tc3p.onrender.com

To deploy your own copy:

1. Push the repo to GitHub.
2. In Render choose **New**, then **Blueprint**, then pick the repository. Render reads
   `render.yaml` from it.
3. Render asks for the two secrets. Paste them into the dashboard, they are never stored in git:
   `PINECONE_API_KEY` and `GROQ_API_KEY`. `PINECONE_INDEX_NAME` and `GROQ_MODEL` are already set
   in the blueprint.
4. Deploy. The log should end with `Application startup complete`.

`/health` is wired up as Render's health check, so the service is only marked live once the API
answers.

A few things worth knowing about the deployed service:

- **Nothing is ingested at deploy time.** The 113 chunks already live in the Pinecone index. If
  you ever change the embedding model or the chunk size, run `python scripts/ingest.py --rebuild`
  locally first.
- **It needs about 300 MB of RAM** after the first query, because the embedding model runs
  locally. Render's free tier has 512 MB, which fits with room to spare.
- **The first request is slow.** The model downloads and loads on first use, so the first answer
  can take 15 to 30 seconds. `/health` responds straight away.
- **The free tier sleeps** after about 15 minutes with no traffic, and the next request pays that
  cold start again. If you want it to stay warm for a demo or an interview, change `plan` to
  `starter` in `render.yaml`, which is $7 a month.

---

## how to explain this in an interview

**what is RAG?**

Retrieval Augmented Generation. Instead of asking a model to answer from memory, we look up the
relevant pieces of our document first and paste them into the prompt. The model then answers out
of those pieces. It lets a general model answer about one specific book, and it lets us show the
source of every claim.

**why do we need embeddings?**

Because keyword search only matches identical words. Somebody asking about "the cost of an agent"
would not match a paragraph that says "how much does an agent cost". Embeddings turn text into a
list of numbers that captures meaning, so those two are recognised as close even though the words
differ.

**why use a vector database?**

Because it can search a lot of vectors quickly. Comparing a question against 113 chunks by hand is
fine. Doing it against a whole library of documents is not. A vector database stores the vectors
and finds the nearest neighbours for you, returning the closest chunks along with their scores.

**what happens when someone asks a question?**

Four steps. FastAPI checks the question is not empty. The question gets embedded into a vector.
Pinecone returns the 4 chunks closest to that vector. Those chunks plus the question go to the
model, and we send back the answer with the chunks and the score.

**why use LangGraph?**

It makes the pipeline an explicit graph of nodes and edges rather than a chain of function calls.
Here it is `START`, `retrieve`, `generate`, `END`, which describes any RAG system at all. The value
shows up as the pipeline grows. Query rewriting, re-ranking, a self critique step, a not found
branch, each of those is a node and an edge, instead of another level of nested `if` statements.
It also makes debugging easy, because you can print the state after each node.

**how does the chatbot stay grounded?**

Three things. The model only ever sees retrieved chunks, never the raw question on its own. The
system prompt explicitly forbids outside knowledge and tells it exactly what to say when the
context is not enough. And the threshold means a low scoring question never reaches the model at
all. The sources we return let the user check the grounding themselves.

**what does the Pinecone similarity score mean?**

It is the cosine similarity between the question's vector and a chunk's vector, between minus 1
and 1, where 1 means pointing in exactly the same direction and 0 means unrelated. It measures
how semantically close two pieces of text are, not how correct the answer is. So 0.81 means this
chunk looks like a good match for the question. It is a retrieval quality signal, not a
probability of correctness.

**what happens when the answer is not in the pdf?**

Retrieval still runs, but the best score comes back low. In the `generate` node we compare it
against the threshold, and if it is below we return "I couldn't find this information in the
provided ebook" along with that low score, and we never call the model. Even if something slipped
past the threshold, the system prompt tells the model to refuse when the context is not enough.

**why did we choose that chunk size?**

1000 characters is roughly 150 to 200 words. Small enough that several chunks fit in the prompt,
large enough to hold a whole paragraph or a complete thought. The 150 character overlap means a
sentence sitting across a boundary still appears whole in one of the two neighbouring chunks. The
book is about 78,000 characters, which gives 113 chunks, a small enough index that retrieval
handles without any trouble.

**why did we choose these providers?**

Groq gives a fast, free, OpenAI compatible chat endpoint, so generation costs nothing and reuses
the standard SDK. Groq has no embeddings endpoint, and needing a second paid provider for 113
chunks did not seem worth it, so the embedding model runs locally instead. Free, private, and good
enough for an index this size.

**why React, and nothing else in the frontend?**

React was allowed by the brief, and it makes the state obvious. The message list, what is being
typed, whether a request is in flight, one `useState` each. Beyond that I used no libraries at
all, so the whole frontend is `fetch` plus jsx. `api.js` is 37 lines and its request body
matches the FastAPI model exactly.

**what are the limitations?**

The threshold and chunk size are tuned for this one book and would need re measuring elsewhere.
Retrieval is purely semantic, so a question that needs one specific table row can be missed.
There is no conversation memory, so follow up questions have no context. There is no evaluation
set, so quality is judged by reading. And the model can still be over verbose or paraphrase too
tightly. The grounding is enforced, the prose is not.

---

## license

MIT. See [LICENSE](LICENSE).