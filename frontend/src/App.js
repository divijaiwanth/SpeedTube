import './App.css';
import { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000";

// ---------------------------------------------------------------------------
// API helper functions
// Plain fetch calls — no library needed
// ---------------------------------------------------------------------------

const api = {
  ingest: async (url) => {
    const res = await fetch(`${API_BASE}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to load video");
    }
    return res.json();
  },

  query: async (sessionId, question) => {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Query failed");
    }
    return res.json();
  },

  summary: async (sessionId, summaryType = "concise") => {
    const res = await fetch(
      `${API_BASE}/summary?session_id=${sessionId}&summary_type=${summaryType}`
    );
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Summary failed");
    }
    return res.json();
  },
};

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function LoadingDots() {
  return (
    <span className="loading-dots">
      <span>.</span><span>.</span><span>.</span>
    </span>
  );
}

function VideoLoader({ onLoaded }) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLoad = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError("");
    try {
      const data = await api.ingest(url);
      onLoaded(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="video-loader">
      <div className="loader-inner">
        <h1 className="logo">SpeedTube</h1>
        <p className="tagline">Ask anything about any YouTube video</p>
        <div className="input-row">
          <input
            type="text"
            placeholder="Paste a YouTube URL..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            disabled={loading}
            className="url-input"
          />
          <button
            onClick={handleLoad}
            disabled={loading || !url.trim()}
            className="btn-primary"
          >
            {loading ? <LoadingDots /> : "Load"}
          </button>
        </div>
        {error && <p className="error-msg">{error}</p>}
      </div>
    </div>
  );
}

function ChatMessage({ msg }) {
  return (
    <div className={`message ${msg.role}`}>
      <div className="message-content">
        <p>{msg.content}</p>
        {msg.sources && msg.sources.length > 0 && (
          <details className="sources">
            <summary>{msg.sources.length} source chunk{msg.sources.length > 1 ? "s" : ""}</summary>
            {msg.sources.map((src, i) => (
              <p key={i} className="source-item">
                <span className="source-num">{i + 1}</span> {src}
              </p>
            ))}
          </details>
        )}
      </div>
    </div>
  );
}

function ChatPanel({ sessionId, videoMeta }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("chat");
  const [summary, setSummary] = useState("");
  const [summaryType, setSummaryType] = useState("concise");
  const [summaryLoading, setSummaryLoading] = useState(false);
  const bottomRef = useRef(null);

  // Auto scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setInput("");

    // Add user message immediately
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);

    try {
      const data = await api.query(sessionId, question);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${e.message}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const fetchSummary = async () => {
    setSummaryLoading(true);
    setSummary("");
    try {
      const data = await api.summary(sessionId, summaryType);
      setSummary(data.summary);
    } catch (e) {
      setSummary(`Error: ${e.message}`);
    } finally {
      setSummaryLoading(false);
    }
  };

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="panel-header">
        <div className="video-info">
          <span className="video-id">video: {videoMeta.video_id}</span>
          <span className="chunk-count">{videoMeta.chunk_count} chunks indexed</span>
          {videoMeta.cached && <span className="cached-badge">cached</span>}
        </div>
        <div className="tabs">
          <button
            className={`tab ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            Chat
          </button>
          <button
            className={`tab ${activeTab === "summary" ? "active" : ""}`}
            onClick={() => setActiveTab("summary")}
          >
            Summary
          </button>
        </div>
      </div>

      {/* Chat Tab */}
      {activeTab === "chat" && (
        <>
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <p>Ask anything about this video</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <ChatMessage key={i} msg={msg} />
            ))}
            {loading && (
              <div className="message assistant">
                <div className="message-content">
                  <LoadingDots />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          <div className="input-area">
            <input
              type="text"
              placeholder="Ask a question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              disabled={loading}
              className="chat-input"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="btn-primary"
            >
              Send
            </button>
          </div>
        </>
      )}

      {/* Summary Tab */}
      {activeTab === "summary" && (
        <div className="summary-panel">
          <div className="summary-controls">
            <select
              value={summaryType}
              onChange={(e) => setSummaryType(e.target.value)}
              className="summary-select"
            >
              <option value="concise">Concise — 3-5 sentences</option>
              <option value="detailed">Detailed — topic by topic</option>
              <option value="bullets">Bullets — key takeaways</option>
            </select>
            <button
              onClick={fetchSummary}
              disabled={summaryLoading}
              className="btn-primary"
            >
              {summaryLoading ? <LoadingDots /> : "Generate"}
            </button>
          </div>
          {summary && (
            <div className="summary-output">
              <p>{summary}</p>
            </div>
          )}
          {!summary && !summaryLoading && (
            <div className="empty-state">
              <p>Choose a style and click Generate</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  const [session, setSession] = useState(null);

  const handleLoaded = (data) => {
    setSession(data);
  };

  const handleReset = () => {
    setSession(null);
  };

  return (
    <div className="app">
      {!session ? (
        <VideoLoader onLoaded={handleLoaded} />
      ) : (
        <div className="workspace">
          <button className="btn-ghost reset-btn" onClick={handleReset}>
            ← Load another video
          </button>
          <ChatPanel sessionId={session.session_id} videoMeta={session} />
        </div>
      )}
    </div>
  );
}