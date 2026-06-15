import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { uploadPdf, askQuestion, getSummary } from "./api";
import "./App.css";


function App() {
  const [file, setFile] = useState(null);
  const [paperId, setPaperId] = useState("");
  const [filename, setFilename] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);

  const suggestions = [
    "Summarize this paper in simple words",
    "What is the main objective of this paper?",
    "What methodology is used?",
    "What datasets are mentioned?",
    "What are the limitations?",
    "Explain the conclusion",
  ];

  const handleUpload = async () => {
    if (!file) {
      alert("Please choose a PDF first.");
      return;
    }

    try {
      setUploading(true);

      const data = await uploadPdf(file);

      setPaperId(data.paper_id);
      setFilename(data.original_filename);

      setMessages([
        {
          role: "assistant",
          content: `Your paper "${data.original_filename}" is ready. You can now ask questions about it.`,
          type: "success",
        },
      ]);
    } catch (error) {
      console.error(error);
      alert("Upload failed. Please check if your backend is running.");
    } finally {
      setUploading(false);
    }
  };

  const sendQuestion = async (customQuestion = null) => {
    const finalQuestion = customQuestion || question;

    if (!paperId) {
      alert("Please upload a paper first.");
      return;
    }

    if (!finalQuestion.trim()) {
      alert("Please type a question.");
      return;
    }

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: finalQuestion,
      },
    ]);

    setQuestion("");

    try {
      setThinking(true);

      const data = await askQuestion(paperId, finalQuestion, false);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sourcePages: data.source_pages || [],
          faithfulness: data.faithfulness,
        },
      ]);
    } catch (error) {
      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I could not generate an answer right now. Please check the backend or Gemini quota.",
          type: "error",
        },
      ]);
    } finally {
      setThinking(false);
    }
  };

  const handleSummary = async () => {
    if (!paperId) {
      alert("Please upload a paper first.");
      return;
    }

    try {
      setThinking(true);

      const data = await getSummary(paperId);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.summary,
          sourcePages: data.source_pages || [],
          type: "summary",
        },
      ]);
    } catch (error) {
      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I could not generate the summary right now. Please check the backend or Gemini quota.",
          type: "error",
        },
      ]);
    } finally {
      setThinking(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setQuestion("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  };

  return (
    <div className="page">
      <div className="aurora aurora-one"></div>
      <div className="aurora aurora-two"></div>
      <div className="aurora aurora-three"></div>

      <div className="app-container">
        <aside className="sidebar">
          <div className="brand">
            <div className="logo">SL</div>

            <div>
              <h1>ScholarLens</h1>
              <p>Multilingual RAG Assistant</p>
            </div>
          </div>

          <div className="feature-card">
            <span className="feature-tag">Research AI</span>

            <h2>Chat with your paper</h2>

            <p>
              Upload academic PDFs and get clean, source-backed answers using
              retrieval augmented generation.
            </p>

            <div className="feature-list">
              <span>Semantic Search</span>
              <span>Source Pages</span>
              <span>Multilingual Q&A</span>
            </div>
          </div>

          <div className="upload-card">
            <h2>Upload PDF</h2>

            <label className="file-upload">
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files[0])}
              />

              <span>{file ? file.name : "Choose research paper"}</span>
            </label>

            <button
              className="primary-btn"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? "Indexing paper..." : "Upload paper"}
            </button>

            {filename && (
              <div className="paper-ready">
                <div className="pulse-dot"></div>

                <div>
                  <strong>Paper ready</strong>
                  <p>{filename}</p>
                </div>
              </div>
            )}
          </div>

          <div className="tools-card">
            <h2>Quick Tools</h2>

            <button
              className="tool-btn"
              onClick={handleSummary}
              disabled={!paperId || thinking}
            >
              Generate summary
            </button>

            <button className="tool-btn" onClick={clearChat}>
              Clear conversation
            </button>
          </div>

          <div className="mini-card">
            <p className="mini-title">System status</p>

            <div className={paperId ? "status good" : "status warn"}>
              {paperId ? "Ready to answer" : "Waiting for upload"}
            </div>
          </div>
        </aside>

        <main className="chat-panel">
          <header className="chat-header-pro">
            <div className="header-left">
              <span className="workspace-chip">AI Research Workspace</span>

              <h2>Ask your paper anything</h2>

              <p>
                Get clear answers, summaries, methodology explanations, and
                source-page references from your uploaded research paper.
              </p>
            </div>

            <div className="header-status">
              <span className={paperId ? "live-dot active" : "live-dot"}></span>
              <span>{paperId ? "Paper indexed" : "Upload required"}</span>
            </div>
          </header>

          <section className="quick-prompts">
            {suggestions.slice(0, 4).map((item, index) => (
              <button
                key={index}
                onClick={() => sendQuestion(item)}
                disabled={!paperId || thinking}
              >
                {item}
              </button>
            ))}
          </section>

          <section className="chat-body-pro">
            {messages.length === 0 && (
              <div className="empty-chat">
                <div className="empty-icon">✦</div>

                <h2>Ready when your paper is uploaded</h2>

                <p>
                  Ask about objectives, datasets, methodology, results,
                  limitations, or generate a structured summary.
                </p>

                <div className="empty-cards">
                  <div>
                    <span>01</span>
                    <strong>Upload</strong>
                    <p>Add your research paper PDF.</p>
                  </div>

                  <div>
                    <span>02</span>
                    <strong>Ask</strong>
                    <p>Ask questions in simple language.</p>
                  </div>

                  <div>
                    <span>03</span>
                    <strong>Verify</strong>
                    <p>Check source pages used for answers.</p>
                  </div>
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`chat-message ${
                  message.role === "user"
                    ? "chat-message-user"
                    : "chat-message-ai"
                }`}
              >
                <div className="message-avatar">
                  {message.role === "user" ? "You" : "AI"}
                </div>

                <div
                  className={`message-content-card ${
                    message.role === "user"
                      ? "user-content-card"
                      : "ai-content-card"
                  }`}
                >
                  <div className="message-label">
                    {message.role === "user"
                      ? "Your Question"
                      : "ScholarLens Answer"}
                  </div>

                  <div className="message-text">
                     {message.role === "assistant" ? (
                         <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                         </ReactMarkdown>
                      ) : (
                      message.content
                         )}
                  </div>

                  {message.sourcePages && message.sourcePages.length > 0 && (
                    <div className="source-card">
                      <span>Source pages</span>

                      <div className="page-chip-row">
                        {message.sourcePages.map((page, i) => (
                          <b className="page-chip" key={i}>
                            {page}
                          </b>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {thinking && (
              <div className="chat-message chat-message-ai">
                <div className="message-avatar">AI</div>

                <div className="message-content-card ai-content-card typing-card">
                  <div className="message-label">ScholarLens is thinking</div>

                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </section>

          <footer className="chat-input-pro">
            <textarea
              rows="1"
              placeholder={
                paperId
                  ? "Ask a question about the uploaded paper..."
                  : "Upload a paper first to start asking questions..."
              }
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!paperId || thinking}
            />

            <button
              onClick={() => sendQuestion()}
              disabled={!paperId || thinking}
            >
              Send
            </button>
          </footer>
        </main>
      </div>
    </div>
  );
}

export default App;