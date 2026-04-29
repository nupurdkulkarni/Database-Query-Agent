import React, { useState, useRef, useEffect } from 'react';
import { Send, Database, Loader2, Bot, User, Trash2, Plus } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type RecordRow = Record<string, string | number | boolean | null | object>;

interface Source {
  table: string;
  relevance: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sql?: string;
  records?: RecordRow[];
  sources?: Source[];
  timings?: Record<string, number>;
  status?: string;
  error?: string;
  schema_info?: string;
}

interface Session {
  thread_id: string;
  title: string;
  created_at: string;
}

const API_BASE = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';

const generateId = () => Math.random().toString(36).substring(7);

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState(() => generateId());
  const [sessions, setSessions] = useState<Session[]>([]);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions`);
      const data = await res.json();
      setSessions(data);
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    let mounted = true;
    const init = async () => {
      if (mounted) {
        await fetchSessions();
      }
    };
    init();
    return () => { mounted = false; };
  }, []);

  const loadSession = async (tid: string) => {
    setThreadId(tid);
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${tid}/history`);
      const history = await res.json();
      setMessages(history);
    } catch (e) {
      console.error("Failed to load history", e);
    } finally {
      setIsLoading(false);
    }
  };

  const deleteSession = async (tid: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Don't trigger loadSession
    setDeleteId(tid);
  };

  const confirmDelete = async () => {
    if (!deleteId) return;

    try {
      await fetch(`${API_BASE}/api/sessions/${deleteId}`, { method: 'DELETE' });
      setSessions(prev => prev.filter(s => s.thread_id !== deleteId));
      if (threadId === deleteId) {
        setMessages([]);
        setThreadId(generateId());
      }
    } catch (e) {
      console.error("Failed to delete session", e);
    } finally {
      setDeleteId(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage, thread_id: threadId }),
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);

              if (data.type === 'token') {
                assistantMessage += data.content;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], content: assistantMessage };
                  return newMsgs;
                });
              } else if (data.type === 'final') {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = {
                    ...newMsgs[newMsgs.length - 1],
                    content: data.answer || assistantMessage,
                    sql: data.sql,
                    records: data.records,
                    sources: data.sources,
                    timings: data.timings,
                    status: data.status,
                    error: data.error,
                    schema_info: data.schema_info
                  };
                  return newMsgs;
                });
                fetchSessions(); // Refresh sidebar to show new chat
              } else if (data.type === 'error') {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], content: `Error: ${data.message}`, error: data.message };
                  return newMsgs;
                });
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], content: `Connection error: ${errorMessage}` };
        return newMsgs;
      });
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="flex h-screen bg-white text-gray-800 font-sans">
      {/* Sidebar (Streamlit-style) */}
      <div className="w-64 bg-gray-50 border-r border-gray-200 p-4 flex flex-col h-full">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">Chats</h2>
        <button
          onClick={() => {
            setMessages([]);
            setThreadId(generateId());
          }}
          className="w-full bg-white border border-gray-300 rounded-md py-2 px-4 text-sm font-medium hover:bg-gray-50 hover:border-gray-400 transition-colors flex items-center justify-center gap-2 shadow-sm mb-6"
        >
          <Plus size={16} />
          New chat
        </button>

        <div className="flex-1 overflow-y-auto space-y-1">
          {sessions.map((s) => (
            <div
              key={s.thread_id}
              className={`group flex items-center justify-between px-3 py-2 rounded-md text-sm cursor-pointer transition-colors ${threadId === s.thread_id
                  ? 'bg-blue-50 text-blue-700 font-medium'
                  : 'text-gray-600 hover:bg-gray-100'
                }`}
              onClick={() => loadSession(s.thread_id)}
            >
              <span className="truncate flex-1">{s.title || 'Untitled Chat'}</span>
              <button
                onClick={(e) => deleteSession(s.thread_id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-opacity"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <div className="mt-auto pt-4 border-t border-gray-200">
          <div className="text-[10px] text-gray-400 uppercase tracking-widest mb-2 font-bold">
            Engine
          </div>
          <div className="text-xs text-gray-600 bg-gray-100 p-2 rounded flex items-center gap-2">
            <Bot size={14} className="text-blue-500" />
            LangGraph + PostgreSQL
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Delete Confirmation Modal */}
        {deleteId && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-[2px] animate-in fade-in duration-200">
            <div className="bg-white rounded-xl shadow-2xl border border-gray-100 p-6 max-w-sm w-full mx-4 animate-in zoom-in-95 duration-200">
              <div className="flex items-center gap-3 text-red-600 mb-4">
                <div className="p-2 bg-red-50 rounded-full">
                  <Trash2 size={20} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Delete Chat?</h3>
              </div>
              <p className="text-gray-600 text-sm mb-6 leading-relaxed">
                This will permanently remove the conversation history and it cannot be recovered.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setDeleteId(null)}
                  className="flex-1 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-sm font-medium transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDelete}
                  className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Header */}
        <header className="h-14 border-b border-gray-200 flex items-center px-6 bg-white shrink-0">
          <h1 className="text-xl font-semibold">Talk to DB</h1>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 max-w-4xl mx-auto w-full">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-400 space-y-4 mt-20">
              <Database size={48} className="opacity-20" />
              <p className="text-lg">Ask a question about your database...</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className="flex gap-4">
                <div className="shrink-0 pt-1">
                  {msg.role === 'user' ? (
                    <div className="w-8 h-8 bg-blue-500 rounded flex items-center justify-center text-white">
                      <User size={20} />
                    </div>
                  ) : (
                    <div className="w-8 h-8 bg-emerald-500 rounded flex items-center justify-center text-white">
                      <Bot size={20} />
                    </div>
                  )}
                </div>
                <div className="flex-1 space-y-2 min-w-0">
                  <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'text-gray-800' : 'text-gray-700'}`}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    {msg.role === 'assistant' && isLoading && idx === messages.length - 1 && (
                      <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-1 align-middle"></span>
                    )}
                  </div>


                  {/* SQL Expander (or Metadata Info) */}
                  {msg.sql && (
                    <details className="mt-4 border border-gray-200 rounded-md overflow-hidden bg-gray-50">
                      <summary className="px-4 py-2 bg-gray-100 cursor-pointer text-sm font-medium hover:bg-gray-200 text-gray-700">
                        {msg.status === 'metadata' ? 'View Reference Tables' : 'View Generated SQL'}
                      </summary>
                      <div className="p-4 overflow-x-auto">
                        <pre className="text-xs font-mono text-gray-800">
                          {msg.sql}
                        </pre>
                      </div>
                    </details>
                  )}

                  {/* Schema Metadata Expander */}
                  {msg.schema_info && (
                    <details className="mt-2 border border-gray-200 rounded-md overflow-hidden bg-gray-50">
                      <summary className="px-4 py-2 bg-gray-100 cursor-pointer text-sm font-medium hover:bg-gray-200 text-gray-700">
                        View Schema Metadata
                      </summary>
                      <div className="p-4 overflow-x-auto max-h-96">
                        <pre className="text-xs font-mono text-gray-800 whitespace-pre-wrap">
                          {msg.schema_info}
                        </pre>
                      </div>
                    </details>
                  )}

                  {/* Records Expander */}
                  {msg.records && msg.records.length > 0 && (
                    <details className="mt-2 border border-gray-200 rounded-md overflow-hidden bg-gray-50">
                      <summary className="px-4 py-2 bg-gray-100 cursor-pointer text-sm font-medium hover:bg-gray-200 text-gray-700">
                        View Raw Records ({msg.records!.length})
                      </summary>
                      <div className="p-4 overflow-x-auto max-h-96">
                        <table className="min-w-full divide-y divide-gray-200 text-sm border border-gray-200">
                          <thead className="bg-gray-100">
                            <tr>
                              {Object.keys(msg.records![0] || {}).map((key) => (
                                <th key={key} className="px-3 py-2 text-left font-medium text-gray-700 uppercase tracking-wider border-b">
                                  {key}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {msg.records!.slice(0, 50).map((row, i) => (
                              <tr key={i} className="hover:bg-gray-50">
                                {Object.keys(msg.records![0] || {}).map((key) => (
                                  <td key={key} className="px-3 py-2 whitespace-nowrap text-gray-600 truncate max-w-xs border-b">
                                    {typeof row[key] === 'object' && row[key] !== null ? JSON.stringify(row[key]) : String(row[key] ?? '')}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {msg.records!.length > 50 && (
                          <div className="text-xs text-gray-500 mt-2 text-center">
                            Showing first 50 rows.
                          </div>
                        )}
                        {msg.status && msg.status !== 'ok' && (
                          <div className="mt-2 text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Status: {msg.status}
                          </div>
                        )}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-white border-t border-gray-200 shrink-0">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSubmit} className="relative flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
                placeholder="Ask a question about your database..."
                className="w-full pl-4 pr-12 py-3 bg-white border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500 transition-shadow text-gray-900"
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="absolute right-2 p-2 text-gray-400 hover:text-blue-500 disabled:opacity-50 disabled:hover:text-gray-400 transition-colors"
              >
                {isLoading ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
              </button>
            </form>
            <div className="text-center mt-2">
              <span className="text-xs text-gray-400">AI-generated SQL executing against real-time database schema.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
