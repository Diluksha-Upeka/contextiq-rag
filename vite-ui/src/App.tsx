import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, Loader2, Info, Bot, User, BookOpen, ChevronDown, ChevronUp, FileText, Scissors, Brain, Database, Sparkles, CheckCircle2, Clock3, Copy, Check, MessageSquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const API_DOWN_HELP = `Cannot reach API at ${API_BASE_URL}. Start backend with: python main.py`;

const SUGGESTED_PROMPTS = [
  "Summarize this in 5 bullet points",
  "What are the key takeaways?",
  "Explain the core methodology",
  "What are the main challenges mentioned?"
];

const INDEXING_STAGES = [
  { title: 'Uploading PDF', detail: 'Transferring your file securely', seconds: 2.5, icon: FileText },
  { title: 'Extracting Text', detail: 'Reading pages and collecting document text', seconds: 4.5, icon: Sparkles },
  { title: 'Chunking Content', detail: 'Splitting content into retrieval-ready chunks', seconds: 6.0, icon: Scissors },
  { title: 'Generating Embeddings', detail: 'Turning chunks into semantic vectors', seconds: 9.0, icon: Brain },
  { title: 'Indexing in Pinecone', detail: 'Storing vectors for fast question answering', seconds: 9.0, icon: Database },
] as const;

function getIndexingProgressState(elapsedMs: number) {
  const totalMs = INDEXING_STAGES.reduce((sum, s) => sum + s.seconds * 1000, 0);
  const clampedElapsed = Math.max(0, elapsedMs);
  let cursor = 0;
  let stageIndex = INDEXING_STAGES.length - 1;

  for (let i = 0; i < INDEXING_STAGES.length; i += 1) {
    const stageMs = INDEXING_STAGES[i].seconds * 1000;
    const nextCursor = cursor + stageMs;
    if (clampedElapsed < nextCursor) {
      stageIndex = i;
      break;
    }
    cursor = nextCursor;
  }

  const stageMs = INDEXING_STAGES[stageIndex].seconds * 1000;
  const stageElapsed = Math.max(0, clampedElapsed - cursor);
  const stageRatio = Math.min(1, stageElapsed / stageMs);

  const stageStartProgress = (stageIndex / INDEXING_STAGES.length) * 100;
  const stageEndProgress = ((stageIndex + 1) / INDEXING_STAGES.length) * 100;
  const easedProgress = stageStartProgress + (stageEndProgress - stageStartProgress) * stageRatio;

  const progress = Math.min(97, Math.max(4, (clampedElapsed / totalMs) * 15 + easedProgress * 0.85));
  return {
    stageIndex,
    progress,
    elapsedSeconds: Math.floor(clampedElapsed / 1000),
  };
}

export default function App() {
  const [messages, setMessages] = useState<{role: 'user' | 'assistant', content: string, sources?: {id: number, text: string}[]}[]>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isUploadCompleting, setIsUploadCompleting] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [namespace, setNamespace] = useState<string | null>(null);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  const [indexingElapsedMs, setIndexingElapsedMs] = useState(0);
  const [completionProgress, setCompletionProgress] = useState(100);
  const [copiedMsgIdx, setCopiedMsgIdx] = useState<number | null>(null);
  const completionStartRef = useRef(100);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const toggleSources = (messageIndex: number) => {
    setExpandedSources(prev => ({ ...prev, [messageIndex]: !prev[messageIndex] }));
  };

  const stripInlineContextSources = (content: string) => {
    const marker = /\n\s*context sources\s*$/im;
    const match = marker.exec(content);
    if (!match || match.index <= 0) return content;
    return content.slice(0, match.index).trim();
  };

  const renderInlineMarkdown = (text: string) => {
    const chunks = text.split(/(\*\*[^*]+\*\*)/g);
    return chunks.map((chunk, idx) => {
      const boldMatch = chunk.match(/^\*\*([^*]+)\*\*$/);
      if (boldMatch) {
        return <strong key={idx} className='font-semibold text-slate-900'>{boldMatch[1]}</strong>;
      }
      return <React.Fragment key={idx}>{chunk}</React.Fragment>;
    });
  };

  const renderAssistantContent = (rawContent: string, hasStructuredSources: boolean) => {
    const content = hasStructuredSources ? stripInlineContextSources(rawContent) : rawContent;
    const lines = content.split('\n').map((line) => line.trimEnd());
    const nonEmptyLines = lines.filter((line) => line.trim().length > 0);
    const bulletLikeCount = nonEmptyLines.filter((line) => /^[-*•]\s+/.test(line.trim())).length;
    const isMostlyBullets = nonEmptyLines.length > 0 && bulletLikeCount >= Math.ceil(nonEmptyLines.length * 0.4);

    if (isMostlyBullets) {
      return (
        <div className='space-y-3'>
          {lines.map((line, idx) => {
            const trimmed = line.trim();
            if (!trimmed) {
              return <div key={idx} className='h-1' />;
            }

            if (/^[-*•]\s+/.test(trimmed)) {
              const text = trimmed.replace(/^[-*•]\s+/, '');
              return (
                <div key={idx} className='flex items-start gap-2'>
                  <span className='mt-2 h-1.5 w-1.5 rounded-full bg-blue-500 flex-shrink-0' />
                  <p className='leading-relaxed'>{renderInlineMarkdown(text)}</p>
                </div>
              );
            }

            return <p key={idx} className='leading-relaxed whitespace-pre-wrap'>{renderInlineMarkdown(line)}</p>;
          })}
        </div>
      );
    }

    return (
      <div className='space-y-3'>
        {lines.map((line, idx) => (
          line.trim()
            ? <p key={idx} className='leading-relaxed whitespace-pre-wrap'>{renderInlineMarkdown(line)}</p>
            : <div key={idx} className='h-1' />
        ))}
      </div>
    );
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    if (!isUploading && !isUploadCompleting) {
      setIndexingElapsedMs(0);
      return;
    }

    if (!isUploading) return;

    const startedAt = Date.now();
    const tick = () => setIndexingElapsedMs(Date.now() - startedAt);
    tick();
    const interval = window.setInterval(tick, 180);
    return () => window.clearInterval(interval);
  }, [isUploading, isUploadCompleting]);

  useEffect(() => {
    if (!isUploadCompleting) return;

    const from = completionStartRef.current;
    const durationMs = 900;
    let raf = 0;
    const startedAt = performance.now();

    const step = (now: number) => {
      const t = Math.min(1, (now - startedAt) / durationMs);
      const eased = 1 - (1 - t) ** 3;
      setCompletionProgress(from + (100 - from) * eased);
      if (t < 1) {
        raf = window.requestAnimationFrame(step);
      } else {
        window.setTimeout(() => {
          setIsUploadCompleting(false);
          setCompletionProgress(100);
        }, 240);
      }
    };

    raf = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(raf);
  }, [isUploadCompleting]);

  const indexingState = getIndexingProgressState(indexingElapsedMs);
  const overlayVisible = isUploading || isUploadCompleting;
  const displayProgress = isUploadCompleting ? completionProgress : indexingState.progress;
  const displayElapsedSeconds = Math.floor(indexingElapsedMs / 1000);
  const currentStage = isUploadCompleting
    ? {
        title: 'Finalizing Workspace',
        detail: 'Wrapping up indexing and preparing your chat session',
      }
    : INDEXING_STAGES[indexingState.stageIndex];
  const ringRadius = 56;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const ringOffset = ringCircumference * (1 - displayProgress / 100);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setIsUploadCompleting(false);
    setCompletionProgress(100);
    const formData = new FormData();
    formData.append('file', file);
    let uploadSucceeded = false;

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Upload failed (${response.status}): ${err}`);
      }
      const data = await response.json();
      uploadSucceeded = true;
      setNamespace(data.namespace);
      setMessages([{ role: 'assistant', content: 'Document successfully indexed! What would you like to know about it?' }]);
    } catch (error) {
      console.error('Upload failed:', error);
      if (error instanceof TypeError) {
        alert(API_DOWN_HELP);
      } else {
        alert('Failed to upload document');
      }
    } finally {
      if (uploadSucceeded) {
        const startAt = Math.max(88, getIndexingProgressState(indexingElapsedMs).progress);
        completionStartRef.current = startAt;
        setCompletionProgress(startAt);
        setIsUploadCompleting(true);
      }
      setIsUploading(false);
    }
  };

  const handleCopy = (text: string, idx: number) => {
    const rawText = stripInlineContextSources(text);
    navigator.clipboard.writeText(rawText);
    setCopiedMsgIdx(idx);
    setTimeout(() => setCopiedMsgIdx(null), 2000);
  };

  const handleChipClick = (query: string) => {
    setInput(query);
    submitUserQuery(query);
  };

  const submitUserQuery = async (userQuery: string) => {
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setIsTyping(true);

    const explainRef = userQuery.match(/^\s*(?:explain|elaborate|expand(?:\s+on)?)\s*\[?(\d+)\]?\s*$/i);
    const latestAssistantWithSources = [...messages].reverse().find(
      (m) => m.role === 'assistant' && m.sources && m.sources.length > 0,
    );

    let effectiveQuery = userQuery;
    if (explainRef) {
      const targetId = Number(explainRef[1]);
      const source = latestAssistantWithSources?.sources?.find((s) => s.id === targetId);
      if (!source) {
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: `I could not find source [${targetId}] in the latest answer. Try a source number shown in the most recent Context Sources list.` },
        ]);
        setIsTyping(false);
        return;
      }
      effectiveQuery = `Explain source [${source.id}] in simple terms and include key takeaways:\n${source.text}`;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: effectiveQuery, namespace }),
      });
      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Query failed (${response.status}): ${err}`);
      }
      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources.map((s: { id: number; text: string }) => ({ id: s.id, text: s.text }))
      }]);
    } catch (error) {
      console.error('Query failed:', error);
      const content = error instanceof TypeError
        ? `${API_DOWN_HELP}`
        : 'Sorry, I encountered an error processing your request.';
      setMessages(prev => [...prev, { role: 'assistant', content }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSendMessage = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || !namespace || isTyping || overlayVisible) return;
    submitUserQuery(input.trim());
  };

  return (
    <div className='flex h-screen bg-[#f9fafb] text-slate-800 font-sans selection:bg-blue-200 selection:text-blue-900 relative'>
      
      {/* Sidebar - Apple Inspired */}
      <aside className='w-80 bg-white/80 backdrop-blur-xl border-r border-slate-200/60 p-6 flex flex-col items-center justify-between shadow-[4px_0_24px_-12px_rgba(0,0,0,0.1)] z-10'>
        <div className='w-full'>
          <div className='flex items-center justify-center gap-3 mb-10 mt-4'>
            <div className='bg-blue-600 p-2.5 rounded-2xl shadow-lg shadow-blue-600/20'>
              <BookOpen className='w-6 h-6 text-white' />
            </div>
            <h1 className='text-2xl font-bold tracking-tight text-slate-900'>Context<span className='text-blue-600'>IQ</span></h1>
          </div>

          <div className='w-full p-5 bg-slate-50 border border-slate-200 rounded-3xl mb-6 relative overflow-hidden group hover:border-blue-300 transition-colors duration-300'>
            <div className='absolute inset-0 bg-gradient-to-br from-blue-50/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none' />
            <h2 className='text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wider'>Knowledge Base</h2>
            <p className='text-sm text-slate-500 mb-4 leading-relaxed'>Upload a PDF document to begin an interactive, context-aware conversation.</p>
            
            <input 
              type='file' 
              accept='.pdf' 
              className='hidden' 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
            />
            
            <button 
              onClick={() => fileInputRef.current?.click()}
              disabled={overlayVisible}
              className='w-full py-3 px-4 bg-white border shadow-sm border-slate-200 text-slate-700 rounded-xl hover:bg-slate-50 hover:text-blue-600 hover:border-blue-200 transition-all duration-200 flex items-center justify-center gap-2 font-medium disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98]'
            >
              {overlayVisible ? (
                <Loader2 className='w-5 h-5 animate-spin text-blue-600' />
              ) : (
                <>
                  <Upload className='w-4 h-4' />
                  Select PDF Document
                </>
              )}
            </button>
            
            {namespace && (
              <div className='mt-4 flex items-center gap-2 text-xs font-medium text-emerald-600 bg-emerald-50 py-1.5 px-3 rounded-lg border border-emerald-100'>
                <div className='w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse' />
                Document Indexed Successfully
              </div>
            )}
          </div>
        </div>

        <div className='text-xs text-slate-400 text-center flex items-center gap-1.5 bg-slate-100 py-2 px-4 rounded-full'>
          <Info className='w-3.5 h-3.5' />
          Public demo. Do not upload sensitive data.
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className='flex-1 flex flex-col h-screen relative bg-[#f9fafb]'>
        
        {/* Chat Header */}
        <header className='h-16 border-b border-slate-200/60 bg-white/60 backdrop-blur-md sticky top-0 z-10 flex items-center px-8 shadow-sm'>
          <h2 className='text-sm font-medium text-slate-600 flex items-center gap-2'>
            <div className='w-2 h-2 rounded-full bg-indigo-400' />
            Active Session
          </h2>
        </header>

        {/* Messages */}
        <div className='flex-1 overflow-y-auto px-4 py-8 scroll-smooth'>
          <div className='max-w-3xl mx-auto space-y-8'>
            
            {messages.length === 0 ? (
              <div className='flex flex-col items-center justify-center h-full mt-32 text-center'>
                 <div className='w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mb-6 shadow-[0_0_40px_-10px_rgba(37,99,235,0.2)]'>
                    <Bot className='w-10 h-10 text-blue-500' />
                 </div>
                 <h2 className='text-2xl font-semibold text-slate-800 mb-2'>Welcome to ContextIQ</h2>
                 <p className='text-slate-500 max-w-md'>Your intelligent document assistant. Upload a PDF on the left to start analyzing context.</p>
              </div>
            ) : (
              <AnimatePresence initial={false}>
                {messages.map((m, idx) => (
                  <motion.div 
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                    className={`flex gap-4 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {m.role === 'assistant' && (
                      <div className='w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shadow-md flex-shrink-0 mt-1'>
                        <Bot className='w-4 h-4 text-white' />
                      </div>
                    )}
                    
                    <div className={`max-w-[85%] rounded-3xl px-6 py-4 shadow-sm relative group ${
                      m.role === 'user'
                        ? 'bg-blue-600 text-white rounded-br-sm'
                        : 'bg-white border border-slate-200 text-slate-800 rounded-bl-sm'
                    }`}>
                      {m.role === 'assistant' && (
                        <button
                          onClick={() => handleCopy(m.content, idx)}
                          className='absolute top-3 right-3 p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg opacity-0 group-hover:opacity-100 transition-all focus:opacity-100 focus:outline-none'
                          title='Copy response'
                        >
                          {copiedMsgIdx === idx ? <Check className='w-4 h-4 text-emerald-500' /> : <Copy className='w-4 h-4' />}
                        </button>
                      )}

                      {m.role === 'assistant'
                        ? renderAssistantContent(m.content, Boolean(m.sources?.length))
                        : <p className='leading-relaxed whitespace-pre-wrap'>{m.content}</p>}
                      
                      {m.role === 'assistant' && idx === 0 && namespace && messages.length === 1 && (
                        <div className='mt-8 mb-2 flex flex-wrap gap-2'>
                          {SUGGESTED_PROMPTS.map((prompt, pIdx) => (
                            <button
                              key={pIdx}
                              onClick={() => handleChipClick(prompt)}
                              className='text-xs font-medium bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-100 px-3 py-2 rounded-full transition-colors flex items-center gap-1.5'
                            >
                              <MessageSquare className='w-3 h-3' />
                              {prompt}
                            </button>
                          ))}
                        </div>
                      )}

                      {m.sources && m.sources.length > 0 && (
                        <div className='mt-4 pt-4 border-t border-slate-100'>
                          <button
                            type='button'
                            onClick={() => toggleSources(idx)}
                            className='w-full flex items-center justify-between text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-1 hover:text-blue-600 transition-colors'
                          >
                            <span>Context Sources ({m.sources.length})</span>
                            {expandedSources[idx] ? <ChevronUp className='w-4 h-4' /> : <ChevronDown className='w-4 h-4' />}
                          </button>
                          <div
                            className={`grid transition-all duration-300 ease-out ${expandedSources[idx] ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}
                          >
                            <div className='overflow-hidden'>
                              <div className='space-y-2'>
                                {m.sources.map((src, sIdx) => (
                                  <div key={sIdx} className='bg-slate-50 rounded-xl p-3 text-sm text-slate-600 border border-slate-100 hover:border-blue-100 hover:bg-blue-50/30 transition-colors'>
                                    <span className='font-semibold text-slate-700'>[{src.id}] </span>
                                    {src.text}
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {m.role === 'user' && (
                      <div className='w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center flex-shrink-0 mt-1 shadow-md'>
                        <User className='w-4 h-4 text-white' />
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            )}

            {isTyping && (
               <motion.div 
                 initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                 className='flex gap-4 justify-start'
               >
                 <div className='w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shadow-md flex-shrink-0 mt-1'>
                    <Bot className='w-4 h-4 text-white' />
                 </div>
                 <div className='bg-white border border-slate-200 rounded-3xl rounded-bl-sm px-5 py-4 shadow-sm flex items-center gap-1.5'>
                    <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0 }} className='w-2 h-2 bg-slate-300 rounded-full' />
                    <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }} className='w-2 h-2 bg-slate-300 rounded-full' />
                    <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.4 }} className='w-2 h-2 bg-slate-300 rounded-full' />
                 </div>
               </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className='p-6 bg-transparent'>
          <div className='max-w-3xl mx-auto'>
            <form onSubmit={handleSendMessage} className='relative group flex items-end'>
              <div className='absolute inset-0 bg-blue-400/5 rounded-2xl blur-xl transition-all duration-300 group-hover:bg-blue-400/10 pointer-events-none' />
              <textarea
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px';
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                rows={1}
                disabled={!namespace || isTyping || overlayVisible}
                placeholder={namespace ? 'Ask a question about the document... (Shift+Enter for new line)' : 'Upload a PDF to start asking...'}
                className='w-full pl-6 pr-14 py-4 bg-white border border-slate-200 text-slate-800 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] focus:outline-none focus:ring-4 focus:ring-blue-500/10 focus:border-blue-400 transition-all disabled:opacity-60 relative z-10 text-base placeholder:text-slate-400 resize-none overflow-y-auto'
                style={{ minHeight: '56px', maxHeight: '180px', lineHeight: '1.5' }}
              />
              <button
                type='submit'
                disabled={!input.trim() || !namespace || isTyping || overlayVisible}
                className='absolute right-2.5 bottom-2.5 p-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all z-20 shadow-md shadow-blue-600/20 active:scale-95 flex-shrink-0'
              >
                <Send className='w-4 h-4' />
              </button>
            </form>
            <div className='text-center mt-3 text-xs text-slate-400 font-medium'>
              ContextIQ AI may generate inaccurate information. Verify source references.
            </div>
          </div>
        </div>
      </main>

      <AnimatePresence>
        {overlayVisible && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className='absolute inset-0 z-50 bg-slate-900/45 backdrop-blur-md flex items-center justify-center p-5'
          >
            <motion.div
              initial={{ y: 12, opacity: 0.9, scale: 0.98 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 8, opacity: 0.96, scale: 0.99 }}
              transition={{ duration: 0.28, ease: 'easeOut' }}
              className='w-full max-w-3xl rounded-3xl bg-white/95 border border-white/80 shadow-[0_30px_120px_-24px_rgba(15,23,42,0.5)] overflow-hidden'
            >
              <div className='relative p-8 md:p-10'>
                <div className='absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_top_right,rgba(59,130,246,0.14),transparent_45%),radial-gradient(circle_at_bottom_left,rgba(16,185,129,0.12),transparent_50%)]' />

                <div className='relative flex flex-col md:flex-row items-start md:items-center gap-8'>
                  <div className='relative shrink-0'>
                    <svg className='w-32 h-32 -rotate-90' viewBox='0 0 140 140' aria-hidden='true'>
                      <circle cx='70' cy='70' r={ringRadius} stroke='rgb(226 232 240)' strokeWidth='10' fill='none' />
                      <motion.circle
                        cx='70'
                        cy='70'
                        r={ringRadius}
                        stroke='url(#indexingGradient)'
                        strokeWidth='10'
                        strokeLinecap='round'
                        fill='none'
                        strokeDasharray={ringCircumference}
                        animate={{ strokeDashoffset: ringOffset }}
                        transition={{ duration: 0.4, ease: 'easeOut' }}
                      />
                      <defs>
                        <linearGradient id='indexingGradient' x1='0%' y1='0%' x2='100%' y2='100%'>
                          <stop offset='0%' stopColor='#2563eb' />
                          <stop offset='100%' stopColor='#06b6d4' />
                        </linearGradient>
                      </defs>
                    </svg>
                    <div className='absolute inset-0 flex flex-col items-center justify-center'>
                      <span className='text-2xl font-bold text-slate-800'>{Math.round(displayProgress)}%</span>
                      <span className='text-[11px] font-semibold tracking-wide text-slate-500 uppercase'>Processing</span>
                    </div>
                  </div>

                  <div className='flex-1 w-full'>
                    <div className='flex items-center gap-2 mb-2 text-xs font-semibold text-blue-700 uppercase tracking-[0.12em]'>
                      <Clock3 className='w-3.5 h-3.5' />
                      {isUploadCompleting ? 'Completing Setup' : 'Indexing In Progress'}
                    </div>
                    <h3 className='text-2xl font-bold text-slate-900 mb-1'>{currentStage.title}</h3>
                    <p className='text-slate-600 leading-relaxed mb-6'>{currentStage.detail}</p>

                    <div className='space-y-2'>
                      {INDEXING_STAGES.map((stage, idx) => {
                        const Icon = stage.icon;
                        const done = isUploadCompleting || idx < indexingState.stageIndex;
                        const active = isUploadCompleting ? idx === INDEXING_STAGES.length - 1 : idx === indexingState.stageIndex;
                        return (
                          <div key={stage.title} className={`flex items-center gap-3 rounded-xl px-3 py-2 transition-colors ${active ? 'bg-blue-50 border border-blue-100' : 'border border-transparent'}`}>
                            <div className='w-5 h-5 flex items-center justify-center'>
                              {done ? (
                                <CheckCircle2 className='w-5 h-5 text-emerald-500' />
                              ) : (
                                <Icon className={`w-4 h-4 ${active ? 'text-blue-600' : 'text-slate-400'}`} />
                              )}
                            </div>
                            <span className={`text-sm ${active ? 'text-slate-800 font-semibold' : done ? 'text-slate-600' : 'text-slate-400'}`}>
                              {stage.title}
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    <div className='mt-6 flex flex-wrap items-center gap-3 text-xs'>
                      <span className='inline-flex items-center rounded-full bg-slate-100 px-3 py-1.5 text-slate-600 font-medium'>Elapsed: {displayElapsedSeconds}s</span>
                      <span className='inline-flex items-center rounded-full bg-blue-50 px-3 py-1.5 text-blue-700 font-medium'>Please keep this tab open</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
