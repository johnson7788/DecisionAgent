import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import FormComponent from './FormDisplay';
import ResearchDisplay from './ResearchDisplay';
import MessageRenderer from './MessageRenderer';
import { sendMessageStreaming } from '../services/a2aApiService';


function ChatInterface({ agentCard }) {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const currentStreamingMessageIdRef = useRef(null);
  const abortStreamingRef = useRef(null);
  const [sessionId, setSessionId] = useState('');
  //思考的数据
  const [thinkingMessages, setThinkingMessages] = useState({});
  // 状态的数据
  const [statusMessages, setStatusMessages] = useState({});
  const [isThinkingCollapsed, setIsThinkingCollapsed] = useState({});
  const [researchResults, setResearchResults] = useState({});
  //根据researchResults的变化更新referenceMap
  const [referenceMap, setReferenceMap] = useState({});

  useEffect(() => {
    // 检查 researchResults 是否有效并包含 data
    if (researchResults && Object.keys(researchResults).length > 0) {
      const newMap = {};

      Object.values(researchResults).forEach((results) => {
        // 遍历 results 数组中的每个 result
        results.forEach((result) => {
          if (result.data && Array.isArray(result.data.data)) {
            result.data.data.forEach((item) => {
              item.match_sentences.forEach((sentence_info) => {
                sentence_info["title"] = item.title; // 添加标题信息
                newMap[sentence_info.id] = sentence_info;
              });
            });
          }
        });
      });

      setReferenceMap(newMap);
    } else {
      setReferenceMap({});
    }
  }, [researchResults]);

  useEffect(() => {
    setSessionId(uuidv4());
    setMessages([]);
    setThinkingMessages({});
    setStatusMessages({});
    setIsThinkingCollapsed({});
  }, [agentCard]);

  const toggleThinkingCollapse = (messageId) => {
    setIsThinkingCollapsed((prev) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || !agentCard || !sessionId) return;

    const userMessage = { id: uuidv4(), type: 'user', text: prompt };
    const agentMessage = {
      id: uuidv4(),
      type: 'agent',
      text: '',
      isStreaming: true,
      thinking: [],
      form: null,
    };
    setMessages((prev) => [...prev, userMessage, agentMessage]);
    setThinkingMessages((prev) => ({ ...prev, [agentMessage.id]: [] }));
    setIsThinkingCollapsed((prev) => ({ ...prev, [agentMessage.id]: false }));
    setPrompt('');
    setIsSending(true);
    setError(null);

    const taskId = uuidv4();
    const agentEndpointUrl = agentCard.agentEndpointUrl;


    currentStreamingMessageIdRef.current = agentMessage.id;

    abortStreamingRef.current = sendMessageStreaming(
      agentEndpointUrl,
      prompt,
      {
        contextId: sessionId,
      },
      (streamEvent) => {
        console.log('接收到流事件：', streamEvent);

        if (streamEvent.result?.status?.state === 'working' && streamEvent.result?.status?.message) {
          streamEvent.result.status.message.parts.forEach((part) => {
            if (part.kind === 'text' && part.text) {
              setThinkingMessages((prev) => ({
                ...prev,
                [agentMessage.id]: [...(prev[agentMessage.id] || []), part.text],
              }));
            } else if (part.kind === 'data' && part.data?.kind === 'tool_calls') {
              const toolCalls = part.data.data;
              const displayMessages = toolCalls.map((toolCall) => toolCall.display).join('');
              setStatusMessages((prev) => ({
                ...prev,
                [agentMessage.id]: [...(prev[agentMessage.id] || []), displayMessages],
              }));
              console.log('statusMessage tool call', statusMessages);
            } else if (part.kind === 'data' && part.data?.kind === 'tool_result') {
              const toolResults = part.data.data;
              setResearchResults(prev => ({
                ...prev,
                [agentMessage.id]: [...(prev[agentMessage.id] || []), ...toolResults]
              }));
              const displayMessages = toolResults.map((toolRes) => toolRes.display).join('');
              setStatusMessages((prev) => ({
                ...prev,
                [agentMessage.id]: [...(prev[agentMessage.id] || []), displayMessages],
              }));
              console.log('statusMessages tool result', statusMessages);
            }
          });
        }

        if (streamEvent.result?.artifact) {
          const { parts, append, artifactId } = streamEvent.result.artifact;
          parts.forEach((part) => {
            if (part.kind === 'text' && part.text) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMessage.id
                    ? {
                        ...msg,
                        text: append ? msg.text + part.text : part.text,
                        isStreaming: !streamEvent.result.artifact.lastChunk,
                      }
                    : msg
                )
              );
            } else if (part.kind === 'data' && part.data?.kind === 'form') {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === agentMessage.id
                    ? { ...msg, form: part.data, isStreaming: false }
                    : msg
                )
              );
            }
          });
        }

        if (
          streamEvent.result?.final ||
          streamEvent.result?.status?.state === 'completed' ||
          streamEvent.result?.status?.state === 'input-required'
        ) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === agentMessage.id ? { ...msg, isStreaming: false } : msg
            )
          );
          setIsSending(false);
        }
      },
      (streamError) => {
        console.error('流式传输错误：', streamError);
        setError(`流式传输错误：${streamError.message}`);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === agentMessage.id
              ? { ...msg, text: `错误：${streamError.message}`, isStreaming: false }
              : msg
          )
        );
        setIsSending(false);
      }
    );

  };

  useEffect(() => {
    return () => {
      if (abortStreamingRef.current) {
        abortStreamingRef.current();
      }
    };
  }, []);

  const messagesEndRef = useRef(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, thinkingMessages, statusMessages]);

  if (!agentCard) {
    return (
      <div className="p-6 text-center text-gray-500 text-lg font-medium bg-white rounded-xl shadow-lg">
        请先选择一个智能体以开始聊天。
      </div>
    );
  }

  if (!sessionId) {
    return (
      <div className="p-6 text-center text-gray-500 text-lg font-medium bg-white rounded-xl shadow-lg">
        正在生成会话 ID...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[700px] max-h-[2000px] bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="p-6 bg-gray-50 border-b border-gray-200">
        <h2 className="text-2xl font-bold text-gray-800">Chat with {agentCard.name}</h2>
        <p className="text-sm text-gray-500 mt-1">
          Session ID (Base64):{' '}
          <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">{sessionId.substring(0, 30)}...</code>
        </p>
      </div>

      <div className="flex-grow p-6 space-y-6 overflow-y-auto bg-gray-50">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'} items-start`}
          >
            {msg.type === 'agent' && (
              <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center mr-3">
                <svg
                  className="w-6 h-6 text-indigo-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
              </div>
            )}
            <div
              className={`max-w-3xl px-4 py-3 rounded-2xl shadow-md ${
                msg.type === 'user' ? 'bg-indigo-600 text-white' : 'bg-white text-gray-800'
              }`}
            >
              {msg.type === 'agent' && statusMessages[msg.id]?.length > 0 && (
                <div className="mb-3">
                  <div className="mt-2 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-800 border border-yellow-200">
                    <ul className="list-disc pl-5">
                      {statusMessages[msg.id].map((status, idx) => (
                        <p key={idx} className="mb-1 whitespace-pre-wrap">{status}</p>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              {msg.type === 'agent' && thinkingMessages[msg.id]?.length > 0 && (
                <div className="mb-3">
                  <button
                    onClick={() => toggleThinkingCollapse(msg.id)}
                    className="text-sm text-indigo-600 hover:text-indigo-800 flex items-center font-medium"
                  >
                    {isThinkingCollapsed[msg.id] ? 'Show Thought Process' : 'Hide Thought Process'}
                    <svg
                      className={`w-4 h-4 ml-1 transform ${
                        isThinkingCollapsed[msg.id] ? '' : 'rotate-180'
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </button>
                  {!isThinkingCollapsed[msg.id] && (
                    <div className="mt-2 p-3 bg-gray-100 rounded-lg text-sm text-gray-600">
                      <p className="whitespace-pre-wrap">
                        {thinkingMessages[msg.id]
                          ?.join('')
                          .replace(/\\\\n/g, '\\n')
                          .replace(/\\n/g, '\n')}
                      </p>
                    </div>
                  )}
                </div>
              )}
              {msg.form ? (
                <FormComponent
                  formData={msg.form}
                  onSubmit={handleFormSubmit}
                  messageId={msg.id}
                />
              ) : (
                <div className="text-lg leading-relaxed">
                  {msg.text ? (
                    <MessageRenderer text={msg.text} referenceMap={referenceMap} />
                  ) : (
                    msg.isStreaming ? (
                      <span className="text-lg leading-relaxed">Thinking...<span className="animate-pulse">▍</span></span>
                    ) : null
                  )}
                </div>
              )}
              {/* 显示每个数据库的搜索结果的组件 */}
              {msg.type === 'agent' && !msg.isStreaming && researchResults[msg.id] && researchResults[msg.id].length > 0 && (
                <div className="mt-4">
                  <ResearchDisplay data={researchResults[msg.id]} />
                </div>
              )}
            </div>
            {msg.type === 'user' && (
              <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center ml-3">
                <svg
                  className="w-6 h-6 text-indigo-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M3 5h18M5 9h14M3 17h18M5 21h14M5 3l4 3m0 0l-4 3m4-3v14"
                  />
                </svg>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {error && (
        <div className="p-4 border-t bg-red-50 text-red-700 text-base font-medium">{error}</div>
      )}

      <form onSubmit={handleSendMessage} className="p-4 border-t flex items-center space-x-3 bg-white">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Type your message..."
          className="flex-grow p-3 border border-gray-200 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-base placeholder-gray-400 disabled:bg-gray-100"
          disabled={isSending || !agentCard}
        />
        <button
          type="submit"
          className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 transition-colors"
          disabled={isSending || !prompt.trim() || !agentCard}
        >
          {isSending ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default ChatInterface;