import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatMessage = ({ message, isProcessing }) => {
  if (isProcessing) {
    return (
      <div className="message assistant">
        <div className="message-header">Assistant</div>
        <div className="message-content">
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    );
  }

  // Check if this is a mode activation message
  const isModeActivationMessage = message.role === 'system' && 
    (message.content.includes('mode activated') || 
     message.content.includes('Mode activated'));

  return (
    <div className={`message ${message.role} ${isModeActivationMessage ? 'mode-activated' : ''}`}>
      <div className="message-header">
        {message.role === 'user' ? 'You' : 
         message.role === 'assistant' ? 'Assistant' : 'System'}
      </div>
      <div className="message-content">
        {message.role === 'assistant' ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        ) : (
          <pre>{message.content}</pre>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;