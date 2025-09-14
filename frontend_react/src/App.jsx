import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ChatMessage from './components/ChatMessage';
import DocumentList from './components/DocumentList';
import ConversationHistory from './components/ConversationHistory';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [documentInfo, setDocumentInfo] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [showAttachmentOptions, setShowAttachmentOptions] = useState(false);
  const [forcedAgent, setForcedAgent] = useState(null); // Track forced agent selection
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const attachmentButtonRef = useRef(null);

  // Create a new session on component mount, but only if one doesn't exist
  useEffect(() => {
    const storedSessionId = localStorage.getItem('sessionId');
    if (storedSessionId) {
      setSessionId(storedSessionId);
    } else {
      createNewSession();
    }
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle scroll events for shadow effects
  useEffect(() => {
    const messagesContainer = document.querySelector('.messages');
    if (!messagesContainer) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
      const isAtTop = scrollTop <= 5;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 5;
      
      // Toggle classes for shadow effects
      messagesContainer.classList.toggle('scrolled', scrollTop > 5);
      messagesContainer.classList.toggle('at-top', isAtTop);
      messagesContainer.classList.toggle('at-bottom', isAtBottom);
    };

    // Initial check
    handleScroll();
    
    // Add scroll listener
    messagesContainer.addEventListener('scroll', handleScroll);
    
    // Cleanup
    return () => {
      messagesContainer?.removeEventListener('scroll', handleScroll);
    };
  }, [messages]);

  // Ensure shadow classes are updated when messages change
  useEffect(() => {
    const messagesContainer = document.querySelector('.messages');
    if (!messagesContainer) return;
    
    // Trigger a scroll event to update shadow classes
    const event = new Event('scroll');
    messagesContainer.dispatchEvent(event);
  }, [messages]);

  // Close attachment options when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      // Check if click is outside both the attachment button and the dropdown
      if (attachmentButtonRef.current && !attachmentButtonRef.current.contains(event.target)) {
        // Also check if the click is not on the attachment options dropdown
        const attachmentOptions = document.querySelector('.attachment-options');
        if (attachmentOptions && !attachmentOptions.contains(event.target)) {
          setShowAttachmentOptions(false);
        }
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const createNewSession = async () => {
    try {
      const response = await axios.post(`${API_BASE}/session`);
      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      setMessages([]);
      setDocumentInfo([]);
      setForcedAgent(null); // Reset forced agent on new session
      // Store session ID in localStorage
      localStorage.setItem('sessionId', newSessionId);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  const loadConversation = async (conversationData) => {
    try {
      // Set the session ID to the conversation ID
      setSessionId(conversationData.id);
      
      // Load messages
      const loadedMessages = conversationData.messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      setMessages(loadedMessages);
      
      // For document info, we would need to retrieve it from the backend
      // This is a simplified implementation
      setDocumentInfo([]);
      setForcedAgent(null);
    } catch (error) {
      console.error('Error loading conversation:', error);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setShowAttachmentOptions(false);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile || isProcessing) return null;

    // Ensure we have a session before uploading
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      await createNewSession();
      currentSessionId = sessionId;
    }

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE}/upload-document?session_id=${currentSessionId}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      if (response.data.status === 'success') {
        // Return the document info instead of updating state here
        return response.data.document_info;
      }
      return null;
    } catch (error) {
      console.error('Error uploading document:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Error uploading document: ${error.message}`
      }]);
      return null;
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !sessionId || isProcessing) return;

    let currentDocumentInfo = [...documentInfo];
    let uploadedDocumentInfo = null;
    let isInterruptResponse = false;
    
    // Check if the last message was an interrupt (clarification request)
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      // Check if the last message was from the assistant and had interrupt data
      if (lastMessage.role === 'assistant' && lastMessage.isInterrupt) {
        isInterruptResponse = true;
      }
    }
    
    // Upload file if selected
    if (selectedFile) {
      uploadedDocumentInfo = await handleFileUpload();
      if (uploadedDocumentInfo) {
        // Update the documentInfo state
        setDocumentInfo(prev => [...prev, uploadedDocumentInfo]);
        currentDocumentInfo = [...documentInfo, uploadedDocumentInfo];
        // Add system message to indicate successful upload
        setMessages(prev => [...prev, {
          role: 'system',
          content: `Document "${uploadedDocumentInfo.file_name}" uploaded successfully`
        }]);
      }
    }

    // Add user message to chat
    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsProcessing(true);

    try {
      if (isInterruptResponse) {
        // Send interrupt response to resume endpoint
        console.log('Sending interrupt response to resume endpoint');
        const requestData = {
          message: inputMessage,
          session_id: sessionId,
          document_info: currentDocumentInfo
        };
        
        const response = await axios.post(`${API_BASE}/chat/resume`, requestData);

        // Check if response contains another interrupt
        if (response.data.interrupt && response.data.interrupt.type === "clarification_request") {
          // Handle interrupt - add the clarification question to chat with interrupt flag
          const interruptMessage = { 
            role: 'assistant', 
            content: response.data.interrupt.question,
            isInterrupt: true  // Flag to indicate this is an interrupt message
          };
          setMessages(prev => [...prev, interruptMessage]);
          // Keep the forced agent active for when user responds
          // Don't reset forced agent yet
        } else {
          // Normal response - add assistant message to chat
          const assistantMessage = { role: 'assistant', content: response.data.response };
          setMessages(prev => [...prev, assistantMessage]);
          
          // Update document info if provided
          if (response.data.document_info && Array.isArray(response.data.document_info)) {
            setDocumentInfo(response.data.document_info);
          }
          
          // Reset forced agent after completing interrupt
          setForcedAgent(null);
        }
      } else {
        // Send normal message to chat endpoint
        // Prepare request data with forced agent if set
        const requestData = {
          message: inputMessage,
          session_id: sessionId,
          document_info: currentDocumentInfo
        };
        
        // Add forced agent if set
        if (forcedAgent) {
          requestData.user_forced_agent = forcedAgent;
          console.log('Sending forced agent:', forcedAgent);
          console.log('Sending document info:', currentDocumentInfo);
        }

        const response = await axios.post(`${API_BASE}/chat`, requestData);

        // Check if response contains an interrupt
        if (response.data.interrupt && response.data.interrupt.type === "clarification_request") {
          // Handle interrupt - add the clarification question to chat with interrupt flag
          const interruptMessage = { 
            role: 'assistant', 
            content: response.data.interrupt.question,
            isInterrupt: true  // Flag to indicate this is an interrupt message
          };
          setMessages(prev => [...prev, interruptMessage]);
          // Keep the forced agent active for when user responds
          // Don't reset forced agent yet
        } else {
          // Normal response - add assistant message to chat
          const assistantMessage = { role: 'assistant', content: response.data.response };
          setMessages(prev => [...prev, assistantMessage]);
          
          // Update document info if provided
          if (response.data.document_info && Array.isArray(response.data.document_info)) {
            setDocumentInfo(response.data.document_info);
          }
          
          // Reset forced agent after use for normal responses
          setForcedAgent(null);
        }
      }
    } catch (error) {
      const errorMessage = { 
        role: 'assistant', 
        content: `Error: ${error.response?.data?.detail || error.message}` 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      // Clear selected file after processing
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleUploadFileOption = () => {
    console.log('Upload file option clicked');
    setShowAttachmentOptions(false);
    // Simply click the hidden file input
    if (fileInputRef.current) {
      console.log('Triggering file input click');
      fileInputRef.current.click();
    }
  };

  const toggleAttachmentOptions = (event) => {
    event.stopPropagation();
    if (isProcessing) return;
    console.log('Toggling attachment options');
    setShowAttachmentOptions(prev => {
      const newValue = !prev;
      console.log('Setting showAttachmentOptions to:', newValue);
      return newValue;
    });
  };

  const handleWebSearch = () => {
    console.log('Web search option clicked');
    setShowAttachmentOptions(false);
    // Set forced agent to web search 
    setForcedAgent('web_search_agent');
    // Add system message to indicate web search mode
    setMessages(prev => [...prev, {
      role: 'system',
      content: '🔍 Web search mode activated. Your next message will be processed with web search.'
    }]);
  };

  const handleDocumentAnalysis = () => {
    console.log('Document analysis option clicked');
    setShowAttachmentOptions(false);
    setForcedAgent('rag_agent');
    setMessages(prev => [...prev, {
      role: 'system',
      content: '📄 Document analysis mode activated. Your next message will be processed with document analysis.'
    }]);
  };

  const handleImageAnalysis = () => {
    console.log('Image analysis option clicked');
    setShowAttachmentOptions(false);
    setForcedAgent('image_analysis_agent');
    setMessages(prev => [...prev, {
      role: 'system',
      content: '🖼️ Image analysis mode activated. Your next message will be processed with image analysis.'
    }]);
  };

  const handleDeepResearch = () => {
    console.log('Deep research option clicked');
    setShowAttachmentOptions(false);
    setForcedAgent('deep_research_agent');
    setMessages(prev => [...prev, {
      role: 'system',
      content: '🔬 Deep research mode activated. Your next message will be processed with comprehensive research analysis.'
    }]);
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 Agentic Assistant</h1>
        <button onClick={createNewSession} className="new-session-btn">
          New Session
        </button>
      </header>

      <div className="main-container">
        {/* Add conversation history sidebar */}
        <ConversationHistory 
          onConversationSelect={loadConversation} 
          currentConversationId={sessionId}
        />
        
        <div className="chat-container">
          <div className="messages">
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}
            {isProcessing && <ChatMessage isProcessing={true} />}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            {/* File preview - only show when file is selected */}
            {selectedFile && (
              <div className="file-preview">
                <span className="file-name">📎 {selectedFile.name}</span>
                <button 
                  onClick={removeSelectedFile}
                  className="remove-file-btn"
                  disabled={isProcessing}
                >
                  ✕
                </button>
              </div>
            )}
            
            {/* Active mode indicator */}
            {forcedAgent && (
              <div className="mode-indicator">
                {forcedAgent === 'web_search_agent' && '🔍 Web Search Mode'}
                {forcedAgent === 'rag_agent' && '📄 Document Analysis Mode'}
                {forcedAgent === 'image_analysis_agent' && '🖼️ Image Analysis Mode'}
                {forcedAgent === 'deep_research_agent' && '🔬 Deep Research Mode'}
                <button 
                  onClick={() => setForcedAgent(null)}
                  className="clear-mode-btn"
                >
                  ✕
                </button>
              </div>
            )}

            <div className="text-input-container">
              <div className="text-input-wrapper">
                <button 
                  ref={attachmentButtonRef}
                  onClick={toggleAttachmentOptions}
                  className={`attachment-btn ${forcedAgent ? 'active-mode' : ''}`}
                  disabled={isProcessing}
                >
                  {forcedAgent === 'deep_research_agent' ? '🔬' : (forcedAgent ? '✓' : '+')}
                </button>
                
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Message Agentic Assistant..."
                  disabled={isProcessing}
                  rows="1"
                  className="text-input"
                />
                
                <button 
                  onClick={handleSendMessage} 
                  disabled={!inputMessage.trim() || isProcessing}
                  className="send-btn"
                >
                  ↵
                </button>
              </div>

              {/* Attachment options dropdown */}
              {showAttachmentOptions && (
                <div className="attachment-options">
                  <div className="option" onClick={(e) => { e.stopPropagation(); handleUploadFileOption(); }}>
                    <span className="option-icon">📎</span>
                    <span>Upload file</span>
                  </div>
                  <div className="option" onClick={(e) => { e.stopPropagation(); handleWebSearch(); }}>
                    <span className="option-icon">🔍</span>
                    <span>Web search</span>
                  </div>
                  <div className="option" onClick={(e) => { e.stopPropagation(); handleDocumentAnalysis(); }}>
                    <span className="option-icon">📄</span>
                    <span>Document analysis</span>
                  </div>
                  <div className="option" onClick={(e) => { e.stopPropagation(); handleImageAnalysis(); }}>
                    <span className="option-icon">🖼️</span>
                    <span>Image analysis</span>
                  </div>
                  <div className="option" onClick={(e) => { e.stopPropagation(); handleDeepResearch(); }}>
                    <span className="option-icon">🔬</span>
                    <span>Deep research</span>
                  </div>
                </div>
              )}

              {/* Hidden file input */}
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange} 
                className="file-input"
              />
            </div>
          </div>
        </div>

        <div className="sidebar">
          <DocumentList documents={documentInfo} />
        </div>
      </div>
    </div>
  );
}

export default App;