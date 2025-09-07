import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

const ConversationHistory = ({ onConversationSelect, currentConversationId }) => {
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/conversations`);
      setConversations(response.data);
    } catch (error) {
      console.error('Error fetching conversations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleConversationClick = async (conversationId) => {
    if (collapsed) return; // Don't handle clicks when collapsed
    try {
      const response = await axios.get(`${API_BASE}/conversations/${conversationId}`);
      onConversationSelect(response.data);
    } catch (error) {
      console.error('Error loading conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    if (collapsed) return; // Don't handle when collapsed
    try {
      // Create a new session in the backend first
      const sessionResponse = await axios.post(`${API_BASE}/session`);
      const newSessionId = sessionResponse.data.session_id;
      
      // Create a conversation record with the new session ID
      const response = await axios.post(`${API_BASE}/conversations`, {
        title: "New Conversation",
        id: newSessionId
      });
      
      // Refresh the conversation list
      fetchConversations();
      // Load the new conversation
      handleConversationClick(response.data.id);
    } catch (error) {
      console.error('Error creating new conversation:', error);
    }
  };

  const handleDeleteConversation = async (conversationId, event) => {
    if (collapsed) return; // Don't handle when collapsed
    // Prevent the click from propagating to the parent element
    event.stopPropagation();
    
    // Confirm deletion
    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return;
    }
    
    try {
      await axios.delete(`${API_BASE}/conversations/${conversationId}`);
      // Refresh the conversation list
      fetchConversations();
      
      // If we just deleted the currently active conversation, create a new one
      if (conversationId === currentConversationId) {
        handleNewConversation();
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
      alert('Error deleting conversation');
    }
  };

  const toggleCollapse = () => {
    setCollapsed(!collapsed);
  };

  if (loading) {
    return (
      <div className={`conversation-sidebar ${collapsed ? 'collapsed' : ''}`}>
        <button className="slider-toggle" onClick={toggleCollapse}>
          {/* Content is handled by CSS ::after pseudo-element */}
        </button>
        {!collapsed && <div className="conversation-history">Loading conversations...</div>}
      </div>
    );
  }

  return (
    <div className={`conversation-sidebar ${collapsed ? 'collapsed' : ''}`}>
      <button className="slider-toggle" onClick={toggleCollapse}>
        {/* Content is handled by CSS ::after pseudo-element */}
      </button>
      {!collapsed && (
        <>
          <div className="conversation-header">
            <h3>Conversations</h3>
            <button onClick={handleNewConversation} className="new-conversation-btn">
              +
            </button>
          </div>
          <div className="conversation-list">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`conversation-item ${
                  conversation.id === currentConversationId ? 'active' : ''
                }`}
                onClick={() => handleConversationClick(conversation.id)}
              >
                <div className="conversation-title">{conversation.title}</div>
                <div className="conversation-meta">
                  <div className="conversation-date">
                    {new Date(conversation.updated_at).toLocaleDateString()}
                  </div>
                  <button 
                    className="delete-conversation-btn"
                    onClick={(e) => handleDeleteConversation(conversation.id, e)}
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
            {conversations.length === 0 && (
              <div className="no-conversations">No conversations yet</div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default ConversationHistory;