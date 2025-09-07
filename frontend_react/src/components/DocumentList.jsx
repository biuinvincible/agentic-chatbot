import React from 'react';

const DocumentList = ({ documents }) => {
  if (!documents || documents.length === 0) {
    return (
      <div className="document-list">
        <h3>Processed Documents</h3>
        <p className="no-documents">No documents processed yet</p>
      </div>
    );
  }

  return (
    <div className="document-list">
      <h3>Processed Documents ({documents.length})</h3>
      <ul>
        {documents.map((doc, index) => (
          <li key={index} className="document-item">
            <div className="document-name">
              <strong>{doc.file_name}</strong>
            </div>
            <div className="document-details">
              <span className="chunk-count">{doc.chunk_count} chunks</span>
              <span className={`status ${doc.processing_complete ? 'ready' : 'processing'}`}>
                {doc.processing_complete ? '✅ Ready' : '⏳ Processing'}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default DocumentList;