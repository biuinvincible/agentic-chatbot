# Agentic Assistant - React Frontend

This is a modern React-based frontend for the Agentic Assistant chatbot. It provides a user-friendly interface with enhanced features including user control over agents and enhanced context processing.

## Features

- **Modern UI/UX Design**: Clean, gradient-based interface with responsive layout
- **Real-time Chat Interface**: Smooth messaging experience with auto-scrolling
- **Drag-and-Drop Document Upload**: Intuitive file uploading with visual feedback
- **Markdown Rendering**: Proper formatting of assistant responses with syntax highlighting
- **Session Management**: Automatic session creation with manual reset option
- **Document Processing Status**: Visual indicators for document processing progress
- **User Agent Control**: Option to manually select which agent to use for each step
- **Enhanced Context Processing**: Toggle for advanced context understanding
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices
- **Typing Indicators**: Visual feedback when the assistant is processing requests

## Prerequisites

Before you begin, ensure you have the following installed:
- Node.js (version 14 or higher)
- npm (usually comes with Node.js)
- Python 3.8 or higher (for the backend)
- The Agentic Assistant backend running on port 8000

## Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend_react
   ```

2. Install the required dependencies:
   ```bash
   npm install
   ```

## Running the Application

1. First, make sure the backend API is running:
   ```bash
   # In the main project directory
   python backend_api.py
   ```
   The backend should be accessible at http://localhost:8000

2. Start the React development server:
   ```bash
   # In the frontend_react directory
   npm run dev
   ```
   The frontend will be accessible at http://localhost:3000 (or another port if 3000 is in use)

## Building for Production

To create a production build:
```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
frontend_react/
├── src/
│   ├── components/
│   │   ├── ChatMessage.jsx      # Component for displaying chat messages
│   │   └── DocumentList.jsx     # Component for displaying processed documents
│   ├── App.jsx                  # Main application component
│   ├── App.css                  # Application styles
│   ├── main.jsx                 # Entry point
│   └── index.css                # Global styles
├── index.html                   # HTML template
├── vite.config.js              # Vite configuration
├── package.json                # Project dependencies and scripts
└── README.md                   # This file
```

## Usage

1. Once both the backend and frontend are running, open your browser to http://localhost:3000
2. You'll see a new session has been automatically created
3. Type your message in the text input at the bottom and press Enter or click Send
4. To upload a document:
   - Drag and drop a file into the upload area
   - Or click the upload area to select a file using the file browser
5. Supported document formats: PDF, DOCX, TXT, CSV, XLSX, PY, JS, HTML, MD
6. Use the "Force Next Agent" dropdown to manually select which agent to use for the next step
7. Toggle "Enhanced Context Processing" for advanced context understanding
8. Click "New Session" to start a fresh conversation

## API Integration

The frontend communicates with the backend API through a proxy configuration:
- Frontend makes requests to `/api/*`
- Vite proxies these requests to `http://localhost:8000/*`

## Customization

### Changing the API Base URL

You can modify the API base URL by editing the `.env` file:
```
VITE_API_BASE=http://localhost:8000
```

### Styling

All styles are contained in `src/App.css`. You can modify colors, spacing, and other visual elements by editing this file.

## Components

### ChatMessage Component
Handles the display of individual chat messages with different styling for user, assistant, and system messages. Supports Markdown rendering for assistant responses.

### DocumentList Component
Displays processed documents with status indicators and chunk information.

## Development

### Adding New Features

1. Create new components in the `src/components` directory
2. Import and use them in `App.jsx`
3. Add any necessary styling to `App.css`

### Component Structure

```jsx
// Example component structure
const MyComponent = ({ prop1, prop2 }) => {
  return (
    <div className="my-component">
      {/* Component content */}
    </div>
  );
};

export default MyComponent;
```

## Dependencies

- React 18: Core UI library
- React DOM: DOM-specific methods for React
- Axios: HTTP client for API requests
- React Markdown: Markdown rendering component
- Remark GFM: GitHub Flavored Markdown support
- Vite: Build tool and development server

## Troubleshooting

### "Port already in use" error
If you see a message that port 3000 is in use, the development server will automatically try another port. Check the terminal output for the actual port being used.

### "Network Error" when sending messages
Make sure the backend API is running on port 8000. You should see a response when visiting http://localhost:8000 in your browser.

### File upload not working
Ensure the backend API is running and that you're uploading a supported file type.

### Build Issues
If you encounter build errors:
1. Ensure all dependencies are installed: `npm install`
2. Check that `index.html` exists in the root directory
3. Verify there are no syntax errors in the React components

## Deployment

### Production Build
```bash
npm run build
```

The build output will be in the `dist` directory, which can be served by any static file server.

### Environment Variables
For production, you may need to set the API base URL:
```
VITE_API_BASE=https://your-api-domain.com
```

## Browser Support

The application supports modern browsers:
- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

## Accessibility

The interface follows accessibility best practices:
- Proper semantic HTML structure
- Keyboard navigation support
- Sufficient color contrast
- Focus indicators for interactive elements

## Performance

The application is optimized for performance:
- Code splitting for faster initial loads
- Efficient React component rendering
- Lazy loading of non-critical resources
- Optimized asset delivery

## Security

The frontend implements several security best practices:
- Content Security Policy through Vite
- Secure API communication
- Input validation and sanitization
- Protected against common web vulnerabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Agentic Assistant and follows the same license as the main project.