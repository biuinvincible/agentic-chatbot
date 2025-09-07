import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

// Add a performance monitoring wrapper
const AppWithPerformance = () => {
  return (
    <React.StrictMode>
      <App />
    </React.StrictMode>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <AppWithPerformance />
)