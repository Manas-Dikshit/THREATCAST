import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Global port contract: frontend on 5173, backend API on :8000 under /api/v1.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
