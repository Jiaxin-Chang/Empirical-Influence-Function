import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

/** Defaults: UI 5275 → API 8765. Override with ANNOTATION_VIEWER_PORT / ANNOTATION_API_PORT. */
const ANNOTATION_VIEWER_PORT = Number(process.env.ANNOTATION_VIEWER_PORT || 5275)
const ANNOTATION_API_PORT = Number(process.env.ANNOTATION_API_PORT || 8765)

export default defineConfig({
  plugins: [react()],
  server: {
    port: ANNOTATION_VIEWER_PORT,
    strictPort: true,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${ANNOTATION_API_PORT}`,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: ANNOTATION_VIEWER_PORT,
    strictPort: true,
  },
})
