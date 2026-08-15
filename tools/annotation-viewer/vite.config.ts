import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

/** Fixed port for annotation-viewer. Fail if busy — do not silently bump to 5175+. */
const ANNOTATION_VIEWER_PORT = 5174

export default defineConfig({
  plugins: [react()],
  server: {
    port: ANNOTATION_VIEWER_PORT,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: ANNOTATION_VIEWER_PORT,
    strictPort: true,
  },
})
