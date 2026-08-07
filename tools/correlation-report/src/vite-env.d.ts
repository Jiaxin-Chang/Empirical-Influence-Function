/// <reference types="vite/client" />
declare module '*?raw' {
  const content: string;
  export default content;
}

interface ImportMetaEnv {
  readonly VITE_ANNOTATION_VIEWER_URL?: string
  readonly VITE_TRAIN_GT_JSONL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
